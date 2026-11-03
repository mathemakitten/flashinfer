# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Diagnostic single-CTA FC2 scheduler with static and hardware CLC claims.

Both modes use the same FC2-only, order-independent decoder. Inputs and
readiness counters must be populated before launch. No FC1 execution,
communication, or inter-block producer dependency is supported.
"""
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cutlass_dsl import Int32, extract_mlir_values, new_from_mlir_values
from moe_nvfp4_swapab.fc1_fc2_fuse_sched import MoEFusedFc12PersistentTileScheduler as Base


class Fc2ClcScheduler:
    @staticmethod
    def make_storage_struct(params, ext, **kwargs):
        base_storage = Base.make_storage_struct(params, ext)

        @cute.struct
        class Storage:
            base: base_storage
            clc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            clc_response: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4], 16]
        return Storage

    @staticmethod
    def create(params, block_idx, grid_dim, sched_storage, num_consumer_threads, ext, use_clc=True, num_tiles=0):
        result = Fc2ClcScheduler()
        result.base = Base.create(params, block_idx, grid_dim, sched_storage.base,
                                  num_consumer_threads, ext)
        result.use_clc = use_clc
        result.num_tiles = num_tiles
        result.stride = Int32(grid_dim[2])
        result.response = sched_storage.clc_response.data_ptr()
        result.pipe = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=sched_storage.clc_mbar.data_ptr(), num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            tx_count=16, cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            defer_sync=True)
        result.producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.ProducerConsumer, 1)
        result.consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        result.first = True
        result.initial_id = Int32(block_idx[2])
        return result

    @property
    def current_work(self):
        return self.base.current_work

    def internal_init(self, warp_idx, sched_warp_id):
        # Both pipelines were initialized collectively by create(). No claim
        # until the kernel's pipeline_init_wait has completed.
        pass

    def make_consumer(self):
        return self.base.make_consumer()

    @cute.jit
    def gen_next_work(self):
        tile_id = self.initial_id
        valid = Int32(1)
        if cutlass.const_expr(self.use_clc and not self.first):
            self.pipe.producer_acquire(self.producer)
            barrier = self.pipe.producer_get_barrier(self.producer)
            with cute.arch.elect_one():
                cute.arch.issue_clc_query(barrier, self.response)
            self.producer.advance()
            self.pipe.consumer_wait(self.consumer)
            _, _, tile_id, valid = cute.arch.clc_response(self.response)
            cute.arch.fence_acq_rel_cta()
            self.pipe.consumer_release(self.consumer)
            self.consumer.advance()
        self.first = False
        if cutlass.const_expr(not self.use_clc):
            valid = Int32(tile_id < Int32(self.num_tiles))
            self.initial_id = self.initial_id + self.stride
        self.decode_fc2(tile_id, valid)

    @cute.jit
    def decode_fc2(self, tile_id, valid):
        # Reset the search on each claim; CLC responses need not be ordered.
        params = self.base.params
        expert = Int32(0)
        previous = Int32(0)
        data_row = Int32(0)
        sf_row = Int32(0)
        block_row = Int32(0)
        local_id = tile_id
        tokens = Int32(0)
        found = Int32(0)
        nblocks = self.base._num_fc2_hidden_blocks
        m = params.cta_tile_shape_mnk[0]
        while expert < params.expert_cnt and found == Int32(0) and valid != Int32(0):
            end = params.expert_token_prefix_sum[expert]
            tokens = end - previous
            blocks = (tokens + Int32(m - 1)) // Int32(m)
            work_count = blocks * nblocks
            if local_id < work_count:
                found = Int32(1)
            else:
                local_id = local_id - work_count
                data_row = data_row + ((tokens + Int32(params.token_padding_block - 1)) // Int32(params.token_padding_block)) * Int32(params.token_padding_block)
                sf_row = sf_row + ((tokens + Int32(params.sf_padding_block - 1)) // Int32(params.sf_padding_block)) * Int32(params.sf_padding_block)
                block_row = block_row + blocks
                previous = end
                expert = expert + Int32(1)
        tile_m = local_id // nblocks
        tile_n = local_id % nblocks
        count = cutlass.min(cutlass.max(tokens - tile_m * Int32(m), Int32(0)), Int32(m))
        if found == Int32(0):
            expert = Int32(-1)
        work = self.base._ext.WorkTileInfo(
            expert_idx=expert, tile_m_idx=tile_m, tile_n_idx=tile_n,
            cumulative_data_physical_row=data_row, cumulative_sf_physical_row=sf_row,
            cumulative_token_block_count=block_row,
            valid_tokens_in_cta_cluster_tile=(count << Int32(16)) | count,
            phase_and_peek=Int32(2), fc1_counter_index=tile_m)
        self.base.current_work = self.base._ext.enrich_work_tile_info(work)

    @cute.jit
    def publish_work(self):
        self.base.publish_work()

    @cute.jit
    def produce_tail(self):
        self.base.produce_tail()
        if cutlass.const_expr(self.use_clc):
            self.pipe.producer_tail(self.producer)

    def __extract_mlir_values__(self):
        values = []
        for name in ('base', 'response', 'producer', 'consumer', 'initial_id', 'stride'):
            values.extend(extract_mlir_values(getattr(self, name)))
        return values

    def __new_from_mlir_values__(self, values):
        result = Fc2ClcScheduler()
        index = 0
        for name in ('base', 'response', 'producer', 'consumer', 'initial_id', 'stride'):
            original = getattr(self, name)
            count = len(extract_mlir_values(original))
            setattr(result, name, new_from_mlir_values(original, values[index:index+count]))
            index += count
        result.pipe = self.pipe
        result.first = self.first
        result.use_clc = self.use_clc
        result.num_tiles = self.num_tiles
        return result
