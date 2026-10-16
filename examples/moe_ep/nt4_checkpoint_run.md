# Running the NT4 checkpoint-layer gate

## Implementation status

The local implementation consists of:

- `nt4_checkpoint_capture.py`: opt-in wrapper around the working MCore inference
  command; exports a loaded layer, replays the live NVLS/FlashInfer baseline,
  checks MegaMoE, and optionally times both.
- `nt4_checkpoint_artifact.py`: versioned raw-byte artifact and checksum schema;
  inverse cuBLAS scale-layout transform with a byte round-trip assertion.
- `nt4_checkpoint_replay.py`: replay saved samples without reloading MCore;
  quantization-aware reference using only selected expert broadcasts.

CPU tests and syntax checks are available locally. **The new checkpoint adapter,
reference, live dual-runtime execution and timing have not been run on GPUs.**
The earlier synthetic MegaMoE tests passed; those do not qualify these new paths.

The local MCore checkout includes the FlashInfer ClampedRelu2 integration.
The wrapper modifies no production MCore files and has no numerical
fallback: missing private adapter methods, wrong precision, or an unsupported
baseline fails explicitly. Sync all `nt4*.py` files in this examples directory
and the existing MegaMoE prototype, including its owned kernel/shim files.

## Executable entry point

`examples/inference/advanced/vlm_dynamic_inference.py` in current MCore is a
helper module; running it directly only imports definitions and exits. It does
not load a model or generate tokens. Use the new FlashInfer-side
`nt4_checkpoint_generate.py` driver after the capture wrapper's standalone `--`.
The driver reuses MCore's VLM/GPT detection and engine builder, then its offline
request loop. All original model and prompt arguments follow the driver path.
It supports text prompts only. For VLM it applies MCore's standard TE-to-MCore
MXFP8 inference conversion before building the engine, as the GPT path does;
this can requantize TE storage once. Export preserves the resulting loaded
inference bytes shared by both compared backends, not necessarily raw checkpoint
TE bytes. This driver is syntax-checked locally but still needs GPU validation.

For the current `helenn-fused-tanh-sqrelu` MCore branch, use
`--transformer-impl inference_optimized` for GPT checkpoint inference and omit
`--vlm-language-transformer-impl`, `--inference-flashinfer-bounded-rows`, and
`--inference-allow-gtp-padding`: that branch does not register these options.
The separate VLM language-model implementation selector belonged to the earlier
checkout. MIMO VLM loading from #7839 also uses the standard
`--transformer-impl inference_optimized` selector for its language stack while
keeping separate TE vision specs. The driver recognizes this path and applies
the MXFP8 inference conversion after loading. Other VLM loaders still need their
own language inference integration. Removing a padding option does not add support for
checkpoint shapes that require that feature.

## First run: one captured sample, eager, no timing

For diverse text requests, the generation driver accepts MCore's native
`--prompt-file` JSONL format: one object with a string `text` field per line.
Replace `--prompts ...` in the model arguments (after `--`) with:

```bash
--prompt-file /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/helenn/research/routing/prompts_dapo50.jsonl \
--prompt-file-num-truncate 50 \
--incoming-requests-per-sec -1
```

This makes all requests available together; engine token/KV limits still control
admission. For an initial diverse-input run, omit `--sweep-counts` and set wrapper
`--samples 3 --benchmark --graph` to measure actual captured steps first. Existing
skip-call and row-count filters do not guarantee decode-only steps; inspect
active-token counts and routing diversity before labeling the result decode.

Source the cache setup before launching Python on each new node/container:

```bash
source "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_checkpoint_setup.sh"
```

This sets `FLASHINFER_WORKSPACE_BASE=/lustre/fsw/portfolios/nemotron/users/helenn`,
placing the FlashInfer cache under that directory's `.cache/flashinfer/` instead
of the container's home directory. It does not migrate existing container-local
builds or change an already running process.

For a source checkout whose JIT-provider API differs from published cache wheels,
uninstall the incompatible cache and opt into source compilation for this driver:

```bash
"$PY" -m pip uninstall -y flashinfer-jit-cache
export NT4_ALLOW_SOURCE_JIT=1
```

This bypasses only MCore's precompiled-cache presence check during model
construction, restores the check afterward, and logs `source_jit_enabled`.
FlashInfer's compilation and kernel validation remain active. First launch may
take several minutes; compilation is outside the harness's timed iterations.
Version-check bypasses do not fix an incompatible cache-provider API.

Keep the user's existing NT4 command, checkpoint step 30000, EP4/TP1, tokenizer,
precision, FlashInfer/NVLS options and graph-disabled model configuration.
Set `FLASHINFER_CHECKOUT` and `MCORE_CHECKOUT` to the actual remote checkout
directories. Set `NT4_ARTIFACT` to a **new**, writable directory on shared storage
visible to all four ranks. The wrapper refuses an existing directory.

In the working command, replace only this script entry:

```bash
examples/inference/advanced/gpt_dynamic_inference_vlm.py
```

with this block, leaving all original model arguments after it:

```bash
"$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_checkpoint_capture.py" \
  --mcore-root "$MCORE_CHECKOUT" \
  --artifact "$NT4_ARTIFACT" \
  --layer 0 --samples 1 --skip-calls 1 --max-local-tokens 64 \
  -- "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_checkpoint_generate.py"
```

Keep `torch.distributed.run --standalone --nproc-per-node=4` and the existing
`/tmp/flashinfer_rank_bootstrap.py` wrapper, if used, before this replacement.
Set `PYTHONPATH` to the modified FlashInfer checkout as for synthetic bring-up.

`--layer 0` selects the first MXFP8 MoE layer and prints its actual layer number;
subsequent runs can pin that number. Selection checks the required NT4 shape,
activation, EP configuration, FlashInfer backend and NVLS dispatcher. It rejects
a BF16 layer chosen explicitly. The artifact preserves the MXFP8 representation
produced by the normal model loader; it never requantizes those loaded weights.

`--skip-calls 1` skips the first invocation of that layer and
`--max-local-tokens 64` filters out larger steps. These are filters, **not an
automatic prefill/decode classifier**. A result is labelled
`captured_step_unclassified` until its call is mapped to a decode step in the
original run. The command errors if it finishes without the requested captures.
It resumes normal inference after each check, returning the saved original
output to the model. This run's end-to-end latency is not meaningful.

Look for:

1. `selected_layer`, then `capture_begin` with all four token counts.
2. `adapter_vs_original_model_call`: the replayed live baseline reproduces the
   original model's routed output at the latent boundary.
3. `live_baseline_vs_capture`, exact repeated baseline and MegaMoE checks.
4. `captured_baseline_vs_reference`, `megamoe_vs_baseline`, and
   `correctness: PASS` on every rank.
5. `capture_complete` and `manifest.json` with `capture_complete: true`.

The wrapper's watchdog defaults to 7200 seconds, including model loading, first-use
JIT compilation, and reference work. Use `--timeout-seconds 7200` when overriding
an older command's explicit timeout. Exceptions
are printed before process exit so peers cannot hide a failure in cleanup.

## Artifacts and standalone replay

### GB300 scheduling experiments

`--preset group256` tests groups 256/384/512/768 and Group 256 combined with
dispatch flag batches 1/2/8. Default and Group 256 controls bracket the pass
(20 jobs across two artifacts). Configuration changes affect only MegaMoE;
correctness gates and routing-weight placement remain unchanged.

For a subsequent live CUTLASS comparison, the capture wrapper now accepts
`--knobs-json '{"group_hint":256}'` before its standalone `--`, alongside
`--benchmark --graph`. Substitute the measured winner's explicit knobs.
The selected knobs are preserved in provenance and per-sample results. Leave
the JSON option out to retain the original default; tuning never changes it
automatically. Use the diverse prompt file and inspect actual active rows and
expert counts before comparing runs.

After the initial sweep, `--preset refine` tests group sizes 16, 32, 64, 96,
128, 192, and 256. It also tests Group 128 with dispatch flag batches 1/8 and
epilogue batches (2,1), (1,2), (2,2), changing one setting at a time from Group
128. Both default and Group 128 controls bracket this pass. Across two artifacts
this launches 30 isolated jobs; prior trial runtimes suggest roughly 12 minutes
plus any new compilation. This estimate is not a timeout guarantee.

`nt4_gb300_tune.py` launches isolated four-rank replay jobs on SM103 GB300,
using saved artifacts only. Supply the diverse artifact first, and the narrow
routing control second with repeated `--artifact`. It tests static/atomic work
assignment, scheduler grouping, dispatch notification batching, FC1/FC2 epilogue
notification batching, and dispatch-warp return scheduling. The MMA tile remains
256x256x128, two CTAs; smaller tiles need a separate NT4 epilogue/layout audit.

Every candidate passes full reference/captured-baseline correctness before
20 warmup calls and five rounds of 100 graph timings. Default controls bracket
the sweep to expose clock/load drift. These are MegaMoE-only timings, not a new
live CUTLASS comparison. No winning candidate is promoted automatically.
Trials save logs plus `results.json` after every case and stop on failure.
`--candidates atomic,group32,group128` restricts the first run; omit it to include
flag1, flag8, epilogue2, dispatch_return, and atomic_group128 experiments too.

This is an experimental tuning harness, CPU-tested only. It requires remote GPU
measurements before claiming improvement or selecting a production configuration.
Once a candidate wins reproducibly on the diverse case, check the control and
profile that candidate with replay's `--knobs-json` and `--profile-dir` options
to confirm that main-kernel time improves. Recompare with the live baseline
before publishing an end-to-end speedup. First use of a new configuration may
compile a specialization; all runs should use the persistent Lustre cache.

### Profile saved routing cases

Standalone replay accepts `--profile-dir NEW_DIRECTORY --profile-iterations 10`.
After correctness, 20 warmup calls, and optional graph capture (`--graph`), it
records only the warmed execution window with PyTorch's CPU/CUDA profiler.
Outputs are Chrome/Perfetto traces at `sample0/megamoe-rankN.json`. Use distinct
directories for the narrow-routing and diverse-routing artifacts. No full NT4
model load is needed, but reference validation still runs before profiling.

Inspect kernel durations, staging/quantization, the persistent MegaMoE kernel,
top-k reduction, launch gaps, and rank-to-rank differences. This trace cannot
separate internal phases of the persistent kernel or establish memory-bandwidth
utilization; Nsight Compute would be a subsequent targeted step. Profiler timings
are diagnostic; retain unprofiled benchmark numbers for performance claims.
Cases with different token counts do not isolate routing diversity causally.

The live capture wrapper accepts the same flags and also writes
`sample0/baseline-rankN.json`. That option requires a model load because the saved
artifact contains baseline outputs, not an executable MCore dispatcher. When
`--benchmark` is also enabled, timing runs before profiling. Do not enable
`--trace` for these runs. The token census helper additionally reports 512 expert
assignment counts and four expert-owner totals for checking routing skew.

### Active-token census and one-load sweep

Inspect an existing saved sample without loading MCore or initializing GPUs:

```bash
"$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_token_sweep.py" --artifact "$NT4_ARTIFACT"
```

`buffer_rows` includes padding; `active_routed_rows` counts rows with any valid
expert ID. `positive_weight_rows` separately counts rows with a positive valid
weight. EP copies may represent replicated tokens, so their sum is not a count
of unique requests. New live captures also log the engine's pre-SP real-token
count alongside these route-based counts after applying the padding mask.

To measure 1–64 active rows per rank with fixed 64-row buffers, add these capture
wrapper flags before the standalone `--` in the full model command:

```bash
--samples 1 --sweep-counts 1,2,4,8,16,32,64 --sweep-buffer-rows 64 \
--benchmark --graph --routing-weight-placement after_fc2 \
--warmup 20 --iterations 100 --rounds 5 --timeout-seconds 7200
```

Use a new artifact directory. One model load and one MegaMoE workspace serve
the original sample and all sweep points. Each point repeats only active rows
from the captured EP-wide pool, preserves their routes/weights, masks unused
rows, and passes correctness before timing. These are synthetic repeated-input
measurements, not independent production routing samples. They isolate active
row count at fixed padded capacity; they do not establish a general routing
threshold. The engine count mask is disabled for explicitly masked sweep inputs
and restored afterward. `--sweep-counts` implies benchmarking.

Results are printed as `sweep_point`, saved incrementally to `sweep0.json`, and
included under each rank's `results[].sweep` in the final manifest. Inspect:

```bash
jq '.[] | {active_rows_per_rank, buffer_rows, baseline_us: .benchmark.baseline.median_us, megamoe_us: .benchmark.megamoe.median_us, speedup: .benchmark.baseline_over_megamoe}' "$NT4_ARTIFACT/sweep0.json"
```

Checkpoint capture and replay default to `--routing-weight-placement after_fc2`
to compare with MCore's CUTLASS baseline. This selects the existing weighted
top-k reducer: FC1 activation is quantized without the routing weight, FC2
outputs are rounded to BF16, then weighted and summed in FP32 with slot-ordered
FMA before the final BF16 output conversion. The contribution gate checks raw
BF16 expert outputs and exact weighted recombination. Gates are not loosened.
Use `--routing-weight-placement before_quantization` to replay the original
weighted-activation variant. The public kernel config keeps that original mode
as its default; checkpoint tools explicitly select the baseline-compatible mode.
The two placements have separate compile/workspace keys. Late weighting is
CPU-checked but awaits GPU qualification; its first launch may compile a new
MegaMoE specialization. Existing saved artifacts can be reused with
`--allow-incomplete`; no full-model load is needed for offline correctness.

For a stalled capture run, add `--trace` to the capture wrapper arguments before
`--`. This logs host entry/return around the selected layer's dispatcher stages
and experts and dumps Python stacks every 120 seconds. Host returns do not prove
GPU completion; tracing adds no CUDA synchronizations. The watchdog also dumps
all Python thread stacks before terminating. Disable tracing for timing runs.

The directory contains `weights-rankN.pt`, `sampleI-rankN.pt`, `manifest.json`,
source revisions/patches, copies of the harness and owned kernel sources, and
available GPU topology/state output. Weights and scales are canonical uint8
payloads; samples retain BF16 inputs/outputs and FP32 probabilities. SHA256
checksums cover every tensor. The manifest is written before MegaMoE validation,
then updated after success so a failed gate still leaves replayable samples.

In the same four-GPU environment, without loading the full model:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_checkpoint_replay.py" \
  --artifact "$NT4_ARTIFACT"
```

Add `--graph` for fixed-buffer MegaMoE graph replay. Use `--allow-incomplete`
only to inspect successfully saved samples from an interrupted/failed capture.
Standalone replay compares against the **saved** baseline and canonical-weight
reference; it cannot time or rerun a live MCore baseline. It validates the
artifact schema and hashes before using tensors.

The reference broadcasts each globally selected expert once in a fixed order,
reconstructs canonical values from payloads/scales, and evaluates FP32 FC1/FC2
with the weighted clamped activation and MXFP8 intermediate round trip. It uses
FP64 absolute-product sums for the v2 accumulation allowance. Diagnostic
allocations and collective reference work are outside timing.

## Follow-up correctness and load cases

After one sample passes, use a new artifact directory and `--samples 4` for
several changed model calls. Add wrapper `--graph` to qualify the MegaMoE graph
path. Full-model graph flags stay disabled. Each capture has its own graph;
standalone replay stages different samples into a fixed-capacity session but
currently recaptures a graph per sample. The previous synthetic runner supplies
the same-graph/changing-input regression coverage.

For controlled replay counts, add e.g. `--replay-counts 0,4,8,16`. This repeats
or truncates captured local rows; a rank cannot synthesize nonempty data from
an empty capture. Counts must fit the already allocated NVLS buffers. The
wrapper first qualifies the original baseline invocation, then builds and saves
a fresh baseline result for the synthetic inputs. It restores the model's
original token metadata before resuming inference.

Use `--mask-odd-slots` to mask alternating routes without renormalizing active
probabilities. These cases are labelled `resampled_capture` in provenance.
Sweep equal counts by separate runs with `--replay-counts N,N,N,N`, for
N in 1, 2, 4, 8, 16, 32, 64; include unequal and empty-rank cases separately.
This is a controlled load sweep, not a claim about actual per-rank decode
occupancy. Prefill samples should be captured and reported separately.

## Timing after correctness

Add wrapper `--benchmark` to an otherwise passing capture command, using a new
artifact directory. Defaults are 20 warmups, 100 calls per round, five rounds,
alternating baseline/MegaMoE order. Every sample must pass correctness first.

Eager is the primary comparison. Wrapper `--graph --benchmark` additionally
checks a captured baseline graph against its saved result and times both
backends under graphs. Unsupported baseline capture raises an error rather than
substituting eager timings. GPU events bracket each whole routed branch on its
completion stream. There is a barrier before each measurement batch, not inside
each invocation. Post-measurement aggregation takes the maximum rank duration
per sample, then reports median and p95. Event-instrumented batch wall time is
also reported; it includes Python/event overhead and is not pure launch latency.

`benchmark` JSON records and manifest results include all rank token counts,
capacity, graph mode, both medians/p95s, baseline/MegaMoE ratio, and peak memory.
Memory covers the whole process with the model, both backends and persistent
diagnostic allocations; it is not an isolated per-backend memory comparison.
Packing and first-launch times are separate log records. The production
invalid-slot clear remains timed; reference work and snapshots do not.

Review available `gpu-state.txt` and `gpu-topology.txt` and concurrent jobs when
interpreting variance. Save stdout/stderr as well as the artifact directory.
An isolated routed-branch improvement does not establish full-layer or
end-to-end decode benefit, and GB300 measurements do not establish a Rubin win.

## Local checks

### Independent GB300 NVLS hybrid track

The default now uses fused glue: one route-mask kernel, one slot-ordered FP32
weighted-partial kernel writing directly into the NVLS buffer, and no redundant
local BF16 top-k reduction. `--legacy-hybrid` retains the original composition
described below for A/B comparison. Direct-return EP4 behavior is unchanged.
The skipped reduction is restricted to local-only NT4 late-weighted mode and
participates in the compile key. That mode produces contribution slots only;
passing a reduced output target is rejected.

Contributions-only mode also skips invalid-slot clearing. The FP32 partial
kernel masks loads using current route IDs; invalid slots are unspecified and
may contain NaNs. Before timing, the runner poisons the contribution buffer,
checks all-padding eager/graph output is zero without clearing that poison,
then restores real routes through the same graph and checks the original output.
Direct and legacy hybrid paths retain their invalid-slot clearing.

The runner checks masks and FP32 partial sums exactly against a separate
PyTorch slot-ordered multiply/add calculation before benchmarking. This uses
separate FP32 multiply/add, not FMA; the legacy Torch sum's reduction order can
differ. Existing end-output L2 gates and eager/graph/padding checks still apply.
GPU unit coverage is in `tests/moe_ep/test_nt4_nvls_kernels.py`. CPU checks
cannot validate Triton compilation or the changed CuTe launch specialization.

Branch `helenn-nt4-gb300-nvls`, worktree `flashinfer-nt4-gb300-nvls`.
Sync this worktree to a separate remote checkout and point FLASHINFER_CHECKOUT
and PYTHONPATH at it explicitly. Do not overwrite the Rubin checkout.

```bash
export PYTHONPATH="$FLASHINFER_CHECKOUT:$MCORE_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_nvls_hybrid.py" \
  --mcore-root "$MCORE_CHECKOUT" \
  --artifact "$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901" \
  --graph --benchmark --timeout-seconds 1800
```

Repeat with the `20261012-075027` narrow capture after the first passes. Requires
the same MCore checkout's multimem collectives and PyTorch symmetric memory.
No full model load. No dependency reinstall or JIT cache deletion is required.

The prototype performs NVLS AllGatherV, rank-owned route masking, a local-only
instance of the NT4 fused expert kernel, FP32 weighted local contribution sum,
then NVLS ReduceScatterV. It never changes global `MEGA_NO_DIST` and never maps
the local kernel's rank zero to NVSHMEM PE zero: local allocations use identity
peer offsets. The original direct-return kernel still runs EP4 for comparison.

Both collectives use the fixed padded buffer sizes from the captured experiment.
The first implementation retains local MegaMoE packing/cleanup, a redundant
BF16 top-k result, and Torch FP32 partial assembly. All are timed. It is a
composition prototype, not yet a lean compute-only implementation. It does not
contain the explicit staged-return mode from the earlier experiment.

Acceptance checks: direct kernel's existing reference/contribution gate;
hybrid against saved CUTLASS output and live direct output; eager/replay/graph
equality; real → all-padding → real transitions. The hybrid does not yet have
an independent per-expert contribution gate. The FP32 rank-local reduction can
change summation order, so final outputs use the existing L2 gate. GPU validation
is pending; CPU routing tests cannot qualify the new local-only kernel launch.

The shared benchmark emits `baseline=nvls_hybrid_prototype` and
`megamoe=group512_direct`. It does not time live CUTLASS. A prototype loss includes
the known extra local work and does not rule out a lean NVLS hybrid; a prototype
win should be confirmed against live NVLS + CUTLASS before promoting anything.

To profile the validated composition, add:

```bash
--profile-dir "$NT4_LOG_DIR/nt4-nvls-profile-$(date +%Y%m%d-%H%M%S)" \
--profile-iterations 10
```

Both `sample0/graph` and `sample0/eager_annotated` contain per-rank
`baseline-rankN.json` (hybrid) and `megamoe-rankN.json` (direct) Chrome traces.
The eager trace maps launches into dispatch, route masking, quantization/staging,
local fused experts, local FP32 combine, NVLS combine and output cast ranges.
Use the graph trace to inspect GPU costs for the timed execution mode; eager
annotation/launch overhead is not the graph benchmark. `profiler_warmup` ranges
absorb profiler startup skew and must be excluded. Persistent-kernel internals
remain opaque in these traces. Profiling follows correctness, and unprofiled
timings remain the performance evidence.

### Quick GB300 explicit-return control before Rubin

The existing kernel already has an explicit protocol. With
`token_back_mode="reuse_dispatch_warps"`, FC2 writes a local output pool,
fences the stores and publishes per-expert completion counters. Return warps
wait for those counters and copy outputs into the owning rank's BF16 top-k
slots. Cross-rank barriers establish destination completion and retire/reset
workspace state before reuse. The separate weighted top-k sum is unchanged.
The ordinary `epi_warps` path writes the peer slots directly from the epilogue.

Run the two saved captures without reloading the model:

```bash
"$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_gb300_tune.py" \
  --artifact "$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901" \
  --artifact "$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-075027" \
  --preset return_control \
  --output "$NT4_LOG_DIR/nt4-return-control-$(date +%Y%m%d-%H%M%S)" \
  --timeout-seconds 1800
```

Six isolated EP4 jobs: direct, staged, direct, each on both captures. All use
group512 and late routing weighting. Each checks eager/graph equality, exact
replay, contribution validity and saved-baseline agreement before graph timing.
The default 20 warmups and 5 rounds of 100 iterations exercise workspace reuse.
Results include explicit protocol labels; any failed rank stops the experiment.
The staged variant may require a first-use JIT compilation; preserve the cache.

This is an existing-protocol qualification, not a new per-token destination
readiness implementation. Timing includes changed staging and worker assignment;
do not call the difference pure signaling overhead. Destination completion is
still barrier-based in both arms. No counted operations or vendor kernel edits
are part of this GB300 step.

On Rubin, first establish a supported compute baseline. Then compare ordinary
versus counted transfers with matched chunking, destination slots, producer
schedule and final sum on Rubin itself. Do not compare a GB300 explicit time
against a Rubin counted time to attribute a feature win, and do not remove the
retirement protocol just because delivery is counted. If a matched transport
control is needed, implement it there; the GB300 staged path is a starting point,
not proof that only a completion instruction needs replacing.

### CPU checks

```bash
python tests/moe_ep/test_nt4_checkpoint_artifact.py
python tests/moe_ep/test_nt4_megamoe_validation.py
python tests/moe_ep/test_nt4_megamoe_config.py
```

These cover CPU artifact integrity/layout and acceptance-policy behavior.
They do not test the live MCore hooks, NVSHMEM/NVLS coexistence, CUDA capture,
or the new reference broadcasts; the first GPU capture run is the next gate.

### Three-way live GB300 comparison

`nt4_three_way.py` reuses `provenance.command` from a completed EP4 capture
and launches the current capture/generate scripts. This reloads the checkpoint;
it does not treat saved CUTLASS outputs as a timed backend. Existing checkpoint,
tokenizer, and prompt paths from that command must remain accessible.

```bash
export NT4_THREE_WAY="$NT4_LOG_DIR/nt4-three-way-$(date +%Y%m%d-%H%M%S)"
"$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_three_way.py" \
  --source-artifact "$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901" \
  --mcore-root "$MCORE_CHECKOUT" --artifact "$NT4_THREE_WAY" \
  > "${NT4_THREE_WAY}.log" 2>&1
```

Add `--dry-run` to print the reconstructed command without running it.
The launcher captures one eligible step, compares all three live paths, then
sweeps 1,2,4,8,16,32,64,128,256,512 active rows per rank. Counts through 64
use 64-row buffers; larger counts use 128/256/512-row buckets. Override
`--sweep-counts` and `--sweep-bucket-rows` together to change this plan.
The live NVLS allocation is checked before starting the sweep; if too small,
increase the model's `--inference-dynamic-batching-max-tokens`. Allocation,
weight packing and compilation at bucket changes are outside timing, and
NVSHMEM stays initialized across those changes. The launcher retains
the recorded model/prompt arguments. Override `--samples` for more captured
steps. The sweep repeats captured active rows with their original routes; it
is not an independent random-routing or real request-batch benchmark.

All three candidates use the same layer weights and inputs, graph timing,
20 warmups, 100 iterations per round, and six rounds covering all candidate
orders. Durations use the slowest rank per iteration. Each point validates
the hybrid against live CUTLASS and direct MegaMoE before timing. Direct uses
group512, and the hybrid uses its fused local group512 computation.

Results use explicit keys `nvls_cutlass`, `nvls_megamoe`, `direct_megamoe`, and
`speedups_vs_nvls_cutlass`. Per-point token census records actual active rows,
unique experts and owner assignment counts. Results persist in the manifest
and `sweep0.json`. The current three-way harness requires equal padded row
counts across ranks. Uneven active counts and new routing distributions still
need separate qualification before choosing a production dispatch policy.

The launcher pins PYTHONPATH to its own FlashInfer checkout and the selected
MCore root, then logs the actual imported FlashInfer path/version before torchrun.
MCore's generic minimum-version error can also mean a swallowed ImportError;
preflight leaves that traceback visible. A fresh synced worktree may lack
`flashinfer/_build_meta.py` (git-ignored), causing `0.0.0+unknown`. Generate normal
metadata with an editable install of that checkout:

```bash
BUILD_NVEP=0 "$PY" -m pip install --no-deps --no-build-isolation -e "$FLASHINFER_CHECKOUT"
```

This does not clear JIT caches or install replacement dependencies. Do not
bypass the minimum-version check or copy version metadata from another branch.

The launcher also checks CUTLASS, CCCL and spdlog source headers before loading
the model. A newly created Git worktree can have empty submodule directories;
editable installation alone does not populate them. Include their actual
contents when syncing the GB300 worktree to a remote without Git. The local
GB300 worktree's dependency contents were copied from the original checkout
at matching gitlink revisions after the 20261012-143121 missing-header failure.
No JIT cache deletion is needed after restoring those headers.

### Real decode batches (no resampling)

For the remaining expert-231/channel-970 and expert-377/channel-1463 cases,
`nt4_checkpoint_replay.py --diagnose-prequant` enables a separate instrumented
kernel (`NT4_DEBUG_PREQUANT=1`). It writes the actual FC1 accumulator and
post-activation/pre-quantization register values for channels 970 and 1463 into
a small local debug buffer indexed by pool row. The buffer starts with NaNs so
missing stores fail explicitly. Source metadata maps the records to original
rank/token/slot. The `prequant_diagnostic` records compare against FP64 values,
report the adjacent FP8 midpoint and signed distances, reevaluate the activation
at the kernel's FC1 value, and requantize the captured activation with its actual
E8M0 scale. Check that the instrumented run reproduces the original quantized
discrepancies before attributing their cause.

This requires compiling a debug kernel variant. It does not relax v3 and may
still terminate with the expected validation assertion. Benchmarking/profiling
is prohibited with instrumentation enabled; no performance claim may use this
variant. The debug path currently supports only late (`after_fc2`) routing.
CPU scalar diagnostics and Python syntax were tested; CuTe compilation and
instrumented stores require remote GB300 verification.

Checkpoint replay now uses `nt4_stagewise_v4_provisional` by default, including
live capture. It checks every valid expert intermediate. Source metadata must
exactly match the expected rank/token/top-k assignments, excluding padding.
The first check remains exact whole-block agreement with one of the three
independent FP32/FP64 arithmetic variants from v3.

For `after_fc2` only, blocks failing that check have a boundary-aware fallback:

- Compute FC1 in FP64 from the independently quantized input and saved weights.
  Use `gamma_K * sum(abs(x*w))` for FP32 accumulation, with
  `gamma_K = K*eps/(1-K*eps)` and `eps=2^-23` (conservative for directed rounding
  as well as nearest). Inflate the FP64 reference/product sum by `gamma_2K`
  at double precision. Add an allowance for flushed subnormal intermediates.
- Propagate that interval through monotonic ReLU and tanh. The fixed factors
  1/16 and 16 are exact for normal binary floats. Use a `3*eps` relative tanh
  allowance, covering the documented non-approximate tanhf 2-ULP error relative
  to correctly rounded FP32 plus conversion rounding; include square rounding.
  See [CUDA hyperbolic function accuracy](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/05-appendices/mathematical-functions.html#hyperbolic-functions).
- Independently quantize outward-rounded interval endpoints. Their per-block
  scales must agree with each other and the stored E8M0 scale. Ambiguous scale
  selection fails closed. Each changed value must be an adjacent nonnegative
  E4M3 code relative to the FP64 variant, with its rounding midpoint inside the
  independent activation interval. A whole block must satisfy these checks.
- Keep the intermediate per-row relative L2 limit of 0.015. Log the number of
  additional accepted `boundary_blocks`; zero means the fallback was unneeded.

The interval uses operand magnitudes, never observed kernel differences or a
coordinate-specific tolerance. Its FP32 accumulation and non-approximate tanh
assumptions make this a **provisional numerical model**, not a hardware/compiler
proof. Unexpected lowering, ambiguous scales, or non-adjacent discrepancies
still need investigation. The before-quantization routing mode retains the
exact v3 block check without this fallback.

After independent intermediate validation, FC2 is recomputed in FP64 using the
validated kernel intermediate and checkpoint weights. The existing BF16/FP32
contribution budget, exact recombination, zero/masked-output checks, original
reference per-expert L2, and original-reference/production-baseline output
checks remain unchanged. No failed stage proceeds to timing.

The fallback adds an untimed FP64 absolute-product GEMM only for experts failing
exact block matching. No kernel code or timed path changes. CPU tests cover the
two observed boundary cases, non-boundary and non-adjacent corruption, incorrect
and ambiguous scales, aggregate L2 rejection, and existing FC2/combine rejection.
Distributed GPU qualification remains pending.

Sync the replay and diagnostics Python files, unset `NT4_DEBUG_PREQUANT`, and
rerun the saved artifact without `--diagnose-prequant`. No checkpoint recapture
or cache clearing is needed. This uses the ordinary kernel specialization.

For a saved contribution failure, replay with `--allow-incomplete --graph
--knobs-json '{"group_hint":512}' --diagnose-input-quantization
--diagnose-expert-term 30 6 3` (substitute the reported row, slot and channel).
The sample and weights are saved before validation, so a checkpoint reload is
unnecessary. Keep the original scheduling knobs. The diagnostic logs the kernel
term and FP64 FC2 reference with unchanged quantized operands, then recomputes
FC1 and activation in higher precision to report intermediate quantization
sensitivity. It also reads the retained FC1 FP8 data and E8M0 scales from the
owning rank's completed workspace, locating the exact source rank/token/slot
within the selected expert's valid pool rows. Data and scale pools use separate
padding. The `kernel_intermediate` record compares the actual decoded row against
the reference and recomputes FP64 FC2 from that row. This requires no kernel
instrumentation or recompilation; keep it outside timing and before another
MoE invocation. The snapshot decoder has CPU layout/metadata tests but still
requires validation on GB300.
Aggregate baseline comparisons are printed before the unchanged contribution
gate. The run still fails if that gate fails; no benchmark is allowed afterward.

For controlled fixed-length occupancy, also pass `--ignore-eos`. This explicitly
sets model `--termination-id -1` and removes stop words; the output-length and
sequence-length limits still apply. It preserves the strict active-token capture
threshold. Treat these results as fixed-length decode, not natural EOS-stopped
traffic: tokens generated past EOS can change routing. The generated command is
saved in provenance and `generation_start` logs the termination policy.

All completed generation runs now write `capture_filter_summary.json`, even when
the requested captures are missing. It records peak decode occupancy, peak
occupancy within the padded-row limit, and rejection counts. Logs include new
decode occupancy peaks and periodic filter records beyond the first eight calls.

Pass `--prompt-file` and `--request-count` to `nt4_three_way.py` to disable the
synthetic sweep. The launcher replaces the saved prompt source, truncation,
request admission limit and arrival rate. It preserves the original token
budget unless the requested batch cannot fit it. `--samples 3` captures three
eligible steps separated by at least eight layer calls by default.

The generation driver registers its engine context in `nt4_capture_state.py`.
Capture requires decode-only state with zero prefill requests on every rank,
and at least `--min-active-tokens` actual engine tokens (default: request count).
Padding is not counted toward this threshold. Each result stores per-rank
engine request/token/prefill counts and the independent routed-token census.
Do not divide request count by EP size without inspecting these records.
If admission, EOS, or KV pressure prevents the target occupancy, the capture
fails rather than substituting repeated rows or prefill. An explicitly smaller
`--min-active-tokens` is allowed; report the measured counts in that case.

```bash
export NT4_REQUESTS=128
export NT4_REAL="$NT4_LOG_DIR/nt4-real-${NT4_REQUESTS}-$(date +%Y%m%d-%H%M%S)"
"$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_three_way.py" \
  --source-artifact "$NT4_LOG_DIR/nt4-checkpoint-layer-20261012-080901" \
  --mcore-root "$MCORE_CHECKOUT" --artifact "$NT4_REAL" \
  --prompt-file /lustre/fsw/portfolios/nemotron/users/helenn/research/routing/openmath2_unique2048.jsonl \
  --request-count "$NT4_REQUESTS" --samples 3 \
  > "${NT4_REAL}.log" 2>&1
```

Start with 128, then repeat at 256 and 512 after checking the actual occupancy.
The actual production CUTLASS path is kept intact. `baseline_audit` records its
NVLS capacity, gathered expert-input rows, weight type, and candidate row budget.
This is still an isolated routed-section measurement. Production full-capacity
CUTLASS quantization versus bucket-sized candidate buffers remains a performance
confounder; a bounded CUTLASS control and integrated tests are separate gates.

### Inspect a rejected v4 boundary block

Replay automatically emits `boundary_rejection_diagnostic` for blocks rejected
by both the exact variants and the boundary fallback (up to eight per expert).
It includes the original source row/top-k slot, channel, FP64 FC1/activation,
activation interval, midpoint and adjacency tests, stored FP8/scale bytes,
and independently computed lower/upper endpoint scales and block amax bounds.
This distinguishes an unexplained value from conservative scale ambiguity;
it does not relax any acceptance criteria or instrument/recompile the kernel.

Use `--sample-index 2` to replay only the third saved sample, retaining
`--allow-incomplete` for failed captures. Omit `--diagnose-prequant` and unset
`NT4_DEBUG_PREQUANT`; the existing recorded expert buffers are sufficient for
this first diagnostic. A standalone replay may fail to reproduce a fault tied
to prior samples' workspace state; rerun without `--sample-index` in that case.

The boundary gate decodes stored E8M0 scale bytes by constructing exact binary64
exponent bits. CUDA `pow(2.0, double_exponent)` produced a one-ULP scale error
for byte 123 in the third 128-token sample, causing false scale/representability
rejections. This is a validator repair; all equality checks and numerical
budgets remain unchanged. CPU regression tests cover every finite E8M0 byte
and the captured expert-121 FP8 block, including rejection of a wrong scale.

### Profile the live three-way performance gap

`nt4_three_way.py --profile-dir NEW_DIRECTORY --profile-iterations 10` now
forwards profiling to the capture wrapper. With `--samples 1 --request-count 128`
and the real prompt file, it validates and benchmarks all three paths before
recording warmed CUDA-graph traces for each backend and rank under `sample0/`:
`nvls_cutlass-rankN.json`, `nvls_megamoe-rankN.json`, `direct_megamoe-rankN.json`.
The production baseline requires a live checkpoint load; standalone replay
cannot reconstruct it from saved outputs. Use a completed source artifact for
model arguments. Retain unprofiled benchmark results for performance claims;
profiler startup/warmup ranges are not measurements. Trace kernel durations can
locate overhead in collectives, packing, and the main kernel, but do not expose
internal scheduling stalls within a persistent kernel.

### Diverse-routing scheduling pass

Use `nt4_gb300_tune.py --preset diverse --allow-incomplete` on the saved
`nt4-real-128-v4-20261013-120204` artifact after the exact E8M0 decode fix.
The 13-trial sequence brackets 11 candidates with group512 controls. It changes
FC1/FC2 grouping, static/atomic work assignment, and flag batching independently.
Every trial replays all saved samples, validates, and times CUDA graphs. No
checkpoint model load is needed and no production defaults are promoted.
`diverse-summary.json` pairs samples to the controls and reports speedups,
worst-sample regression, and control drift. Repeated `--artifact` arguments add
other routing workloads. These timings compare direct MegaMoE configurations;
a winning candidate still needs live comparison with NVLS+CUTLASS and hybrid.
MXFP8 M/N tile geometry remains fixed; a smaller tile requires a separate kernel
implementation/layout audit. Keep the original numerical gates for that work.


### Flag publication and work-group interaction

`flag_batch` batches readiness publication in each input-pulling warp. After a
completed token copy, the warp records the FC1 tile counter to increment. The
default of 4 delays four increments and publishes them together; 1 publishes
each immediately. Both perform one counter increment per token, and leftover
notifications are flushed when pulling finishes. FC1 still waits for its tile's
required input count. This is GPU-scope release-counter signaling on GB300,
independent of epilogue batching and Rubin counted writes.

The diverse three-sample sweep found roughly 4% lower direct-path latency with
`group_hint=512, flag_batch=1`. This is experimental, not a production default.
Test the interaction with work grouping using `--preset flag_interaction`:
512/flag1 control, 512/flag4, 1024/flag4, 1024/flag1, 768/flag1, 512/flag1 control.
Every trial checks all saved samples before reporting graph timings. The output
`flag-interaction-summary.json` compares corresponding samples against the mean
of the two flag1 controls, including drift and worst-sample regression.

```bash
unset NT4_DEBUG_PREQUANT
export NT4_TUNE="$NT4_LOG_DIR/nt4-flag-interaction-$(date +%Y%m%d-%H%M%S)"
"$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_gb300_tune.py" \
  --artifact "$NT4_LOG_DIR/nt4-real-128-v4-20261013-120204" \
  --preset flag_interaction --allow-incomplete \
  --timeout-seconds 1800 --output "$NT4_TUNE" \
  >"${NT4_TUNE}.log" 2>&1
```

These are direct-only timings. Remeasure the selected configuration against the
live production NVLS + CUTLASS path before claiming a production improvement.


### Experimental M128 token tile (GB300)

`--preset m128` compares the established M256/group1024/flag1 direct path
against M128/group1024/flag1 and M128/group512/flag1, then repeats the control.
M128 uses `(128,256,128)` MMA geometry, a `(1,1,1)` cluster and 1-CTA
instructions. Per-CTA M remains 128; N/K, activation, quantization, routing
weight placement and correctness gates are unchanged. This is an explicit NT4
experiment, not an automatic production selection or an established speedup.
Halving M can reduce padded expert computation, but also changes the number of
independent clusters and weight sharing. The group512 arm probes the changed
work-group balance; it is not guaranteed to be an equivalent schedule.

The coupled geometry is accepted only as a complete bundle by the harness.
Dispatch readiness, scheduler tile counts and per-token counter allocation use
M; the auxiliary workspace sizing helper now also uses M rather than N.
CPU tests cover host validation and JSON knob round trips. CUDA compilation,
1-CTA pipeline behavior, graph correctness and performance require remote GPU
validation. A failed trial stops the sweep without weakening numerical gates.

```bash
unset NT4_DEBUG_PREQUANT
export NT4_TUNE="$NT4_LOG_DIR/nt4-m128-$(date +%Y%m%d-%H%M%S)"
"$PY" "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_gb300_tune.py" \
  --artifact "$NT4_LOG_DIR/nt4-real-128-v4-20261013-120204" \
  --preset m128 --allow-incomplete \
  --timeout-seconds 1800 --output "$NT4_TUNE" \
  >"${NT4_TUNE}.log" 2>&1
```

Inspect `m128-summary.json` and all per-sample correctness results. The initial
sweep tests the direct path only; hybrid and production comparisons follow only
a successful experiment. New geometry may require a first-use JIT compilation;
existing cache entries should be retained.

Live three-way M128 follow-up: `nt4_three_way.py --knobs-json` now forwards
validated knobs to capture. Both direct Replay and the hybrid local expert
workspace use the same explicit knobs (previously hybrid was fixed to group512).
Pass `{"mma_tiler_mnk":[128,256,128],"cluster_shape_mnk":[1,1,1],"use_2cta_instrs":false,"group_hint":1024,"flag_batch":1}`.
Production CUTLASS configuration is unchanged. The M128 hybrid path still needs
its first GPU correctness and timing run.


### M128 real-cohort and capacity comparison

Run `bash "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_m128_capacity_sweep.sh"`
after syncing the checkout. Requires PY, FLASHINFER_CHECKOUT, MCORE_CHECKOUT,
and NT4_LOG_DIR. Optional NT4_PROMPT_FILE and NT4_SOURCE_ARTIFACT override the
OpenMath2048 prompt file and completed October 12 source artifact.

The three sequential captures test 128, 256 and 512 real requests with the
original production model token budget 1024. Cohort-sized budgets were removed
from this sweep after the 128-budget run never reached 128 active decode tokens.
Pass positional counts to run only pending cases, for example:
`bash "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_m128_capacity_sweep.sh" 256 512`.
Matched-capacity attribution remains pending a separate experiment that does
not change request admission.
Every run fixes candidate geometry to M128/group1024/flag1 and captures three
decode steps. Each reloads the checkpoint. The source artifact supplies model
arguments only; new activations/routes come from real generation, not tiling.
The script stops on a failed run or capacity/cohort audit. It checks the actual
NVLS capacity, candidate rows and active decode tokens in the saved manifest.

This changes the production engine's own token budget rather than replacing
its dispatcher with an approximation. Smaller budgets can alter prefill and
admission scheduling; compare the saved cohorts/routes and numerical results
before attributing differences solely to buffer size. EOS is disabled to retain
the cohort. These are 128/256/512 unique requests, not EP4 times that many unique
requests. No threshold or production default is promoted from this experiment.
All timings remain isolated routed-layer timings with model overlap excluded.


### Matched candidate capacity, unchanged admission

Run `NT4_MATCH_CAPACITY=1 bash "$FLASHINFER_CHECKOUT/examples/moe_ep/nt4_m128_capacity_sweep.sh" 128 256 512`.
Production retains its 1024-token budget. Candidates allocate the actual NVLS
capacity (1024 rows/rank), stage the unchanged captured rows, and mask all extra
rows with invalid expert IDs and zero routing weights. Direct and hybrid padded
outputs must be zero. Three-way correctness and timing gates remain enabled.
The sweep audits actual allocations after every capture and saves new artifacts.

This removes the smaller candidate allocation advantage without altering request
admission. It is a conservative full-padded-candidate experiment: the current
hybrid communicates and processes its full capacity, whereas production may
limit particular operations using active metadata. It does not claim identical
communication bytes. Compare with the cohort-sized candidate runs on the same
cohort/node where possible, and inspect the saved routing census. GPU validation
of the capacity-expanded hybrid remains pending.


### First end-to-end eager decode substitution

`nt4_e2e.py` reuses a completed EP4 artifact's model arguments but launches the
real generation driver without capture hooks. It runs baseline-start, direct,
then baseline-end in separate torchrun processes on the same node. Each loads
the checkpoint once and runs one untimed warmup followed by three timed full
generations. The model token budget stays 1024; requests default to 128. EOS is
disabled for a controlled 128-output-token workload inherited from provenance.

`nt4_e2e_adapter.py` substitutes eligible MXFP8 routed experts during decode
with M128/group1024/flag1, retaining production prefill, router, shared experts,
latent projections, normalization and shortcut blocks. Each eligible layer
exports its own weights and checks its first decode call against the production
routed output and existing independent numerical gate during warmup. Capacity
matches production. Non-MXFP8 layers keep production. Timed runs reject newly
encountered eligible layers and fail if no MegaMoE calls execute. Production
padding masks remain applied. No per-step CPU tensor export is done after warmup.

Outputs include per-round wall time, output throughput, engine decode/prefill
step timings, substituted layer IDs/call counts and generated token sequences.
`summary.json` reports exact request agreement and first divergent token indices
against baseline-start. Numerical differences can change greedy generations;
exact agreement is reported, not assumed. Compare baseline-end too for noise
and deterministic-output controls. Review any output divergence before claiming
end-to-end equivalence. Initialization, JIT and first-call checks are in warmup.

This is explicitly an eager smoke/performance test: both model paths require
CUDA graphs disabled. Full-model graph integration and graph-enabled serving
performance are not established by this harness. Additional resident candidate
weights/workspaces can increase memory usage; no production code is changed.


E2E update: trials now run FlashInfer+NVLS baseline, direct decode MegaMoE,
MCore's `vllm` backend selection + NVLS, then FlashInfer baseline again.
The vllm selector uses native vLLM-style kernels for BF16; MXFP8 falls back to
MCore scaled grouped GEMM in this checkout. It is not native vLLM MXFP8.
The vllm selection applies to prefill and decode, while direct substitution
remains decode-only; inspect both full wall time and output agreement.
Per-trial launch commands are saved explicitly. All-zero engine step timings
are represented as unavailable (`null`), not measured zero latency.
Normal completion now restores patched methods, closes every candidate workspace
in layer order, releases runtime handles, synchronizes, and destroys the process
group before Python exit. GPU validation of this teardown fix remains pending;
a nonzero child exit still fails the sweep even if timing files were saved.

### E2E routed-section CUDA graphs

Add `--routed-graphs` to `nt4_e2e.py` to capture each eligible MXFP8 decode
layer's dispatch/expert/combine section for **all** trial backends. The outer
model, prefill, router, shared experts, and shortcut operations remain eager.
This is not a BF16 MegaMoE implementation or full-model CUDA graph integration.

Each backend pays three common live-input copies into stable graph buffers;
these remain outside capture and inside wall timing. Direct's internal padded
input staging is captured. Compilation, capture, eager-versus-graph comparisons,
and changed-route-mask checks happen during the excluded warmup generation.
A new layer or shape encountered in a timed round fails instead of capturing.
Graphs are destroyed before the symmetric workspaces during explicit teardown.

Results report `routed_graph_calls` and `routed_graph_count`; zero graph calls
fail the run. Compare synchronized wall times against both bracketing production
controls. The vLLM-selected MXFP8 arm still uses MCore's scaled grouped-GEMM
fallback. Saved token sequences remain available to assess divergence; passing
layer numerical checks does not establish model quality equivalence.

This implementation requires remote GPU validation. Graph capture may expose
backend operations that are not capture-safe; such failures are not bypassed.

### Full E2E precision and request-count sweep

`nt4_e2e_sweep.py` runs powers of two from 1 through 1024 by default, with
routed-section graphs on all measured arms. MXFP8 includes MegaMoE direct;
BF16 includes FlashInfer + NVLS, vLLM grouped GEMM + NVLS, and the new
NT4 BF16 MegaMoE direct variant. An EP4 synthetic eager/graph/empty-rank smoke
gate runs before full model loads; any failure stops the sweep. Each real
checkpoint layer also compares BF16 contributions against an independent
FP32 GEMM/activation/BF16-handoff oracle and checks the live reference. BF16 removes FP8 options from the inherited
checkpoint command. No BF16 performance claim is inferred from MXFP8 results.

The fixed token budget defaults to 2048 across all points, leaving admission
headroom at 1024 requests. This differs from earlier 1024-budget experiments.
Request capacity equals the requested count. Prompt lengths and KV memory can
still prevent reaching the requested decode occupancy: the report records an
active-request histogram and marks such points OCCUPANCY_LIMITED, not PASS.
PASSED timing collection does not establish output quality equivalence.

Each case reloads the real checkpoint per backend, runs an excluded warmup and
three timed rounds, and brackets the candidates with FlashInfer controls.
The full 22-case sweep launches 88 model processes (plus the smoke test); expect a long run. Source
JIT caches are reused. Failed points are logged and the sweep continues.
`sweep-summary.json` is updated after every case. Rerun the same command with
`--resume` and the identical output path/configuration to skip completed cases
and retry failed attempts without overwriting their logs. A completed but
occupancy-limited point is retained; use a new sweep with changed configuration
if admission needs adjustment. GPU validation of the new BF16 MegaMoE and graph arm remains
pending.


### Graph memory at large request counts

Routed graphs use separate private pools per layer and bucket. The attempted
per-layer pool sharing was withdrawn after BF16/512 and BF16/1024 failed the
added cross-bucket replay check. Pool aliasing versus synchronization in that
check remains unresolved; do not use that revision for performance claims.
The earlier separate-pool implementation exhausted memory in large-batch
baseline runs. This is a harness limitation, not an established backend limit.
`routed_graph_ready` logs the pool policy and allocated/reserved CUDA bytes.
Keep the same graph policy across all backends in a reported comparison.

### BF16 profiling and scheduling experiment

`nt4_bf16_tuning.sh` provides four independent phases: `capture`, `profile`,
`tune`, and `confirm`. Set `NT4_BF16_TUNE_ROOT` to a new persistent directory
and keep it unchanged between phases. Use the same checkout revision throughout;
do not sync new code into a running experiment. Run GPU phases sequentially on
an otherwise idle EP4 GB300 node. They can be split across node leases.

Capture loads the real model at request counts 128 and 256 with token budget
2048, saving BF16 expert weights and two full-occupancy decode inputs/outputs at
layers 7, 32, and 68 during untimed warmup. These are baseline-generated routes.
It does not reuse MXFP8 captures as BF16 truth. Each capture has a completion
manifest only after every rank has saved all required samples. Weights are saved
once per layer/rank (roughly 18 GiB per request-count artifact); allow space for
both artifacts and profiles. Interrupted captures must use a fresh output root
or be moved aside explicitly; incomplete directories are never overwritten.

Profile uses Nsight Systems on the default isolated kernel path at each request
count. Only 20 warmed graph replays enter the profiler capture range. Inspect
main-kernel time, separate weighted reduction, and clearing/staging launches.
Nsight Systems cannot separate compute from waits *inside* the fused kernel;
that requires a focused Nsight Compute pass after inspecting this trace.
Profiling is separate from performance measurements.

Tune compares eight explicit scheduling variants, bracketed by unchanged
controls, over all captured layers and samples at both sizes. Supported knobs:
group_hint, static versus atomic work assignment, flag_batch, and epi_flag_batch.
The BF16 kernel's accepted geometry remains (256,256,64), cluster (2,1,1), two CTA;
changing tiles requires a separate kernel implementation change, not a flag.
The default flag_batch is already 1. No arithmetic or tolerance changes are made.
Each trial runs the independent BF16 expert/recombination/baseline checks and
graph/eager comparison before timing, in a fresh torchrun process with a timeout.
Timing includes capacity clearing/staging, the main kernel, and late reduction;
results use the maximum per-rank CUDA-event duration. The script preserves each
round, sample, capacity, active-row count, log, and candidate configuration.

The recommendation gate requires <=3% bracketing-control drift and at least 2%
improvement against the faster control on *every* sample. Failure at one size or
layer disqualifies the candidate. This is a conservative selection heuristic,
not statistical significance. `candidate-knobs.json` is emitted only if a
candidate qualifies. The controller does not change runtime defaults.

Confirm adds an unchanged MegaMoE direct-control arm alongside the tuned direct
arm, vLLM grouped GEMM + NVLS, and initial/final FlashInfer + NVLS controls. It
runs five timed rounds at both 128 and 256 with independent private graph pools.
Inspect baseline drift, occupancy, and output comparisons before making a claim;
isolated-kernel wins need not become E2E wins. Do not run a full sweep or promote
a preset solely on the tuning result. Remote GPU validation remains required.


### BF16 isolated/E2E discrepancy audit

The October 15 group1024 E2E confirmation did not reproduce the isolated gain:
median throughput changed by -0.86% at 128 requests and -2.39% at 256 versus
unchanged MegaMoE. The 256 FlashInfer opening/closing control drift was +58.9%,
so that baseline comparison is invalid. Do not promote this candidate yet.

Both BF16 standalone tuning and routed E2E now emit `bf16_effective_config`
after compilation: requested knobs, actual frontend config, effective kernel
object group hint, compile key, capacity, and source paths. Explicit knob
mismatches fail the run. This audits configuration, not the generated SASS.

Add `--diagnostic-timing-layers 7,32,68` to `nt4_e2e.py` with routed graphs to
collect per-call CUDA-event durations for live input copies and graph replay.
Events are allocated/initialized during warmup and reused in measured rounds.
Each result includes every rank's per-layer graph sums, medians, and per-call
samples under `routed_diagnostic_timing`. No per-call CPU synchronization is
added; event results are collected after generation synchronizes. Instrumented
wall throughput is diagnostic only. Compare equivalent layer, occupancy and
capacity across arms; do not sum independent rank maxima as a critical path.
The standalone benchmark repeatedly executes fixed inputs for one layer;
E2E rotates through all layers and changing inputs. This difference is a
hypothesis to investigate, not an established cause of the discrepancy.


For the rank-arrival investigation, use `--profile-direct-pair --rounds 1`
with BF16 candidate knobs and routed graphs. This runs only direct-control and
direct, each with a full warmup followed by one profiled generation. Nsight
Systems wraps torchrun and traces child ranks in a common timeline with CUDA,
NVTX and OS-runtime activity, plus graph-node detail. CPU sampling and context
switch tracing are disabled; CPU affinity and Torch thread counts are logged.
Model loading, compilation and graph warmup are outside the capture range.
The generation includes prefill and decode; NVTX markers identify routed decode
calls by rank/layer/call/rows. There are no extra per-layer barriers. Do not use
profiled throughput for promotion. Summary output comparisons use direct-control
as their reference. Download both `.nsys-rep` files, logs and summary. Output
must be a new directory; no existing traces are overwritten.


Profiling defaults to `--profile-graph-trace graph` after a run stalled on the
first profiled step with node tracing. This is a diagnostic change, not a
confirmed deadlock fix. `--profile-graph-trace node` retains the previous mode.
Graph-level tracing retains graph and CPU launch timing but omits internal
node detail. Profile startup/return/stop markers and per-rank Python stack dumps
(every 120 seconds, under each arm's output directory) localize future stalls.
The stack dumper reports state and does not terminate the run.


The graph-level run completed generation but rank 1 stalled in
`cudaProfilerStop()`. Other ranks could already be in the following NCCL
elapsed-time reduction. Profile mode now creates a separate Gloo group and
uses CPU-only monitored barriers before and after profiler stop. The latter
prevents CUDA allocation, NCCL aggregation, or cleanup until every stop returns.
The barriers are outside generation and its elapsed timer; no model scheduling
is changed. A 300-second group timeout reports missing ranks. This removes a
shutdown hazard; remote validation is still needed to establish whether it
resolves this profiler hang.


The stop hang reproduced even with the CPU-only barriers: all ranks reached
stop and one remained inside `cudaProfilerStop()`. Profile-pair mode now uses
`--capture-range=nvtx --nvtx-capture=nt4_profile_generation` and a single
rank-0 push/pop range, with `NSYS_NVTX_PROFILER_REGISTER_ONLY=0` for PyTorch's
unregistered NVTX strings. All ranks remain traced as child processes. Gloo
barriers ensure all ranks start after the capture opens and all GPUs finish
before it closes; no CUDA work is submitted until closure returns. There are
no cudaProfilerStart/Stop calls in this E2E mode. This avoids the observed API
stop path, but successful trace export still requires remote validation.


NVTX-controlled CUDA tracing also stalled at rank-0 `nvtx.range_pop()`; the
other ranks subsequently hit the 300-second host barrier timeout. Capture-stop
API choice did not resolve the problem. `--profile-trace host` disables CUDA
collection and keeps NVTX/OS-runtime timelines. This can diagnose host arrival
skew, but cannot establish device start/completion times or expert-work imbalance.
Run `nt4_nsys_smoke.py` under the same host-only nsys configuration first and
require four `profile_smoke_pass` records and an exported report. This is a
collection-path smoke test, not a model/NVSHMEM reproduction. If it fails, stop
full-model profiling attempts and investigate the installed profiler/toolchain.


The host-only four-rank smoke also timed out after capture end while Nsight
announced report generation and symbol downloading. The next isolation test
uses `--capture-range-end=none`: NVTX starts collection, but closing the range
does not stop it. Collection continues through cleanup until process exit,
so report generation is no longer requested from the rank-0 range-pop path.
The profile-pair launcher now uses this setting too. Only interpret the marked
generation interval; teardown will also be present. Require a successful smoke
and exported report before another model run. This workaround is unvalidated
on the remote node and does not establish a symbol-download root cause.


### Profiler-free host arrival audit

`--host-timeline-pair` runs unchanged/tuned BF16 without invoking nsys, NVTX
capture control, Gloo profiling barriers, or CUDA timing events. It requires
BF16 candidate knobs and routed graphs, and cannot combine with profiling or
event timing. During post-warmup generation, it records CPU monotonic timestamps
at routed compute entry, input-copy submission, graph replay submission and
return. No per-call I/O or synchronization is added. Each rank writes
`host-timeline-roundN-rankR.json` after the round's existing synchronization
and wall timer. Files include hostname; compare timestamps only on the same
host. Align records by layer and call index and verify row counts before
comparing rank arrival. Graph submission durations are host API durations,
not GPU execution times. Instrumented throughput remains diagnostic only.

### Rank-skew CPU stack diagnostic

Add `--host-stack-sampling` to `--host-timeline-pair` for a dependency-free
Python stack diagnostic on all four ranks. Use one measured round initially.
After warmup, a sampler thread waits for the first routed decode event, then
samples the main Python thread approximately every 10 ms for at most 30 seconds.
Each arm writes `host-stacks-round1-rank{0,1,2,3}.json` alongside its host timelines.
The files contain interned stacks (innermost frame first), monotonic timestamps,
the most recent routed layer/call count, CPU affinity, and optional Linux
main-thread `/proc/.../schedstat` counters. No locals or tensor contents are saved.

Sampling requires the GIL, so native code holding the GIL can delay samples.
Use actual timestamp gaps, not the nominal interval, when interpreting coverage.
These are Python call sites, not native stacks: a frame at a CUDA call does not
identify the native wait mechanism. Scheduler counters are cumulative; runqueue
accounting may be disabled on the node. Compare deltas only when available.
This mode labels measured rounds `performance_use=diagnostic_only`; use the
separate unsampled run for throughput. It neither invokes Nsight nor changes
kernel code, and requires no JIT cache reset.

The 2026-10-15 `host-stacks128-20261015-113735` diagnostic found repeated
`_expert_weights_use_mxfp8` validation in the adapter's decode preprocess hook
(10–13% of Python stack samples per rank). The adapter now caches the validated
format per layer for the lifetime of the fixed-checkpoint experiment. This does
not support weight refit during a run; create a new adapter after changing weights.
Mixed-format validation still runs before first use. The optimization applies to
all arms using the adapter, including routed-graph baselines. Recheck throughput
without stack sampling before making any speedup claim.

In that diagnostic all main threads accumulated ~99% CPU runtime across the sampled
window and <1 ms each of recorded runqueue wait. Rank 1 was last throughout the
control, whereas the tuned run alternated between ranks 0 and 1; the earlier
rank-0-only observation is not universal. Python call sites cannot distinguish
native driver CPU work from spinning, and the sampling fractions are not exact
wall-time attribution. These results identify removable harness overhead but do
not establish the complete cause of the remaining skew.

### Whole-language-model decode graphs (experimental, GPU validation pending)

Use `--model-graphs` instead of `--routed-graphs` with `nt4_e2e.py`. This sets
MCore `cuda_graph_impl=local`, `inference_cuda_graph_scope=block`, decode-only
capture, three capture warmup forwards and four requested graph buckets. The same
settings apply to FlashInfer + NVLS, unchanged MegaMoE, tuned MegaMoE, and vLLM +
NVLS. MCore may round the requested bucket count; inspect `model_graph_ready`.
Prefill, token sampling, request scheduling and the Python generation loop remain
outside the whole-language-model forward graph.

`nt4_checkpoint_generate.py` temporarily defers the engine's initial graph build.
`nt4_model_graph.py` then disables the language model graph selector for an eager
reference generation (round -1), after the selected backend adapter is installed.
This prepares symmetric allocations, kernels and real-input validation outside
capture. The engine subsequently builds its normal model graphs. The adapter
records its raw operations directly into those graphs, rather than replaying a
nested routed-section graph. Per-bucket routed graphs are used temporarily for
validation and reset before outer capture.

Round 0 warms model graph replay. Every graph round must produce exactly the
same generated tokens as that backend's eager reference. Measured decode forwards
must replay exactly one language-model graph; new captures and eager fallbacks
raise errors. `model_graph_ready` records captured routed layers, while each
`e2e_result` includes `model_graph_count`, `model_graph_replays`,
`model_decode_steps`, and `model_graph_vs_eager_outputs=EXACT`. Python MoE call
counts are not meaningful inside a model replay and are reported as null.
Model graphs are destroyed before freeing symmetric adapter buffers.

Start with BF16, 128 requests, two measured rounds, and the existing group1024
candidate file. Sync the new `nt4_model_graph.py` as well as the launcher,
generation driver and adapters. This is a new GPU integration path: CPU lifecycle
checks pass, but CUDA capture/communication/shortcut stream behavior must be
validated on the remote GB300. Do not use timings from a failed correctness or
replay-coverage gate. No MCore source changes or JIT cache reset are required.

### Combining MegaMoE with PR 8257 shortcut overlap

MCore integration checkout: `/Users/helenn/Documents/github/megatron-lm`, branch
`helenn-nt4-megamoe-shortcut-integration`, PR head
`12a795d3b832923fc0eb4117d4bfec77714d63ec` (base `3914483fd325e053797aff3fa431d4a3141139aa`).
This is the complete PR head, not a patch against the older remote checkout.
Sync that checkout and the FlashInfer experimental checkout before running.
No changes to shortcut mathematical semantics or validation assertions are made.

First run the serial `--model-graphs` experiment. Then add `--shortcut-overlap`
(BF16 only). The overlap experiment runs native vLLM + NVLS before/after the
unchanged and tuned MegaMoE direct candidates. All arms use the PR's
inference-optimized BF16 model, TP/CP1, NVLS dispatcher selection, same graph
buckets, mixer schedule and stream fork/join. Prefill/mixed remain the PR's
native vLLM + NVLS implementation, including in direct candidates.

For decode, `nt4_shortcut_bridge.py` adapts prepare/compute/finish calls to the
existing MegaMoE direct implementation. Its dispatch and combine hooks bypass
NVLS for the active direct branch; MegaMoE performs its own communication and
expert computation on the caller's shortcut stream. This is MegaMoE direct plus
shortcut overlap, not MegaMoE expert-only plus NVLS. Shared experts and the final
join remain owned by the PR. vLLM baselines keep all their native dispatch,
prepare/compute/finish and combine stages.

The PR's expert block cap applies to vLLM, not MegaMoE; the persistent MegaMoE
configuration is controlled by its own knobs. Report that distinction when
comparing concurrent execution. Model capture must visit the direct shortcut
bridge for every routed layer, in addition to the eager/graph exact-output gate
and per-decode model graph replay checks. CPU bridge dispatch/fallback/restore
tests pass; end-to-end GPU correctness and speedup are not yet established.
