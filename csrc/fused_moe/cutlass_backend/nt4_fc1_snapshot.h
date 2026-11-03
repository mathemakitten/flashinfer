#pragma once

// Host-only, explicitly enabled diagnostic. Never used in graph capture/timing.
#include <cstdlib>
#include <fstream>
#include <vector>

template <class OffsetFn>
inline void nt4SnapshotFc1(char const* prefix, void const* output, void const* scales,
                           int const* inverse, int64_t const* expert_offsets,
                           int64_t rows, int64_t width, int topk, int start_expert,
                           int local_experts, int sf_m_alignment, int sf_k_alignment,
                           OffsetFn sf_offset, cudaStream_t stream) {
  constexpr int token = 155, slot = 5, expert = 180;
  TLLM_CHECK(rows > token && topk > slot && width == 4096);
  TLLM_CHECK(expert >= start_expert && expert < start_expert + local_experts);
  cudaStreamCaptureStatus capture;
  TLLM_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
  TLLM_CHECK(capture == cudaStreamCaptureStatusNone);
  int perm = -1;
  int64_t offsets[2];
  int local = expert - start_expert;
  TLLM_CUDA_CHECK(cudaMemcpyAsync(&perm, inverse + slot * rows + token, sizeof(perm),
                                 cudaMemcpyDeviceToHost, stream));
  TLLM_CUDA_CHECK(cudaMemcpyAsync(offsets, expert_offsets + local, sizeof(offsets),
                                 cudaMemcpyDeviceToHost, stream));
  TLLM_CUDA_CHECK(cudaStreamSynchronize(stream));
  TLLM_CHECK(offsets[0] <= perm && perm < offsets[1]);
  auto begin = sf_offset(local, offsets[0]);
  auto end = sf_offset(local + 1, offsets[1]);
  TLLM_CHECK(begin >= 0 && end > begin);
  std::vector<unsigned char> q(width), sf(end - begin);
  TLLM_CUDA_CHECK(cudaMemcpyAsync(q.data(), static_cast<unsigned char const*>(output) + perm * width,
                                 q.size(), cudaMemcpyDeviceToHost, stream));
  TLLM_CUDA_CHECK(cudaMemcpyAsync(sf.data(), static_cast<unsigned char const*>(scales) + begin,
                                 sf.size(), cudaMemcpyDeviceToHost, stream));
  TLLM_CUDA_CHECK(cudaStreamSynchronize(stream));
  auto save = [&](char const* suffix, std::vector<unsigned char> const& data) {
    std::ofstream file(std::string(prefix) + suffix, std::ios::binary);
    TLLM_CHECK(file.good());
    file.write(reinterpret_cast<char const*>(data.data()), data.size());
    file.close();
    TLLM_CHECK(file.good());
  };
  save("-q.bin", q);
  save("-sf.bin", sf);
  std::ofstream meta(std::string(prefix) + "-meta.json");
  meta << "{\"token\":" << token << ",\"slot\":" << slot << ",\"expert\":" << expert
       << ",\"permuted_row\":" << perm << ",\"expert_begin\":" << offsets[0]
       << ",\"expert_end\":" << offsets[1] << ",\"sf_begin\":" << begin
       << ",\"sf_m_alignment\":" << sf_m_alignment << ",\"sf_k_alignment\":" << sf_k_alignment
       << ",\"width\":" << width << ",\"sf_bytes\":" << sf.size() << "}";
  meta.close();
  TLLM_CHECK(meta.good());
}
