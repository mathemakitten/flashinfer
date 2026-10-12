# NT4 MegaMoE: checkpoint-layer validation and decode timing

## Decision and scope

Build a replay harness for **one real NT4 routed MoE layer on four GB300 GPUs**.
Compare the existing MCore FlashInfer/NVLS path with the NT4 MegaMoE prototype
using identical checkpoint weights, latent inputs, routes, and probabilities.
Establish correctness before measuring latency. This is the next contained
project; full-model integration and the Rubin port follow its result.

The initial exporter, replay harness, and benchmark are now implemented locally.
They have CPU test coverage but have not yet been exercised on the GPU cluster.
See [the checkpoint run guide](nt4_checkpoint_run.md) for the first command and
known boundaries. This document remains the project specification. The runnable synthetic harness
is documented in [the bring-up guide](nt4_megamoe_bringup.md).

## Evidence so far

As reported by the user on October 8, 2026:

| Test | Result | Interpretation |
| --- | --- | --- |
| Sparse baseline and NT4, including masking and replay | Passed | Basic dispatch, layout and activation checks |
| Small dense NT4 | Passed | Dense synthetic arithmetic checks |
| Full-shape dense, prequantized, eager; seed 1234, 16 checks | Passed under v2 | Synthetic routed-branch qualification |
| Full-shape dense, prequantized, graph; seed 4321, 16 checks | Passed under v2 | Second seed and fixed-buffer graph replay |
| Real checkpoint layer; performance; Rubin | Not tested | Next gates |

These are user-reported runs, not a locally reproduced or archived benchmark.
Save complete logs and exact source revisions with subsequent results. No speedup
has been established. Larger-input stress with `--input-scale 8` remains unreported.

Two numerical findings explain why the acceptance policy changed:

- Expert 395 differed by one BF16 value, -6.71875 versus -6.75. The other nine
  contributions matched; kernel recombination was exact. Cancellation across
  experts magnified the final relative error.
- Expert 387 differed by 1.31e-6 near zero. Its FC2 absolute-product sum was
  97.77, while the signed result was about -2.31e-5. The difference was only
  1.34e-8 of the absolute-product sum. This is consistent with accumulation
  roundoff; the exact kernel arithmetic source was not isolated.

The `bf16_fp32_contributions_v2` gate checks every expert contribution using a
BF16 allowance plus a conservative FP32 accumulation bound, per-expert vector
and global relative L2 <= 0.015, finite values, exact masked zeros, exact
recombination, and exact replay. The old output comparison remains diagnostic.
This is a development gate, not a model-quality guarantee. Its accumulation
bound assumes identical operands and cannot certify arbitrary differences in
activation or intermediate quantization.

## Contract and comparison boundary

| Property | Required value |
| --- | --- |
| Model hidden / routed latent width | 3072 / 768 |
| Expert intermediate width | 4096, non-gated |
| Experts / top-k | 512 / 10 |
| EP / TP / expert TP | 4 / 1 / 1 |
| Local experts | 128, with explicit global-ID mapping |
| Weights | Actual selected layer's MXFP8 E4M3 bytes and E8M0 scales |
| Activation and weighting | `p * (16 * tanh(relu(z) / 16))**2` before FC2-input quantization |
| Output | BF16 latent-width routed sum, in original local token order |

Start timing immediately before baseline dispatch preprocessing, after the
3072-to-768 latent projection. Stop after baseline combine/postprocessing has
restored the local routed output, before latent output normalization/projection
and shared-expert or shortcut additions. Map the exact calls while implementing;
assert both captured tensors are `[local_tokens, 768]` in the same token order.

The baseline includes its NVLS dispatch, packing, input quantization, expert
compute and combine. MegaMoE includes staging, quantization, invalid-slot clear,
fused execution and final reduction. Feed MegaMoE the original local tokens;
do not run the outer NVLS dispatcher around it.

Router execution, latent projections, norm-before-up-projection, shared expert,
shortcut connection, and wide residual stay outside the first timing boundary.
Capture their real inputs/routes so the replay remains representative. Preserve
the supplied inference configuration: the training script's shortcut-parallel
setting is not evidence that inference enabled it.

## Work package 1: export one layer and its inputs

Use the normal MCore model/checkpoint loader from the user's working inference
command. Source checkpoint:

`/lustre/fsw/portfolios/nemotron/projects/nemotron_n4_pre/nemotron4/derisking/shared_ckpts/nano`

Use checkpoint step **30000**, format `torch_dist`, as in the supplied run
command. Verify the path is available on the development node.

Select one routed layer that actually uses MXFP8. Record the module path and
layer index; do not assume the global FP8 flag overrides selective BF16 layers.
Use the tokenizer and checkpoint arguments from the working NT4 command.

Relevant MCore source inspected in the local checkout:

- `megatron/core/transformer/moe/moe_layer.py`: latent projection,
  `preprocess`, dispatch, routed experts and combine/postprocess boundaries.
- `megatron/core/transformer/moe/experts.py`: `InferenceGroupedMLP`, canonical
  MXFP8 stacking, `_flashinfer_forward`, and derived-weight refresh.
- `megatron/core/inference/moe/flashinfer_mxfp8.py`: canonical scale unshuffle
  and routed Major-K weight preparation. Confirm the remote revision matches;
  the baseline is the user's actual working backend, not an assumed PR name.

Export one file per EP rank plus a manifest. Required contents:

| Data | Required checks |
| --- | --- |
| Canonical W1 `[128,4096,768]`, W2 `[128,768,4096]` | E4M3 payloads, logical shapes, local-to-global expert IDs |
| Canonical scales `[128,4096,24]`, `[128,768,128]` | Raw E8M0 bytes, block size 32 along reduction dimension |
| Local latent inputs and valid counts | BF16 `[T,768]`, token order and padding mask |
| Selected global IDs and probabilities | `[T,10]`, exact post-router probabilities; no second sigmoid/3.16 scaling |
| Baseline routed outputs | BF16 `[T,768]`, captured at the common endpoint |
| Provenance | Checkpoint path/step, layer, git SHAs plus dirty patch, GPU topology, versions, precision settings |

Reformat and permute weight/scale bytes only. Add an inverse-transform byte
comparison to validate export/packing. Do not dequantize and requantize checkpoint
weights. If the loaded representation lacks a required canonical orientation,
stop and document it before inventing conversion semantics. Dequantization is
allowed in the reference computation only.

Capture several consecutive decode steps and a prefill sample separately.
Record the per-rank token counts actually observed; do not assume 64 requests
means 64 distinct tokens on every EP rank. Include the original single-prompt
workload and a batch of distinct prompts for occupied decode cases.

## Work package 2: correctness replay

Implement an explicit four-rank replay script using the exported artifact. Run
baseline and MegaMoE serially with independent workspaces and identical inputs.
Initialize collectives in the same order on every rank, including empty ranks;
print exceptions before collective cleanup and keep a watchdog.

1. Verify manifest, canonical payload hashes, dimensions, expert mapping and
   precision before compilation. Validate all active expert IDs and finite data.
2. Replay the baseline and compare with its captured output to qualify the
   artifact and boundary. Resolve any mismatch here first.
3. Replay MegaMoE with unchanged checkpoint bytes/scales. Require finite output,
   exact inactive-slot zeros and deterministic repeated execution.
4. Build the quantization-aware reference from canonical checkpoint weights,
   using the actual captured input and probabilities. Reuse the existing v2
   contribution checks and reference absolute-product diagnostics. Process
   experts as needed rather than gathering the whole checkpoint on each GPU.
5. Compare both backends to that reference and directly to each other. Report
   global and per-token relative L2, maximum absolute error, strict-output
   mismatch count and contribution metrics. Use <=0.015 global/per-token
   relative L2 as a provisional direct-comparison screen, plus exact empty-row
   zeros; investigate near-zero-row failures with absolute/contribution data.
   Do not silently revise thresholds to pass a checkpoint.
6. Audit probability placement, intermediate MXFP8 scale construction, and
   rounding order if the two backends implement different numerical contracts.
   Record the difference; the synthetic v2 pass cannot resolve it automatically.
7. Repeat with changed captured steps, unequal valid counts, one empty rank,
   partial masking, eager mode and fixed-capacity graph replay. Construct masks
   from captured routes without changing active probabilities.

Done means both paths reproduce the intended checkpoint layer at the common
boundary, with all numerical differences explained and logged. Full-model
quality still needs a later gate after integration.

## Work package 3: decode benchmark

Create a separate timing mode with no reference computation, workspace snapshot,
tensor-to-host copy or per-iteration diagnostic printing. Keep the production
invalid-slot clear enabled. Load/pack weights, allocate and compile once before
measurement; report setup time and peak allocated/reserved memory separately.

- Benchmark captured decode distributions first. Add controlled capacities of
  1, 2, 4, 8, 16, 32 and 64 **local tokens per rank**, reporting actual counts
  on all ranks and aggregate count. Include one uneven and one empty-rank case.
  Label resampled/repeated captured inputs as synthetic load, not real decode.
- Test eager execution first, matching the supplied graph-disabled inference
  run. Report graph timings separately, with equivalent boundaries on both paths.
- Warm up at least 20 calls. Measure at least 100 calls in each of five rounds,
  alternating backend order. Check convergence and extend only if timing is
  unstable. Use CUDA events on the stream that joins all completion work;
  explicitly verify side-stream completion is included.
- Align ranks before a round, not by injecting a barrier into every timed call.
  Gather timings after measurement and report the maximum rank duration per
  sample, then median and p95 across samples. Also report eager host wall time
  with synchronization around whole batches to expose launch overhead.
- Record GPU clocks/power settings if available, concurrent jobs, NVLink
  topology, backend knobs, graph mode and exact versions. Avoid tuning one
  backend while leaving the other in an accidental fallback.

Required result row:

`layer, checkpoint, tokens_per_rank, capacity, routing_source, graph_mode,
baseline_median_us, baseline_p95_us, megamoe_median_us, megamoe_p95_us,
baseline_over_megamoe, peak_memory, correctness_status, source_revision`

The microbenchmark proves only routed-branch latency. A full-layer or decode
benefit must later include latent projections, shared-expert overlap and the
rest of the model. GB300 results do not establish a Rubin-specific win.

## Exit decision and deliverables

- **Correct and repeatably faster on captured low-token cases:** add an opt-in
  MCore integration and measure full-layer/full-model decode before Rubin tuning.
- **Correct but no repeatable benefit:** keep the validated kernel and report
  where time goes; do not commit to broad integration on correctness alone.
- **Numerical discrepancy or unsupported precision/layout:** stop timing claims
  and resolve the specific contract mismatch.

Deliver an exporter, versioned manifest schema, replay/correctness harness,
benchmark mode, saved logs/result table, and a short go/no-go report. Keep the
project on a dedicated branch off main after auditing dependencies on the
current clamped-ReLU2 work. Keep the general masked-combine fix in a separate
commit. Branch migration is planned, not completed by this write-up.
