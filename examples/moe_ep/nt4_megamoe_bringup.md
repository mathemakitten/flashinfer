# NT4 MegaMoE development on GB300

## Status

The Blackwell MXFP8 backend now has an explicit **source prototype** for
NT4's non-gated, tanh-clamped squared ReLU. On October 8, the user reported
passing GB300 sparse checks: baseline and NT4 eager, graph replay with
prequantized weights, and NT4 full shape. The user subsequently reported passing full-shape dense prequantized tests
under `bf16_fp32_contributions_v2`: eager with seed 1234 and graph replay with
seed 4321, 16 checks each. **Checkpoint correctness, resource usage, and
performance remain unverified.**

The next project is specified in [Checkpoint-layer validation and decode
timing](nt4_megamoe_checkpoint_plan.md). The initial implementation and first
GPU command are in [the checkpoint run guide](nt4_checkpoint_run.md).

This runner tests the routed branch only. MCore still uses the existing
FlashInfer/NVLS implementation; there is no new full-model command switch yet.
The latent projections, RMSNorm, shared expert, shortcut scheduling and wide
residual remain the integration boundary for the next phase.

## First run in the GB300 development container

Use the Python environment that already runs NT4. Point `PYTHONPATH` at this
modified FlashInfer checkout, and check the printed `flashinfer_source` path.
Do not install or upgrade dependencies on the Mac. Unlike the counted-write
prototype, this path uses the existing NVSHMEM/CuTe DSL runtime and does not
request NCCL 2.32 counted APIs.

From the modified FlashInfer checkout:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  examples/moe_ep/nt4_megamoe_bringup.py --mode preflight
```

Use your `$PY` interpreter in place of `python` if needed. Preflight requires
CUDA, data-center Blackwell, and importable FlashInfer, CuTe DSL, NVSHMEM and
CUDA bindings. It prints versions and paths but does not prove the fabric works.

Run the SwiGLU baseline first. This checks the communication,
compiler and numerical oracle independently of our NT4 changes:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  examples/moe_ep/nt4_megamoe_bringup.py --mode baseline
```

Then run the NT4 activation/layout variant:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  examples/moe_ep/nt4_megamoe_bringup.py --mode nt4
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  examples/moe_ep/nt4_megamoe_bringup.py --mode nt4 --graph --prequantized
```

After those pass, use the actual routed-branch dimensions:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  examples/moe_ep/nt4_megamoe_bringup.py --mode nt4 --full-shape --checks 16
```

Save stdout and stderr. Each rank has a 600-second watchdog; increase
`--timeout-seconds` if first compilation needs longer. Failure is an error,
never an automatic fallback or a passing skip. Run these commands separately
so a failed dependency/control stops the sequence before NT4 validation.

## What passing means

The small case uses H=128, I=256, E=16, top-10, EP4. The full case uses
H=768, I=4096, E=512, top-10, EP4, with four local token slots. Weights are
sparse power-of-two matrices with expert-dependent permutations so an
independent oracle can check remote expert identity and channel addressing
without gathering the full expert weights. Their MXFP8 representation is exact.

Inputs and routing change on every check; rank zero cycles through fully valid,
fully masked, and partially masked routing. Repeated identical inputs must
produce bit-identical outputs. Graph mode
changes data at fixed addresses between replays. The reference independently
evaluates the sparse expert math and includes input and intermediate MXFP8
round trips and BF16 expert-output rounding. It shares the existing host MXFP8
quantizer, so this is not independent qualification of that quantizer.

### Dense random weight gate

Sync the updated runner, then run these separately, stopping on a failure:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  examples/moe_ep/nt4_megamoe_bringup.py \
  --mode nt4 --weights dense --checks 8 --trace

python -m torch.distributed.run --standalone --nproc-per-node=4 \
  examples/moe_ep/nt4_megamoe_bringup.py \
  --mode nt4 --weights dense --graph --prequantized --checks 8 --trace

python -m torch.distributed.run --standalone --nproc-per-node=4 \
  examples/moe_ep/nt4_megamoe_bringup.py \
  --mode nt4 --weights dense --full-shape --prequantized --checks 16 --trace
```

Repeat the small dense eager case with `--seed 4321 --input-scale 8` to exercise
larger FC1 activations and the nonlinear clamp. Default seed is 1234 and input
scale is 1; both are logged. This is synthetic stress, not a checkpoint input
distribution. Baseline mode also supports dense weights with duplicated
gate/up legs, as in the sparse infrastructure control.

Each expert gets independently seeded Gaussian BF16 matrices, scaled by the
inverse square root of the respective reduction dimension. The reference
regenerates one selected expert at a time using its global ID; it never gathers
all expert weights or reads the backend's packed weights. It quantizes and
dequantizes canonical weights in blocks of 32 along the reduction dimension,
then uses FP32 GEMMs with TF32 disabled. Input and weighted post-activation
values also undergo MXFP8 round trips; each expert's FC2 output is rounded to
BF16 before the FP32 top-k sum and final BF16 cast.

Dense routes explicitly span all four expert owners, including at full shape.
Empty and partially masked routing, changed inputs, and exact repeated-output
checks remain enabled. Each check prints relative L2 and maximum absolute error.
The acceptance policy is `bf16_fp32_contributions_v2`, described below. Small
reference/GPU GEMM differences can cross quantization rounding boundaries;
diagnose any contribution-gate failure before changing that policy.
The reference shares the host quantizer and deterministic weight generator
with setup; neither is independently qualified by this test.

### Masked-route combine correction

The first GB300 baseline run passed epoch zero but returned nonzero output on
the fully masked rank at epoch one. The source skips dispatch for masked routes,
leaves their combine slots unwritten, and sums every slot in the final reduction.
Previously written values can therefore survive into the next output.

The shared MXFP8 shim now launches a small CuTe kernel before each invocation
to zero only invalid-route BF16 combine slots. It makes no stores to valid
slots, which peers may already be writing. The preceding invocation's completion
barrier finishes its peer writes before this clear. The wrapper covers eager
execution, launch thunks, and CUDA graph capture without modifying vendor source.
This adds one launch and routing-mask reads; include that cost in later timings.
The user reported the subsequent sparse runs passing, including graph replay.
The full-shape dense eager and second-seed graph runs also subsequently passed
under the v2 numerical gate, as reported by the user.

### Acceptance policy: bf16_fp32_contributions_v2

The full-shape dense run exposed one output mismatch: expert 395 contributed
-6.71875 instead of -6.75, one adjacent BF16 value. Nine other contributions
matched exactly, and recombining the kernel terms reproduced its output exactly.
Cancellation amplified that difference (absolute contribution sum 32.9883,
reference output -0.19140625); global relative L2 was 7.30e-5. FP32 and FP64
references agreed. This isolates the discrepancy to an expert contribution,
but does not identify its internal arithmetic origin.

With user approval, acceptance now checks **all** expert contributions on every
iteration, including iterations whose final outputs pass the old comparison:

- Each contribution gets one directional BF16 step plus an FP32 accumulation
  allowance `2*gamma_K*sum(abs(a*w))`, where `gamma_K=K*u/(1-K*u)`,
  `u=2^-24`, and K is the FC2 reduction dimension. The reference computes
  absolute-product sums in FP64. This is a conservative bound for two FP32
  accumulation orders with identical operands, not proof of the kernel error
  source; it does not cover arbitrary upstream quantization differences.
  There is no fixed absolute-error floor. Exact cancellation can have a
  nonzero budget; a sum with no nonzero products has none.
- Masked contributions and outputs with no nonzero reference products must
  be exactly zero; all values must be finite.
- The BF16 output must exactly equal an FP32 sum of the captured kernel terms
  followed by BF16 rounding. Exact repeated-output checks remain in place.
- The output absolute-error budget is the sum of the per-term BF16 and FP32
  allowances plus half the larger adjacent spacing at each of the actual
  and reference outputs, accounting for the two final BF16 rounding operations.
- Per-expert output-vector and global relative L2 must both remain <= 0.015,
  independently of the conservative per-element accumulation budget.

The old output comparison (rtol=0.08, atol=0.002) is diagnostic: failures still
print `reference_mismatch`, and PASS records include `strict_output_mismatches`.
PASS means the new contribution policy passed, not that the old comparison did.
The one-step-only v1 policy rejected expert 387: its FC2 absolute-product
sum was 97.77 but its output was about -2.31e-5. The observed 1.31e-6
difference was only 1.34e-8 of that product sum. V2 accounts for this internal
cancellation as well as cancellation across experts.

This is a provisional numerical gate, not a checkpoint-quality guarantee or
proof that every one-step difference is harmless. Multi-seed, larger-input,
broader graph coverage and checkpoint qualification remain necessary. Workspace snapshots add
diagnostic overhead, so this runner must not be used for performance claims.

`--prequantized` supplies canonical MXFP8 weights and E8M0 scale planes to
exercise their packing path. It uses the same synthetic weights; it does not
load or qualify the NT4 checkpoint.

## Next gates

1. Baseline and NT4 sparse cases: user-reported passing on GB300.
2. Full-shape dense eager and second-seed graph replay: user-reported passing
   under v2. Next: real checkpoint-layer validation and a separate timing harness
   following [the checkpoint plan](nt4_megamoe_checkpoint_plan.md).
3. Add the MCore opt-in backend at the existing router/latent-projection
   boundary, with no duplicate outer NVLS dispatch/combine. Preserve actual
   checkpoint bytes/scales and per-layer BF16 overrides; do not assume the
   global MXFP8 flag means every layer is quantized.
4. Compare one layer against #5696 plus NVLS, then full-model outputs and
   decode timing with the user's graph-disabled NT4 command. The training
   script enables shortcut parallelism; the supplied inference command does
   not. Preserve that distinction in the first comparison.
5. Port the proven numerical contract to SM107 and validate/tune on Rubin.

No performance claim follows from this correctness runner. It does
not measure a production MoE layer, and the baseline mode is SwiGLU, not an
NT4 performance comparator.

## Tests

CPU contract checks, with no CUDA/FlashInfer import:

```bash
python tests/moe_ep/test_nt4_megamoe_config.py
```

CPU acceptance-policy tests (requires PyTorch, no CUDA):

```bash
python tests/moe_ep/test_nt4_megamoe_validation.py
```

The hardware runner is also exposed through
`tests/moe_ep/test_nt4_megamoe_multirank.py` for torchrun/pytest. Direct runner
commands are preferable for the first bring-up because missing capability
fails instead of being reported as a pytest skip.
