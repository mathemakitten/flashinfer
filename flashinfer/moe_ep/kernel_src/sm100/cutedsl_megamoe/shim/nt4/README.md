# NT4 non-gated MXFP8 source prototype

**Early weighting passed synthetic GPU checks; late weighting awaits GPU validation.** This is an explicit development
variant, selected by `activation="clamped_relu2"` and
`clamped_relu2_limit=16.0` on the SM100 MXFP8 backend config.

These three implementation files are owned adaptations of the corresponding
files under `../../src/moe_mxfp8_glu` at FlashInfer commit `50d583aa0`.
The original kernel-team drop remains verbatim. Full copies are currently
necessary because FC1 width assumptions occur in the base kernel's JIT launch,
workspace layout, and epilogue. There is no process-global monkey-patching or
runtime source rewriting. A subsequent drop requires an explicit diff review
of these copies as well as the ordinary shim API audit.

## Changes from the gated implementation

- FC1 weights have `I` output rows, with no gate/up interleave; FC2 retains `I`
  input columns. Shared scheduler work counts receive the actual FC1 width.
- The FC1 epilogue emits all 256 columns per tile, in eight 32-value blocks,
  instead of collapsing paired gate/up blocks into 128 columns.
- Activation is `p * (s * tanh(max(z, 0) / s))**2` in FP32, followed by the
  existing MXFP8 quantizer and E8M0 scale generation. No extra BF16 rounding
  is introduced before quantization. This contract still requires comparison
  with #5696's production path and the checkpoint's numerics.
- `routing_weight_placement="before_quantization"` preserves that original
  behavior. `"after_fc2"` selects unweighted activation quantization followed by
  BF16 FC2 output and FP32 weighted top-k reduction, to compare with CUTLASS's
  late-weighting baseline. It uses the existing `apply_topk_in_fc1=False` path;
  the vendor source remains unchanged. Placement participates in compile and
  workspace keys. This does not establish parity with training numerics.
- Four shared-memory stages hold one complete output tile per epilogue warp.
  Eight output scales are addressed through the blocked layout individually;
  the two groups of four scales are not assumed contiguous in memory.
- The original FC2 and NVLink communication protocol are retained. This code
  does not implement counted writes.
- Separate top-k reduction is required initially. Intermediate width must be
  divisible by 256. Only the ordinary inference path is exposed; raw-C debug
  generation and alternative combine encodings are not qualified.

SwiGLU remains the default. Activation and limit participate in compile and
workspace keys. NT4 bypasses the SwiGLU tuning cache; `knobs={}` means use the
frontend defaults, not a measured optimum.

Run instructions and remaining integration gates are in
[the GB300 bring-up guide](../../../../../../../examples/moe_ep/nt4_megamoe_bringup.md).

## BF16 NT4 variant (GPU validation pending)

The owned `megamoe_kernel_bf16.py` and `kernel_bf16_glu_fc12.py` adapt
`src/moe_bf16_glu` without editing vendor files. FC1 and FC2 both use width I;
FC1 has no gate/up interleave. `epilogue_bf16.py` inherits the existing owned
NT4 non-gated activation/layout and disables scale-factor state and overlapping
accumulation. Four shared-memory stages hold complete per-warp output tiles.
FC1's FP32 activation is rounded to BF16 for FC2; each FC2 term is rounded to
BF16 before the separate FP32 late-weighted reduction. No MXFP8 quantization
is used. Raw-C generation and alternative reduction modes are not exposed.

Select `activation="clamped_relu2", clamped_relu2_limit=16.0, knobs={}` on
`Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig`. The original SwiGLU default is
unchanged. BF16 retains its validated upstream geometry (256,256,64), cluster
(2,1,1); MXFP8's M128 tuning is not assumed valid. Activation and clamp are in
compile/workspace keys. The separate reducer masks invalid expert slots.

Run `torchrun --standalone --nproc-per-node=4 examples/moe_ep/nt4_bf16_bringup.py`
first (the E2E sweep does this automatically). CPU tests do not qualify this
prototype for performance or numerical claims. The BF16 reference gate is
provisional: per-contribution relative L2 <= 1% plus 1e-5 absolute L2 allowance,
existing overall/per-row L2 gate, exact recombination and poisoned repeat, and
live-baseline comparison. No thresholds are relaxed on failure.
