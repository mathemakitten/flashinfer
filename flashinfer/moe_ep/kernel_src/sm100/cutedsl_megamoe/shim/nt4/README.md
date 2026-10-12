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
