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

## Internal timing diagnostic

`NT4_EXPERT_INTERNAL_TIMING=1` adds a local workspace region outside the counter
reset prefix and lane-0 globaltimer stores per CTA/warp. Dispatch/packing,
FC1/FC2 epilogue, and tail lifetimes are concurrent and include waits. The tail
interval includes its initial rendezvous, not just zeroing. Instrumentation is
limited to 1024 CTAs and 16 warps per CTA; the launch checks CTA capacity. This is
an owned-shim diagnostic, not a change to the vendor drop. GPU validation pending.

## Detached NT4 MXFP8 tail prototype

`NT4_DETACHED_TAIL=1` retires the large compute CTAs after their role drains and
CTA rendezvous. A same-stream, single-CTA/128-thread kernel performs the existing
peer completion barriers and shared/local counter resets before top-k reduction.
The prototype retains epilogue peer stores; it does not implement counted writes,
Lamport readiness, asynchronous return workers, or a mixer launch signal. There
is no claim of Rubin support or speedup. It is a resource-handoff baseline that
can be tested on GB300 before introducing Rubin-specific APIs.

Only NT4 MXFP8, 1-CTA clusters, epilogue stores and separate reduction are allowed.
IKET/NCU replay are rejected. The flag enters the compile key. Workspaces must not
be reused concurrently; all ranks must use the same mode. The two peer barriers
remain around the shared reset; local reset is last. The finalizer has no software
grid rendezvous and uses the existing sense-reversing peer protocol.

Run `bash examples/moe_ep/nt4_detached_tail.sh` with the ordinary checkout/MCore/log
variables. It gates six bracketed serial/overlap arms on direct numerical checks
and hybrid graph bucket/changed-routing checks, stopping on failure. Uses existing
matched inputs; reports are diagnostic block timings, not throughput. Local syntax
checks do not validate CUDA codegen, cross-PE ordering or graph safety: GPU validation
is required before treating this path as working. No vendor-source modifications.

### MMA wait diagnostic

`NT4_EXPERT_INTERNAL_TIMING=1` also records per-CTA MMA-warp counters:
scheduler consumption, accumulator acquire, operand wait, residual issue and
bookkeeping, producer drain, and FC1/FC2 tile counts. Counters accumulate in
registers and are written once at warp exit. Timer reads and extra registers
perturb execution. Residual issue includes scale copies, pipeline probes and
async MMA submission; it is **not tensor-core execution time**. Operand waits
can include FC1-to-FC2 readiness and memory latency; this diagnostic does not
separate those causes. Do not sum CTAs. Existing numerical and changed-route
graph checks remain required. CPU parser tests do not establish GPU validity.

The handoff diagnostic additionally measures FC2 readiness spin plus acquire
fences in TMA-A, staging-slot acquire, staging issue/bookkeeping, and separate
MMA activation/weight waits. TMA and MMA counters overlap across warps; do not
subtract them to estimate transfer latency. MMA waits are observed sequentially
(A then B), so B reports residual waiting after A. No completion barriers are
added for timing. Existing fences and dependency counters are unchanged.

Timing version 4 includes both initial/lookahead `try_wait()` probes and final
`wait_and_advance()` in operand waits. FC2 producer-slot timing likewise includes
`try_acquire()` probes. Earlier residual issue counters included probe costs; do
not compare their attribution directly against v4. No pipeline calls are moved.

### Lookahead probe ordering experiment

`NT4_LOOKAHEAD_AFTER_MMA=1` moves next-stage A/B consumer probes after the
current tile's scale copies, MMA submission and operand releases. Default is
unchanged; this flag enters the compile key and only affects the NT4 shim.
No dependencies or completion barriers are removed. Test numerical/graph
replays before timing; compare uninstrumented serial and overlap controls on
the same node. A gain would implicate probe placement, not establish pure GEMM
throughput or an E2E deployment win. GPU validation pending.

Timing v5 adds FC1 epilogue counters per warp: accumulator-ready wait,
ready-to-first-subtile-release, and task work after that release. The latter
includes remaining TMEM loads, activation/quantization and output-store issue;
it excludes the later FC1-done publication. Counts are accumulated over FC1
tiles, and roles overlap with the MMA warp. Global counter writes perturb the
epilogue: these are attribution diagnostics, not speed measurements.
Only 1-CTA clusters and the existing overlapping-accumulator path are supported.
No accumulator release, fence, or dependency is moved. GPU validation pending.

### Packed FC1 activation experiment

`NT4_PACKED_FC1_ACTIVATION=1` pairs independent channels for FP32 clamp
scaling, squaring and optional routing-weight multiplication. Division, tanh
(`approx=False`), quantization, stores and synchronization are unchanged.
The default is off; the flag enters the compile key and applies only to the
NT4 MXFP8 epilogue. BF16 retains its own activation override.
`examples/moe_ep/nt4_fc1_packed.sh` runs serial/overlap matched-input arms
with controls before and after, numerical/graph gates enabled and internal
counters disabled. Coarse phase markers remain: this is a branch diagnostic,
not an E2E throughput result. GPU qualification and speedup are pending.

`NT4_KERNEL_DUMP_DIR` enables opt-in PTX/cubin export for rank 0's MegaMoE
compilations only. Use a node-local directory. Each compile configuration has
its own directory and config record; ordinary execution does not export files.
`nt4_fc1_assembly.sh` runs the control and packed variants with numerical gates,
disassembles the cubins with `cuobjdump`, and archives the local files once.
Static instruction counts are not dynamic execution counts or a latency profile;
resource reports must be read alongside warp-specific `setmaxnreg` instructions.

### Hardware tanh experiment

`NT4_HARDWARE_TANH=1` selects `cute.math.tanh(..., approx=True)` only in
NT4 MXFP8 FC1 activation. Default off, separate compile key; clamp scaling,
squaring, routing weight placement, quantization and barriers are unchanged.
This changes numerical approximation and requires qualification. The runner
`nt4_fc1_tanh.sh` first screens the activation against CPU FP64 (explicit 2e-6
absolute tanh tolerance), exercises FP8 midpoint neighbors and changed-input
graph replay, then runs control/candidate/control serial/overlap arms with
all existing numerical gates. No tolerance is relaxed in the full kernel.
Synthetic FP8 comparisons use unit-scale blocks and are diagnostic, not
qualification of arbitrary block scales. GPU validation and speedup pending.

The synthetic hardware-tanh screen measured max absolute tanh error 7.52e-6,
exceeding its conservative 2e-6 threshold. `--explore-approximation` records this
as `SCREEN_EXCEEDED_EXPLORATORY`, retains nonfinite and exact graph checks, and
reports unit-scale FP8 changes before allowing exploration. The accurate-tanh
control must still pass the original screen. `nt4_fc1_tanh.sh` now uses this
explicit mode; full real-input expert/output gates remain unchanged and mandatory.
No speed or full-model accuracy claim has been established for hardware tanh.

`nt4_fc1_tanh_failure.sh` replays only the failing hardware-tanh serial arm.
`NT4_FC1_FAILURE_DIR` enables rank-0 failure-only reporting in checkpoint replay:
exact row-relative L2, changed scales, raw FP8 and dequantized values for the
eight largest-error blocks, and identical-weight FP64 FC2 propagation. One
snapshot per failing expert preserves operands and quantized intermediates;
actual prequant kernel accumulators are not captured. The original gate still
raises, so exit 1 is expected when the numerical rejection is reproduced.

### Detached readiness boundary diagnostic (October 29)

This complements the activation/expert-phase experiment; do not repeat worker
sweeps or the attached/detached matrix. Existing matched timelines already
observe local expert API end, FP32 partial assembly, the inclusive NVLS combine,
output cast, mixer phases and join. The hybrid's local MegaMoE has world_size=1;
its detached peer barriers are not the EP4 NVLS readiness boundary.

With `NT4_EXPERT_INTERNAL_TIMING=1`, the detached finalizer additionally writes
five globaltimer markers into reserved internal_ticks[0,15,32:37]: entry,
first peer barrier done, second peer barrier entry (after the existing CTA
rendezvous), second peer barrier done, and thread-0 local-reset completion.
No new synchronization or reset is added. The old tail_cleanup role still means
compute-tail rendezvous, not this detached finalizer. Thread-0 completion and
the last compute-tail stamp are not exact hardware CTA-exit timestamps.

For selected layers 7/32/68, `NT4_READINESS_TIMING=1` with
`NT4_EXPERT_PHASE_TIMING=1 NT4_EXPERT_INTERNAL_TIMING=1 NT4_DETACHED_TAIL=1`
also allocates a graph-stable NVLS timestamp buffer before capture. It requires
the matching MCore optional `readiness_ticks` hook in
`torch_symm_triton/variable_collectives.py`. The default collective path is
unchanged. Each active reduce-scatter CTA records pre-barrier, post-existing-
barrier/rendezvous, and reduction-issue-end markers. No extra barriers, launches,
fences, altered grid, or arithmetic are introduced inside the collective.
Skipped CTA records must not be consumed; this hybrid uses a fixed capacity.

Per-invocation JSONL now includes `expert_readiness`: finalizer boundaries,
local world size, and individual NVLS CTA intervals. Readers reject stale,
missing and out-of-parent timestamps instead of inferring readiness. The NVLS
barrier interval includes rank arrival, scheduling, signal exchange and existing
CTA synchronization; it is not pure network latency. The last issue marker is
not a proof of store visibility or safe reuse. The enclosing stream completion
remains authoritative. Do not infer earlier per-token readiness from these
whole-operation barriers, and do not reset or retire buffers based on markers.

Only collect one detached, unchanged-computation matched replay when this
instrumentation is GPU-qualified. Compare exposed barrier intervals with the
already recorded mixer completion on each invocation. Keep these diagnostics
out of performance claims. CPU parser tests pass; CuTe codegen, NVLS GPU
correctness, and graph replay with these new markers remain to be qualified.
