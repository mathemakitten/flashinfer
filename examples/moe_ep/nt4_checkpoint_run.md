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
