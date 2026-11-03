"""Conservative backend-level restart bookkeeping for the NT4 clamped comparison."""

import atexit
import fcntl
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ResumeRun:
    def __init__(self, args, checkout):
        self.args = args
        self.root = args.output
        self.lock = (self.root / ".run.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError("Another resume-aware driver owns this output directory")
        atexit.register(self.lock.close)
        config = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k not in ("resume", "resume_legacy", "timeout_seconds")
        }
        config["prompt_sha256"] = fingerprint(args.prompt_file)
        config["manifest_sha256"] = fingerprint(args.source_artifact / "manifest.json")
        import os

        config["environment"] = {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("NT4_", "FLASHINFER_", "NCCL_", "NVSHMEM_"))
            and not k.endswith(("_ROOT", "_DIR"))
        }
        config["trtllm_export"] = os.environ.get("FLASHINFER_TRTLLM_NT4_EXPORT")
        path = self.root / "resume-config.json"
        self.legacy = not path.exists() and args.resume
        if path.exists() and json.loads(path.read_text()) != config:
            raise RuntimeError(
                "Resume configuration differs; use original options/environment or a fresh output"
            )
        if self.legacy and not args.resume_legacy:
            raise RuntimeError(
                "This older run has no resume provenance. Add --resume-legacy only after confirming original options, export and environment; legacy commands/rounds will still be checked."
            )
        write_json(path, config)
        origin = self.root / "resume-origin.json"
        if not origin.exists():
            write_json(origin, dict(legacy_provenance_unverified=self.legacy))
        self.report = dict(
            resumed=args.resume,
            legacy_provenance_unverified=json.loads(origin.read_text())[
                "legacy_provenance_unverified"
            ],
            trials={},
            failures=[],
            control_drift_limit=0.03,
            comparison_valid=False,
        )
        self.report["checker_sha256"] = fingerprint(
            checkout / "examples/moe_ep/nt4_expert_diagnostics.py"
        )
        self.archive = self.root / "attempt-history" / str(time.time_ns())
        for name in ("summary.json", "resume-status.json"):
            self.preserve(self.root / name)
        self.save()

    def preserve(self, path):
        if path.exists():
            self.archive.mkdir(parents=True, exist_ok=True)
            path.rename(self.archive / path.name)

    def save(self):
        write_json(self.root / "resume-status.json", self.report)

    def reusable(self, name, command):
        root = self.root
        command_path = root / f"{name}-command.json"
        if command_path.exists() and json.loads(command_path.read_text()) != command:
            raise RuntimeError(f"{name}: saved launch command differs; refusing reuse")
        receipt = root / f"{name}-exit.json"
        if receipt.exists() and json.loads(receipt.read_text()).get("returncode") != 0:
            return False
        if not receipt.exists() and not self.legacy:
            return False
        try:
            if not command_path.exists():
                return False
            log = (root / f"{name}.log").read_text(errors="replace")
            if log.count('"stage": "e2e_cleanup_complete"') < 4:
                return False
            if any(
                s in log
                for s in (
                    "Traceback (",
                    "SIGSEGV",
                    "Segfault encountered",
                    "ChildFailedError",
                )
            ):
                return False
            rows = json.loads((root / name / "results.json").read_text())
            measured = [r for r in rows if not r.get("warmup", True)]
            if sorted(r["round"] for r in measured) != list(
                range(1, self.args.rounds + 1)
            ):
                return False
            for r in measured:
                if (
                    r.get("precision") != self.args.precision
                    or r.get("model_graphs") != self.args.model_graphs
                    or r.get("shortcut_overlap") != self.args.shortcut_overlap
                    or r.get("model_graph_vs_eager_outputs") != "EXACT"
                    or r.get("requested_batch_reached") is not True
                    or r.get("peak_decode_active_requests") != self.args.request_count
                    or r.get("total_output_tokens")
                    != self.args.request_count * self.args.num_tokens_to_generate
                    or r.get("gc_policy", {}).get("policy")
                    != "collect_before_round_disable_during_round"
                    or not math.isfinite(r["output_tokens_per_second"])
                    or r["output_tokens_per_second"] <= 0
                ):
                    return False
                outputs = json.loads(
                    (root / name / f"outputs-round{r['round']}.json").read_text()
                )
                if len(outputs) != self.args.request_count:
                    return False
            if not receipt.exists():
                write_json(
                    receipt, dict(returncode=0, legacy_inferred_from_complete_logs=True)
                )
            return True
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def prepare(self, name, command):
        reuse = self.args.resume and self.reusable(name, command)
        # Always obtain a current closing control, even for a completed run.
        if reuse and name != "baseline-end":
            self.report["trials"][name] = "REUSED"
            self.save()
            return False
        if self.args.resume:
            for suffix in ("", ".log", "-command.json", "-exit.json"):
                self.preserve(self.root / (name + suffix))
        self.report["trials"][name] = "RUNNING"
        self.save()
        return True

    def finished(self, name, code):
        receipt = self.root / f"{name}-exit.json"
        write_json(
            receipt, dict(returncode=code, checker_sha256=self.report["checker_sha256"])
        )
        if code == 0:
            command = json.loads((self.root / f"{name}-command.json").read_text())
            if not self.reusable(name, command):
                code = 65
                write_json(
                    receipt,
                    dict(
                        returncode=code,
                        reason="Incomplete or invalid measured-round evidence",
                    ),
                )
        self.report["trials"][name] = "PASS" if code == 0 else "FAIL"
        if code:
            self.report["failures"].append(name)
        self.save()
        return code

    def finish(self):
        try:
            medians = []
            for name in ("baseline-start", "baseline-end"):
                rows = json.loads((self.root / name / "results.json").read_text())
                medians.append(
                    statistics.median(
                        r["output_tokens_per_second"]
                        for r in rows
                        if not r.get("warmup", True)
                    )
                )
            drift = abs(medians[1] / medians[0] - 1)
            self.report["control_throughput"] = medians
            self.report["control_drift"] = drift
            self.report["comparison_valid"] = (
                not self.report["failures"] and drift <= 0.03
            )
        except (OSError, ValueError, KeyError, TypeError, statistics.StatisticsError):
            self.report["control_drift"] = None
        self.save()
        return self.report
