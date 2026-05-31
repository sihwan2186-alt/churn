from __future__ import annotations

import json
import hmac
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


WORKSPACE = Path(os.environ.get("CHURN_WORKSPACE", "/workspace")).resolve()
HOST = os.environ.get("CHURN_RUNNER_HOST", "0.0.0.0")
PORT = int(os.environ.get("CHURN_RUNNER_PORT", "8000"))
STEP_TIMEOUT_SEC = int(os.environ.get("CHURN_STEP_TIMEOUT_SEC", "1800"))
DEFAULT_API_KEY = "churn-radar-secret-2026"
API_KEY = os.environ.get("CHURN_RUNNER_API_KEY", DEFAULT_API_KEY)
LOG_FILE = WORKSPACE / "runner.log"

FULL_REPRODUCTION_SCRIPTS = [
    "preprocess_churn.py",
    "phase_3b_differentiation_experiments.py",
    "phase_4_cross_validation.py",
    "phase_5a_interpretability.py",
    "phase_5b_business_impact.py",
    "phase_6_extended_case_studies.py",
    "phase_7_next_experiments.py",
    "phase_8_statistical_validation.py",
    "make_presentation_assets.py",
    "make_final_ppt.py",
]

PPT_SCRIPTS = [
    "make_presentation_assets.py",
    "make_final_ppt.py",
]

KEY_FILES = [
    "ChurnRadar_Final_Presentation.pptx",
    "FINAL_REPORT.md",
    "PRESENTATION_SLIDES.md",
    "PROJECT_FILE_SUMMARY.md",
    "PHASE_8_STATISTICAL_VALIDATION.md",
    "processed/phase_8_statistical_validation/bootstrap_metric_ci.csv",
    "processed/phase_8_statistical_validation/mcnemar_paired_tests.csv",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tail_text(value: str, limit: int = 12000) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


def subprocess_output_to_text(value: str | bytes | bytearray | memoryview | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return bytes(value).decode("utf-8", "replace")


def response_payload(ok: bool, **extra: Any) -> dict[str, Any]:
    payload = {
        "ok": ok,
        "workspace": str(WORKSPACE),
        "timestamp_utc": utc_now(),
    }
    payload.update(extra)
    return payload


def write_log(message: str) -> None:
    """영속적 로그 파일 기록"""
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"[{utc_now()}] {message}\n")
    except Exception:
        pass


def run_script(script_name: str) -> dict[str, Any]:
    script_path = WORKSPACE / script_name
    started = time.monotonic()
    if not script_path.exists():
        return {
            "script": script_name,
            "ok": False,
            "returncode": None,
            "duration_sec": 0,
            "error": f"Missing script: {script_path}",
            "stdout_tail": "",
            "stderr_tail": "",
        }

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"

    try:
        completed = subprocess.run(
            [sys.executable, "-u", str(script_path)],
            cwd=WORKSPACE,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=STEP_TIMEOUT_SEC,
            check=False,
        )
        duration = round(time.monotonic() - started, 3)
        write_log(f"Finished script: {script_name} (Code: {completed.returncode}, Duration: {duration}s)")
        return {
            "script": script_name,
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "duration_sec": duration,
            "stdout_tail": tail_text(subprocess_output_to_text(completed.stdout)),
            "stderr_tail": tail_text(subprocess_output_to_text(completed.stderr)),
        }
    except subprocess.TimeoutExpired as exc:
        duration = round(time.monotonic() - started, 3)
        write_log(f"Timeout script: {script_name} after {STEP_TIMEOUT_SEC}s")
        return {
            "script": script_name,
            "ok": False,
            "returncode": None,
            "duration_sec": duration,
            "error": f"Timed out after {STEP_TIMEOUT_SEC} seconds",
            "stdout_tail": tail_text(subprocess_output_to_text(exc.stdout)),
            "stderr_tail": tail_text(subprocess_output_to_text(exc.stderr)),
        }


def run_scripts(script_names: list[str]) -> dict[str, Any]:
    started = time.monotonic()
    steps = []
    for script_name in script_names:
        result = run_script(script_name)
        steps.append(result)
        if not result["ok"]:
            break

    ok = all(step["ok"] for step in steps) and len(steps) == len(script_names)
    return response_payload(
        ok,
        duration_sec=round(time.monotonic() - started, 3),
        steps=steps,
    )


def count_files() -> dict[str, int]:
    counts: dict[str, int] = {}
    excluded_dirs = {".git", ".venv", "__pycache__"}
    for root, dirs, files in os.walk(WORKSPACE):
        dirs[:] = [name for name in dirs if name not in excluded_dirs]
        for file_name in files:
            if file_name.startswith(".") and file_name.count(".") == 1:
                suffix = file_name.lower()
            else:
                suffix = Path(file_name).suffix.lower() or "[no_extension]"
            counts[suffix] = counts.get(suffix, 0) + 1
    return dict(sorted(counts.items()))


def ppt_status() -> dict[str, Any]:
    ppt = WORKSPACE / "ChurnRadar_Final_Presentation.pptx"
    status: dict[str, Any] = {
        "exists": ppt.exists(),
        "path": str(ppt),
    }
    if not ppt.exists():
        return status

    status["size_bytes"] = ppt.stat().st_size
    status["modified_utc"] = datetime.fromtimestamp(
        ppt.stat().st_mtime,
        timezone.utc,
    ).isoformat()

    try:
        from pptx import Presentation

        status["slide_count"] = len(Presentation(str(ppt)).slides)
    except Exception as exc:  # pragma: no cover - defensive runtime check
        status["slide_count_error"] = str(exc)
    return status


def read_json_file(relative_path: str) -> Any:
    path = WORKSPACE / relative_path
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - summary should not fail
        return {"error": str(exc)}


def build_summary() -> dict[str, Any]:
    key_file_status = {}
    for relative_path in KEY_FILES:
        path = WORKSPACE / relative_path
        key_file_status[relative_path] = {
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
        }

    return response_payload(
        True,
        file_counts=count_files(),
        key_files=key_file_status,
        ppt=ppt_status(),
        phase8_summary=read_json_file(
            "processed/phase_8_statistical_validation/"
            "phase_8_statistical_validation_summary.json"
        ),
        project_summary_exists=(WORKSPACE / "PROJECT_FILE_SUMMARY.md").exists(),
    )


class ChurnRunnerHandler(BaseHTTPRequestHandler):
    server_version = "ChurnRunner/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args))

    def check_auth(self) -> bool:
        auth_header = self.headers.get("X-API-KEY", "")
        return hmac.compare_digest(auth_header, API_KEY)

    def send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def read_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length == 0:
            return {}
        raw = self.rfile.read(content_length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def do_GET(self) -> None:
        if self.path == "/health":
            payload = response_payload(
                WORKSPACE.exists(),
                python=sys.version,
                scripts_available={
                    script: (WORKSPACE / script).exists()
                    for script in FULL_REPRODUCTION_SCRIPTS
                },
            )
            self.send_json(200 if payload["ok"] else 500, payload)
            return

        if not self.check_auth():
            self.send_json(401, response_payload(False, error="Unauthorized"))
            return

        if self.path == "/summary":
            self.send_json(200, build_summary())
            return

        self.send_json(404, response_payload(False, error=f"Unknown GET path: {self.path}"))

    def do_POST(self) -> None:
        if not self.check_auth():
            self.send_json(401, response_payload(False, error="Unauthorized"))
            return

        self.read_body()

        if self.path == "/run/full-reproduction":
            payload = run_scripts(FULL_REPRODUCTION_SCRIPTS)
            self.send_json(200 if payload["ok"] else 500, payload)
            return

        if self.path == "/run/statistical-validation":
            payload = run_scripts(["phase_8_statistical_validation.py"])
            self.send_json(200 if payload["ok"] else 500, payload)
            return

        if self.path == "/run/ppt":
            payload = run_scripts(PPT_SCRIPTS)
            payload["ppt"] = ppt_status()
            self.send_json(200 if payload["ok"] else 500, payload)
            return

        if self.path == "/run/monitor-drift":
            payload = run_scripts(["monitor_drift.py"])
            self.send_json(200 if payload["ok"] else 500, payload)
            return

        self.send_json(404, response_payload(False, error=f"Unknown POST path: {self.path}"))


def main() -> None:
    if not WORKSPACE.exists():
        raise SystemExit(f"Workspace does not exist: {WORKSPACE}")
    server = ThreadingHTTPServer((HOST, PORT), ChurnRunnerHandler)
    print(f"ChurnRunner listening on http://{HOST}:{PORT}")
    print(f"Workspace: {WORKSPACE}")
    server.serve_forever()


if __name__ == "__main__":
    main()
