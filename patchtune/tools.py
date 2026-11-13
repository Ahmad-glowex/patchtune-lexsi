"""The four agent tools for PatchTune-mini, executed in a sandbox directory.

Tools
-----
- ``read_file(path)``     -> file contents
- ``search_code(query)``  -> substring matches across .py files
- ``run_tests()``         -> runs the toy repo's unittest suite
- ``apply_patch(diff)``   -> applies a unified diff

Design notes
------------
* All paths are confined to the sandbox root (``..`` escapes are rejected).
* :func:`validate_call` performs *structural* validation only (known tool,
  required args present, correct types, diff parses). The reward module
  uses it to score well-formedness without executing anything.
* Every dispatch result is a dict with at least ``ok`` (bool). Malformed
  calls return ``{"ok": False, "malformed": True, "error": ...}``;
  well-formed calls that fail (e.g. patch does not apply) return
  ``{"ok": False, "malformed": False, ...}``. The distinction matters:
  malformed = the model produced an invalid call; failed = valid call,
  wrong content.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

TOOL_SPECS: dict[str, dict[str, type]] = {
    "read_file": {"path": str},
    "search_code": {"query": str},
    "run_tests": {},
    "apply_patch": {"diff": str},
}


# --------------------------------------------------------------------------
# Unified-diff parsing / application
# --------------------------------------------------------------------------

class PatchError(ValueError):
    """Raised when a unified diff cannot be parsed or applied."""


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_unified_diff(diff: str) -> dict[str, list[tuple[int, list[str]]]]:
    """Parse a unified diff into {path: [(start_line_1based, hunk_lines)]}.

    Hunk lines keep their leading ' ', '-' or '+' marker. Raises PatchError
    on structurally invalid input.
    """
    if not isinstance(diff, str) or "--- " not in diff or "+++ " not in diff:
        raise PatchError("diff lacks ---/+++ file headers")
    files: dict[str, list[tuple[int, list[str]]]] = {}
    current: str | None = None
    lines = diff.splitlines()
    i = 0
    saw_hunk = False
    while i < len(lines):
        line = lines[i]
        if line.startswith("+++ "):
            path = line[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            current = path
            files.setdefault(current, [])
            i += 1
            continue
        m = _HUNK_RE.match(line)
        if m and current is not None:
            saw_hunk = True
            start = int(m.group(1))
            hunk: list[str] = []
            i += 1
            while i < len(lines) and lines[i] and lines[i][0] in " +-\\" \
                    and not lines[i].startswith("--- "):
                hunk.append(lines[i])
                i += 1
            files[current].append((start, hunk))
            continue
        i += 1
    if not saw_hunk:
        raise PatchError("diff contains no @@ hunks")
    return files


def _apply_hunks(original: list[str], hunks: list[tuple[int, list[str]]]) -> list[str]:
    out = list(original)
    offset = 0
    for start, hunk in hunks:
        old_block = [l[1:] for l in hunk if l[0] in " -"]
        new_block = [l[1:] for l in hunk if l[0] in " +"]
        # Expected position (1-based start), then search nearby for the
        # context block so small line drift does not break application.
        pos = start - 1 + offset
        found = None
        for cand in range(max(0, pos - 3), min(pos + 4, len(out) - len(old_block) + 1) or 1):
            if out[cand:cand + len(old_block)] == old_block:
                found = cand
                break
        if found is None:
            raise PatchError("hunk context does not match file contents")
        out[found:found + len(old_block)] = new_block
        offset += len(new_block) - len(old_block)
    return out


# --------------------------------------------------------------------------
# Call validation (structural; used by reward scoring)
# --------------------------------------------------------------------------

def validate_call(call) -> tuple[bool, str]:
    """Return (is_well_formed, reason). Structural checks only."""
    if not isinstance(call, dict):
        return False, "call is not a JSON object"
    tool = call.get("tool")
    if tool not in TOOL_SPECS:
        return False, f"unknown tool: {tool!r}"
    args = call.get("arguments")
    if not isinstance(args, dict):
        return False, "arguments is not an object"
    for key, typ in TOOL_SPECS[tool].items():
        if key not in args:
            return False, f"missing argument: {key}"
        if not isinstance(args[key], typ) or (typ is str and not args[key].strip()):
            return False, f"argument {key} has wrong type or is empty"
    if tool == "apply_patch":
        try:
            parse_unified_diff(args["diff"])
        except PatchError as exc:
            return False, f"unparseable diff: {exc}"
    return True, "ok"


# --------------------------------------------------------------------------
# Executor
# --------------------------------------------------------------------------

class ToolExecutor:
    """Executes tool calls against one sandboxed toy-repo checkout."""

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()

    # -- helpers -----------------------------------------------------------
    def _clear_pycache(self) -> None:
        """Drop stale bytecode caches.

        Planted-bug fixes are same-length single-line edits, so a patched
        file can share size+mtime with its buggy version and Python would
        happily run the stale .pyc. Always clear before executing code.
        """
        for cache in self.root.rglob("__pycache__"):
            for f in cache.iterdir():
                f.unlink(missing_ok=True)
            cache.rmdir()

    def _safe_path(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if self.root not in p.parents and p != self.root:
            raise PatchError(f"path escapes sandbox: {rel}")
        return p

    # -- tools -------------------------------------------------------------
    def read_file(self, path: str) -> dict:
        p = self._safe_path(path)
        if not p.is_file():
            return {"ok": False, "malformed": False, "error": f"no such file: {path}"}
        text = p.read_text(encoding="utf-8")
        return {"ok": True, "content": text[:4000], "truncated": len(text) > 4000}

    def search_code(self, query: str) -> dict:
        matches = []
        for f in sorted(self.root.rglob("*.py")):
            for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if query in line:
                    matches.append(f"{f.relative_to(self.root)}:{n}: {line.strip()}")
                    if len(matches) >= 20:
                        return {"ok": True, "matches": matches}
        return {"ok": True, "matches": matches}

    def run_tests(self, timeout: int = 60) -> dict:
        self._clear_pycache()
        proc = subprocess.run(
            [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests"],
            cwd=self.root, capture_output=True, text=True, timeout=timeout,
        )
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        summary = tail[-1] if tail else ""
        ran = next((l for l in reversed(tail) if l.startswith("Ran ")), "")
        return {
            "ok": proc.returncode == 0,
            "malformed": False,
            "passed": proc.returncode == 0,
            "summary": summary,
            "ran": ran,
            "output_tail": "\n".join(tail[-8:]),
        }

    def apply_patch(self, diff: str) -> dict:
        try:
            parsed = parse_unified_diff(diff)
            for rel, hunks in parsed.items():
                p = self._safe_path(rel)
                if not p.is_file():
                    return {"ok": False, "malformed": False,
                            "error": f"patch targets missing file: {rel}"}
                original = p.read_text(encoding="utf-8").splitlines()
                updated = _apply_hunks(original, hunks)
                p.write_text("\n".join(updated) + "\n", encoding="utf-8")
            self._clear_pycache()
            return {"ok": True, "files_changed": sorted(parsed)}
        except PatchError as exc:
            return {"ok": False, "malformed": False, "error": str(exc)}

    # -- dispatch ----------------------------------------------------------
    def dispatch(self, call) -> dict:
        ok, reason = validate_call(call)
        if not ok:
            return {"ok": False, "malformed": True, "error": reason}
        tool, args = call["tool"], call["arguments"]
        try:
            if tool == "read_file":
                return self.read_file(args["path"])
            if tool == "search_code":
                return self.search_code(args["query"])
            if tool == "run_tests":
                return self.run_tests()
            if tool == "apply_patch":
                return self.apply_patch(args["diff"])
        except PatchError as exc:
            return {"ok": False, "malformed": False, "error": str(exc)}
        return {"ok": False, "malformed": True, "error": f"unhandled tool {tool}"}  # pragma: no cover
