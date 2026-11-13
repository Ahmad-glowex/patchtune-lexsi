"""PatchTune-mini: post-train Tiny Aya Global for reliable tool calling.

Track 3 (AgentTune + AuditKIT), Problem Statements #1 (reliable tool
calling) and #2 (efficient trajectories). Lexsi AI Safety & Alignment
Hackathon, Nov 2026.

Submodules: ``tools`` (sandboxed tool executor), ``tasks`` (synthetic
dataset generation), ``reward`` (verifier / GRPO reward functions),
``evaluate`` (base-vs-tuned metrics harness).
"""

__version__ = "0.1.0"
