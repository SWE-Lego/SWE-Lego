#!/usr/bin/env python3
"""Convert raw trajectory outputs into verifier input format.

Input supports:
1) Instance-level JSONL/JSON:
   {"instance_id": "...", "run_1": {...}, "run_2": {...}, ...}
   where each run usually contains `funccalloff_messages`, `patch`, `resolved`.

2) Already flattened run-level records:
   {"instance_id": "...", "run": "run_1", "messages": [...], "score": 1}

Output (JSONL):
- instance_id
- run
- score
- messages: [system, user]
"""

import argparse
import json
import os
from typing import Any, Dict, Iterable, List

SYSTEM_PROMPT = """You are an expert judge evaluating AI assistant interactions. Your task is to determine if the assistant successfully resolved the user's request.

Key evaluation criteria:
1. Did the assistant complete the main task requested by the user?
2. Did the assistant handle all edge cases and requirements specified?
3. Were there any errors or issues in the final solution?
4. Did the assistant verify the solution works as intended?

Respond only with "<judgement>YES</judgement>" or "<judgement>NO</judgement>"."""


def load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []

    if path.endswith(".jsonl"):
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        # Fallback for jsonl-like content without ".jsonl" suffix.
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    return []


def build_interaction_log(messages: List[Dict[str, Any]], patch: str) -> str:
    lines: List[str] = []
    lines.append("Please evaluate the following interaction between an AI assistant and a user:")
    lines.append("")
    lines.append("=== INTERACTION LOG ===")
    lines.append("")

    for msg in messages:
        role = str(msg.get("role", "")).lower()
        content = msg.get("content", "")

        if role == "system":
            lines.append("[SYSTEM]")
            lines.append(str(content))
            lines.append("[/SYSTEM]")
            lines.append("")
        elif role == "user":
            lines.append("[USER]")
            lines.append(str(content))
            lines.append("[/USER]")
            lines.append("")
        elif role == "assistant":
            lines.append("[ASSISTANT]")
            lines.append(str(content))
            lines.append("[/ASSISTANT]")
            lines.append("")

    lines.append("=== END INTERACTION ===")
    lines.append("")
    lines.append("=== FINAL PATCH ===")
    lines.append("")
    lines.append("[PATCH]")
    lines.append(str(patch) if patch else "No patch generated")
    lines.append("[/PATCH]")
    lines.append("")
    lines.append("=== END FINAL PATCH ===")
    lines.append("")
    lines.append(
        "Based on the above interaction, did the assistant successfully resolve the user's initial request? Respond with YES or NO."
    )

    return "\n".join(lines)


def score_from_fields(obj: Dict[str, Any]) -> int:
    if "score" in obj and obj["score"] is not None:
        return int(float(obj["score"]) > 0.5)
    if "ground_truth" in obj and obj["ground_truth"] is not None:
        return int(float(obj["ground_truth"]) > 0.5)
    if "resolved" in obj and obj["resolved"] is not None:
        return 1 if bool(obj["resolved"]) else 0
    return 0


def iter_runs(item: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    instance_id = item.get("instance_id", "unknown")

    run_keys = [k for k in item.keys() if k.startswith("run_") and isinstance(item[k], dict)]
    if run_keys:
        run_keys.sort(key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else x)
        for run_key in run_keys:
            run_data = item[run_key]
            messages = run_data.get("funccalloff_messages") or run_data.get("messages") or []
            if not messages:
                continue
            patch = run_data.get("patch") or run_data.get("model_patch") or ""
            yield {
                "instance_id": instance_id,
                "run": run_key,
                "score": score_from_fields(run_data),
                "messages": messages,
                "patch": patch,
            }
        return

    # Flattened input fallback
    messages = item.get("funccalloff_messages") or item.get("messages") or []
    if messages:
        patch = item.get("patch") or item.get("model_patch") or ""
        yield {
            "instance_id": instance_id,
            "run": item.get("run") or item.get("run_id") or "run_1",
            "score": score_from_fields(item),
            "messages": messages,
            "patch": patch,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert trajectories to verifier input format")
    parser.add_argument("--input", required=True, help="Path to raw trajectories (.jsonl or .json)")
    parser.add_argument("--output", required=True, help="Path to output verifier jsonl")
    args = parser.parse_args()

    records = load_records(args.input)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    out_rows = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for item in records:
            if not isinstance(item, dict):
                continue
            for run in iter_runs(item):
                interaction_log = build_interaction_log(run["messages"], run["patch"])
                out = {
                    "instance_id": run["instance_id"],
                    "run": run["run"],
                    "score": run["score"],
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": interaction_log},
                    ],
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                out_rows += 1

    print(f"Done. Wrote {out_rows} verifier samples to: {args.output}")


if __name__ == "__main__":
    main()
