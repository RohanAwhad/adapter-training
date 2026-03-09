from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adapter_training.adapter_format import (
    apply_adapter_output_to_draft,
    build_adapter_draft_payload,
    build_adapter_messages,
)

TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", flags=re.DOTALL
)
ROLE_MAP = {
    "system": "system",
    "human": "user",
    "gpt": "assistant",
    "tool": "tool",
}


@dataclass(frozen=True)
class CanonicalExample:
    source_dataset: str
    source_config: str
    source_row_id: str
    source_turn_index: int
    history_messages: list[dict[str, str]]
    request_options: dict[str, Any]
    gold_content: str
    gold_tool_calls: list[dict[str, Any]] | None


@dataclass(frozen=True)
class DraftVariant:
    variant: str
    corruption_types: tuple[str, ...]
    draft_content: str
    draft_tool_calls: list[dict[str, Any]] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build adapter-style training data from Hermes raw snapshots"
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Input directory for downloaded Hermes JSONL files",
    )
    parser.add_argument(
        "--canonical-path", default="data/canonical/adapter_examples.jsonl"
    )
    parser.add_argument("--generated-dir", default="data/generated")
    parser.add_argument(
        "--max-canonical", type=int, default=0, help="Optional cap for debugging"
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped == "":
                continue
            rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def _normalize_arguments(arguments_value: Any) -> str:
    if isinstance(arguments_value, str):
        parsed_arguments = json.loads(arguments_value)
    else:
        parsed_arguments = arguments_value
    if not isinstance(parsed_arguments, dict):
        raise ValueError("tool call arguments must be an object")
    return json.dumps(parsed_arguments, sort_keys=True, separators=(",", ":"))


def normalize_tool_calls(
    tool_calls: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    if tool_calls is None or len(tool_calls) == 0:
        return None

    normalized: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        call_copy = deepcopy(tool_call)
        function = call_copy.get("function")
        if not isinstance(function, dict):
            raise ValueError("tool call missing function payload")
        if "name" not in function:
            raise ValueError("tool call function missing name")
        arguments_value = function.get("arguments", {})
        function["arguments"] = _normalize_arguments(arguments_value)
        normalized.append(call_copy)
    return normalized


def tool_calls_equal(
    left: list[dict[str, Any]] | None, right: list[dict[str, Any]] | None
) -> bool:
    normalized_left = normalize_tool_calls(left)
    normalized_right = normalize_tool_calls(right)
    return json.dumps(normalized_left, sort_keys=True) == json.dumps(
        normalized_right, sort_keys=True
    )


def parse_request_options(row: dict[str, Any]) -> dict[str, Any]:
    tools_raw = row.get("tools")
    tools: list[dict[str, Any]] = []
    if isinstance(tools_raw, str) and tools_raw.strip() != "":
        parsed_tools = json.loads(tools_raw)
        if isinstance(parsed_tools, list):
            tools = parsed_tools
    elif isinstance(tools_raw, list):
        tools = tools_raw

    if len(tools) == 0:
        return {}
    return {"tools": tools, "tool_choice": "auto"}


def _coerce_message(turn: dict[str, Any]) -> dict[str, str] | None:
    from_role = turn.get("from")
    value = turn.get("value")
    if not isinstance(from_role, str) or not isinstance(value, str):
        return None

    role = ROLE_MAP.get(from_role)
    if role is None:
        return None
    return {"role": role, "content": value}


def parse_assistant_turn(
    raw_text: str, source_stub: str
) -> tuple[str, list[dict[str, Any]] | None]:
    blocks = TOOL_CALL_BLOCK_RE.findall(raw_text)
    if len(blocks) == 0:
        return raw_text.strip(), None

    tool_calls: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        payload = json.loads(block)
        if not isinstance(payload, dict):
            raise ValueError("tool call payload must be an object")

        name_value = payload.get("name")
        if not isinstance(name_value, str) or name_value == "":
            raise ValueError("tool call payload missing name")

        arguments_value = payload.get("arguments", {})
        arguments_string = _normalize_arguments(arguments_value)
        tool_calls.append(
            {
                "id": f"{source_stub}_call_{index}",
                "type": "function",
                "function": {
                    "name": name_value,
                    "arguments": arguments_string,
                },
            }
        )

    remaining_content = TOOL_CALL_BLOCK_RE.sub("", raw_text).strip()
    normalized_calls = normalize_tool_calls(tool_calls)
    return remaining_content, normalized_calls


def canonicalize_rows(
    rows: list[dict[str, Any]], max_canonical: int
) -> list[CanonicalExample]:
    canonical_examples: list[CanonicalExample] = []
    for row in rows:
        source_dataset = str(
            row.get("source_dataset", "NousResearch/hermes-function-calling-v1")
        )
        source_config = str(row.get("source_config", "unknown"))
        source_row_id = str(
            row.get("id") or f"{source_config}:{row.get('source_row_idx', 'na')}"
        )
        request_options = parse_request_options(row)

        conversation = row.get("conversations")
        if not isinstance(conversation, list):
            continue

        for turn_index, turn in enumerate(conversation):
            if not isinstance(turn, dict) or turn.get("from") != "gpt":
                continue

            assistant_text = turn.get("value")
            if not isinstance(assistant_text, str):
                continue

            history_messages: list[dict[str, str]] = []
            for history_turn in conversation[:turn_index]:
                if not isinstance(history_turn, dict):
                    continue
                message = _coerce_message(history_turn)
                if message is not None:
                    history_messages.append(message)

            source_stub = f"{source_config}_{source_row_id}_{turn_index}"
            gold_content, gold_tool_calls = parse_assistant_turn(
                assistant_text, source_stub
            )
            if gold_content == "" and gold_tool_calls is None:
                continue

            canonical_examples.append(
                CanonicalExample(
                    source_dataset=source_dataset,
                    source_config=source_config,
                    source_row_id=source_row_id,
                    source_turn_index=turn_index,
                    history_messages=history_messages,
                    request_options=request_options,
                    gold_content=gold_content,
                    gold_tool_calls=gold_tool_calls,
                )
            )

            if max_canonical > 0 and len(canonical_examples) >= max_canonical:
                return canonical_examples

    return canonical_examples


def _first_argument(arguments: dict[str, Any]) -> tuple[str, Any] | None:
    for key, value in arguments.items():
        return key, value
    return None


def _changed_argument_value(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value)
    if isinstance(value, str):
        if value == "":
            return 7
        return 7
    if isinstance(value, list):
        return {"changed": True}
    if isinstance(value, dict):
        return ["changed"]
    return "changed"


def _tool_calls_with_changed_name(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    updated = deepcopy(tool_calls)
    name_value = updated[0]["function"]["name"]
    updated[0]["function"]["name"] = f"{name_value}_wrong"
    return normalize_tool_calls(updated) or []


def _tool_calls_with_missing_first_arg(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    updated = deepcopy(tool_calls)
    arguments = json.loads(updated[0]["function"]["arguments"])
    first_entry = _first_argument(arguments)
    if first_entry is None:
        return None
    arguments.pop(first_entry[0])
    updated[0]["function"]["arguments"] = json.dumps(
        arguments, sort_keys=True, separators=(",", ":")
    )
    return normalize_tool_calls(updated)


def _tool_calls_with_renamed_arg(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    updated = deepcopy(tool_calls)
    arguments = json.loads(updated[0]["function"]["arguments"])
    first_entry = _first_argument(arguments)
    if first_entry is None:
        return None
    old_key, value = first_entry
    arguments.pop(old_key)
    arguments[f"{old_key}_wrong"] = value
    updated[0]["function"]["arguments"] = json.dumps(
        arguments, sort_keys=True, separators=(",", ":")
    )
    return normalize_tool_calls(updated)


def _tool_calls_with_wrong_arg_type(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    updated = deepcopy(tool_calls)
    arguments = json.loads(updated[0]["function"]["arguments"])
    first_entry = _first_argument(arguments)
    if first_entry is None:
        return None
    key, value = first_entry
    arguments[key] = _changed_argument_value(value)
    updated[0]["function"]["arguments"] = json.dumps(
        arguments, sort_keys=True, separators=(",", ":")
    )
    return normalize_tool_calls(updated)


def _tool_calls_with_extra_call(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    updated = deepcopy(tool_calls)
    extra = deepcopy(updated[0])
    extra["id"] = f"{extra.get('id', 'call')}_extra"
    updated.append(extra)
    return normalize_tool_calls(updated) or []


def _tool_calls_with_missing_call(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    if len(tool_calls) == 1:
        return None
    updated = deepcopy(tool_calls[:-1])
    return normalize_tool_calls(updated)


def build_variants(example: CanonicalExample) -> list[DraftVariant]:
    variants: list[DraftVariant] = [
        DraftVariant(
            variant="clean_lgtm",
            corruption_types=("none",),
            draft_content=example.gold_content,
            draft_tool_calls=example.gold_tool_calls,
        )
    ]

    if example.gold_tool_calls is not None:
        tool_calls = example.gold_tool_calls

        variants.append(
            DraftVariant(
                variant="wrong_function_name",
                corruption_types=("wrong_function_name",),
                draft_content=example.gold_content,
                draft_tool_calls=_tool_calls_with_changed_name(tool_calls),
            )
        )

        missing_first_arg = _tool_calls_with_missing_first_arg(tool_calls)
        if missing_first_arg is not None:
            variants.append(
                DraftVariant(
                    variant="missing_required_arg",
                    corruption_types=("missing_required_arg",),
                    draft_content=example.gold_content,
                    draft_tool_calls=missing_first_arg,
                )
            )

        renamed_arg = _tool_calls_with_renamed_arg(tool_calls)
        if renamed_arg is not None:
            variants.append(
                DraftVariant(
                    variant="wrong_arg_key",
                    corruption_types=("wrong_arg_key",),
                    draft_content=example.gold_content,
                    draft_tool_calls=renamed_arg,
                )
            )

        wrong_arg_type = _tool_calls_with_wrong_arg_type(tool_calls)
        if wrong_arg_type is not None:
            variants.append(
                DraftVariant(
                    variant="wrong_arg_type",
                    corruption_types=("wrong_arg_type",),
                    draft_content=example.gold_content,
                    draft_tool_calls=wrong_arg_type,
                )
            )

        variants.append(
            DraftVariant(
                variant="extra_tool_call",
                corruption_types=("extra_tool_call",),
                draft_content=example.gold_content,
                draft_tool_calls=_tool_calls_with_extra_call(tool_calls),
            )
        )

        missing_call = _tool_calls_with_missing_call(tool_calls)
        if missing_call is not None:
            variants.append(
                DraftVariant(
                    variant="missing_tool_call",
                    corruption_types=("missing_tool_call",),
                    draft_content=example.gold_content,
                    draft_tool_calls=missing_call,
                )
            )
    else:
        variants.extend(
            [
                DraftVariant(
                    variant="markdown_fence",
                    corruption_types=("markdown_fence",),
                    draft_content=f"```\n{example.gold_content}\n```",
                    draft_tool_calls=None,
                ),
                DraftVariant(
                    variant="prefix_chatter",
                    corruption_types=("prefix_chatter",),
                    draft_content=f"Sure! {example.gold_content}",
                    draft_tool_calls=None,
                ),
            ]
        )

    deduped: list[DraftVariant] = []
    seen: set[str] = set()
    for variant in variants:
        key = json.dumps(
            {
                "content": variant.draft_content,
                "tool_calls": normalize_tool_calls(variant.draft_tool_calls),
            },
            sort_keys=True,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(variant)

    return deduped


def build_teacher_output(
    draft_content: str,
    draft_tool_calls: list[dict[str, Any]] | None,
    gold_content: str,
    gold_tool_calls: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    patches: list[dict[str, Any]] = []
    if draft_content != gold_content:
        patches.append({"op": "replace", "path": "/content", "value": gold_content})
    if not tool_calls_equal(draft_tool_calls, gold_tool_calls):
        patches.append(
            {"op": "replace", "path": "/tool_calls", "value": gold_tool_calls}
        )

    if len(patches) == 0:
        return {"decision": "lgtm"}
    return {"decision": "patch", "patches": patches}


def split_name(source_key: str) -> str:
    bucket = int(hashlib.sha256(source_key.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def canonical_to_record(example: CanonicalExample) -> dict[str, Any]:
    return {
        "source_dataset": example.source_dataset,
        "source_config": example.source_config,
        "source_row_id": example.source_row_id,
        "source_turn_index": example.source_turn_index,
        "history_messages": example.history_messages,
        "request_options": example.request_options,
        "gold": {
            "content": example.gold_content,
            "tool_calls": example.gold_tool_calls,
        },
    }


def build_training_records(
    canonical_examples: list[CanonicalExample],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []

    stats: dict[str, Any] = {
        "canonical_examples": len(canonical_examples),
        "generated_rows": 0,
        "decision_counts": {"lgtm": 0, "patch": 0},
        "tool_rows": 0,
        "split_counts": {"train": 0, "val": 0, "test": 0},
    }

    for example in canonical_examples:
        has_tools = (
            isinstance(example.request_options.get("tools"), list)
            and len(example.request_options["tools"]) > 0
        )
        source_key = f"{example.source_config}:{example.source_row_id}"
        split = split_name(source_key)

        for variant in build_variants(example):
            draft_payload = build_adapter_draft_payload(
                content=variant.draft_content,
                tool_calls=variant.draft_tool_calls,
            )
            prompt_messages = build_adapter_messages(
                messages=example.history_messages,
                draft=draft_payload,
                request_options=example.request_options,
            )
            teacher_output = build_teacher_output(
                draft_content=variant.draft_content,
                draft_tool_calls=variant.draft_tool_calls,
                gold_content=example.gold_content,
                gold_tool_calls=example.gold_tool_calls,
            )
            assistant_content = json.dumps(
                teacher_output, sort_keys=True, separators=(",", ":")
            )

            final_content, final_tool_calls = apply_adapter_output_to_draft(
                content=variant.draft_content,
                tool_calls=variant.draft_tool_calls,
                adapter_output=assistant_content,
            )
            if final_content != example.gold_content or not tool_calls_equal(
                final_tool_calls, example.gold_tool_calls
            ):
                raise ValueError("teacher label failed reconstruction check")

            decision = teacher_output["decision"]
            if decision not in ("lgtm", "patch"):
                raise ValueError("unexpected decision value")

            record = {
                "messages": [
                    *prompt_messages,
                    {
                        "role": "assistant",
                        "content": assistant_content,
                    },
                ],
                "metadata": {
                    "split": split,
                    "source_dataset": example.source_dataset,
                    "source_config": example.source_config,
                    "source_row_id": example.source_row_id,
                    "source_turn_index": example.source_turn_index,
                    "sample_variant": variant.variant,
                    "corruption_types": list(variant.corruption_types),
                    "has_tools": has_tools,
                    "decision": decision,
                    "draft": {
                        "content": variant.draft_content,
                        "tool_calls": variant.draft_tool_calls,
                    },
                    "gold": {
                        "content": example.gold_content,
                        "tool_calls": example.gold_tool_calls,
                    },
                },
            }
            records.append(record)
            stats["generated_rows"] += 1
            stats["decision_counts"][decision] += 1
            if has_tools:
                stats["tool_rows"] += 1
            stats["split_counts"][split] += 1

    return records, stats


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    canonical_path = Path(args.canonical_path)
    generated_dir = Path(args.generated_dir)

    raw_files = sorted(raw_dir.glob("hermes_*.jsonl"))
    if len(raw_files) == 0:
        raise ValueError(f"no Hermes raw files found under {raw_dir}")

    raw_rows: list[dict[str, Any]] = []
    for raw_file in raw_files:
        raw_rows.extend(read_jsonl(raw_file))

    canonical_examples = canonicalize_rows(raw_rows, max_canonical=args.max_canonical)
    canonical_records = [canonical_to_record(example) for example in canonical_examples]
    write_jsonl(canonical_path, canonical_records)
    print(
        f"wrote canonical examples -> {canonical_path} ({len(canonical_records)} rows)"
    )

    training_records, stats = build_training_records(canonical_examples)
    split_buckets: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for record in training_records:
        split = record["metadata"]["split"]
        split_buckets[split].append(record)

    for split, rows in split_buckets.items():
        split_path = generated_dir / f"adapter_{split}.jsonl"
        write_jsonl(split_path, rows)
        print(f"wrote {split} -> {split_path} ({len(rows)} rows)")

    stats_path = generated_dir / "adapter_train_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote stats -> {stats_path}")


if __name__ == "__main__":
    main()
