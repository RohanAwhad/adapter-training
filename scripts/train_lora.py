from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LoRA SFT for adapter data using training_hub"
    )
    parser.add_argument(
        "--training-hub-src", default="/Users/rawhad/1_Projects/training_hub/src"
    )
    parser.add_argument("--data-path", default="data/generated/adapter_train.jsonl")
    parser.add_argument("--output-dir", default="outputs/hermes_adapter_lora_v1")
    parser.add_argument("--model-path", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument(
        "--dataset-type",
        default="chat_template",
        choices=["chat_template", "alpaca", "passthrough"],
    )
    parser.add_argument("--field-messages", default="messages")

    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--effective-batch-size", type=int, default=32)

    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)

    parser.add_argument("--qlora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save-model", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--assistant-only-loss", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    training_hub_src = Path(args.training_hub_src)
    if not training_hub_src.exists():
        raise ValueError(f"training_hub src path not found: {training_hub_src}")
    if str(training_hub_src) not in sys.path:
        sys.path.insert(0, str(training_hub_src))

    if not Path(args.data_path).exists():
        raise ValueError(f"training data not found: {args.data_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb_project = args.wandb_project or None
    wandb_entity = args.wandb_entity or None
    wandb_run_name = args.wandb_run_name or None

    train_kwargs = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "ckpt_output_dir": str(output_dir),
        "dataset_type": args.dataset_type,
        "field_messages": args.field_messages,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "max_seq_len": args.max_seq_len,
        "micro_batch_size": args.micro_batch_size,
        "effective_batch_size": args.effective_batch_size,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "load_in_4bit": bool(args.qlora),
        "bf16": True,
        "sample_packing": True,
        "save_model": bool(args.save_model),
        "assistant_only_loss": bool(args.assistant_only_loss),
        "logging_steps": 10,
        "save_steps": 200,
        "save_total_limit": 2,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "wandb_run_name": wandb_run_name,
    }

    print("Training config")
    for key, value in train_kwargs.items():
        print(f"- {key}: {value}")

    if args.dry_run:
        return

    training_hub = importlib.import_module("training_hub")
    lora_sft = getattr(training_hub, "lora_sft")

    lora_module = importlib.import_module("training_hub.algorithms.lora")
    original_build_training_args = getattr(
        lora_module.UnslothLoRABackend, "_build_training_args"
    )

    def _patched_build_training_args(self, params):
        config = original_build_training_args(self, params)
        config.assistant_only_loss = bool(params.get("assistant_only_loss", False))
        return config

    lora_module.UnslothLoRABackend._build_training_args = _patched_build_training_args

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print(f"Using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    lora_sft(**train_kwargs)
    print("training run completed")


if __name__ == "__main__":
    main()
