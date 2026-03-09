# Hermes Adapter Training Plan (v15)

## Goal

Train a small interventionist model that behaves like the adapter stage in `adapter_critic`:

- input: conversation history + tool contract + API draft payload
- output: strict adapter JSON (`{"decision":"lgtm"}` or `{"decision":"patch", ...}`)
- success criterion: patches are valid, apply cleanly, and produce the intended final content/tool calls

## Scope

- base data source: `NousResearch/hermes-function-calling-v1`
- configs for v1: `func_calling_singleturn`, `func_calling`
- training method: LoRA SFT using `training_hub` (Unsloth backend)
- evaluation: synthetic holdout scorer + `experiments.v15.benchmark_adapter_patches`

## Output Artifacts

Create these artifacts (paths are suggestions):

- `data/raw/hermes_singleturn.jsonl`
- `data/raw/hermes_multiturn.jsonl`
- `data/canonical/adapter_examples.jsonl`
- `data/generated/adapter_train.jsonl`
- `data/generated/adapter_val.jsonl`
- `data/generated/adapter_test.jsonl`
- `data/generated/adapter_train_stats.json`
- `results/baseline_benchmark.json`
- `results/posttrain_benchmark.json`
- `results/holdout_eval.json`

## Phase 0 - Environment Setup

1. Define stable roots:
   - `PROJECT_DIR=/Users/rawhad/1_Projects/api_adapter_v2026/adapter_critic`
   - `TRAINING_HUB_DIR=/Users/rawhad/1_Projects/training_hub`
2. Confirm Python env has `datasets`, `orjson`/`json`, `pydantic`.
3. Confirm GPU runtime for training (CUDA, bf16 support).
4. Confirm baseline benchmark script runs:
   - `uv run python -m experiments.v15.benchmark_adapter_patches --dry-run`

## Phase 1 - Download Hermes Data

1. Pull both Hermes configs via `datasets`:
   - `func_calling_singleturn`
   - `func_calling`
2. Save immutable raw snapshots to `data/raw/`.
3. Record metadata in `adapter_train_stats.json`:
   - source dataset id + commit hash if available
   - config names
   - row counts
   - download timestamp

## Phase 2 - Build Canonical Gold Examples

For each candidate row:

1. Extract tool contract:
   - parse `<tools>...</tools>` from system prompt
   - map to OpenAI-style `request_options.tools`
2. Extract conversation history:
   - convert Hermes `from` roles to adapter roles (`system/user/assistant/tool`)
3. Identify gold assistant target turn:
   - tool-call turn: normalize to `tool_calls` list (OpenAI shape)
   - plain answer turn: normalize to `content` string, `tool_calls=None`
4. Build canonical example object:
   - `messages`, `request_options`, `gold_content`, `gold_tool_calls`, source ids
5. Drop row if critical parsing fails (track failure reason counts).

Quality gate:

- >=95% canonicalization success rate on selected configs.

## Phase 3 - Generate Adapter Training Pairs

For each canonical example, create multiple drafts:

1. Positive (`lgtm`) samples:
   - draft == gold
2. Negative (`patch`) samples with targeted corruptions:
   - wrong function name
   - missing required argument
   - wrong argument key
   - wrong argument type/enum
   - extra irrelevant tool call
   - missing required tool call
   - duplicate tool call
   - text formatting violations (prefix/fence/exactness)
3. Build deterministic teacher label:
   - generate minimal `replace` patches from draft -> gold
   - keep patch paths aligned with adapter apply logic
4. Materialize training row:
   - `messages`: system + adapter user payload + assistant target JSON
   - include metadata fields for audit/debug

Target class mix:

- 60% tool-call patch
- 20% text patch
- 20% lgtm

## Phase 4 - Dataset Validation (Pre-Training)

Run validators on generated train/val/test files:

1. JSON + schema validation of assistant target.
2. For every `patch` example:
   - run `apply_adapter_output_to_draft`
   - assert apply succeeds
   - assert final output equals gold target
   - assert patch is not noop
3. Split integrity:
   - split by source conversation id (no leakage)
4. Distribution checks:
   - corruption type counts
   - tool vs non-tool counts
   - average prompt length / token estimate

Hard gate before training:

- 100% patch apply success on dataset labels.

## Phase 5 - Baseline Evaluation (Before Fine-Tuning)

1. Run existing benchmark on unfine-tuned model.
2. Save JSON result to `results/baseline_benchmark.json`.
3. Save core metrics table:
   - valid_json_rate
   - patch_decision_rate
   - apply_success_rate
   - qualified_success_rate
   - tool_qualified_success_rate

## Phase 6 - Training (training_hub LoRA SFT)

Preferred first run: QLoRA on 0.8B model, then repeat on 2B.

Suggested initial config:

- model: `Qwen/Qwen3.5-0.8B-Instruct` (replace with exact available id)
- algorithm: `lora_sft`
- dataset: messages/chat-template format
- `dataset_type=chat_template`
- `field_messages=messages`
- `load_in_4bit=True`
- `lora_r=32`, `lora_alpha=64`, `lora_dropout=0.0`
- `learning_rate=1e-4`
- `num_epochs=3`
- `max_seq_len=4096`
- `effective_batch_size=32` (or nearest stable)

Training command style (via training_hub example script):

```bash
cd "$TRAINING_HUB_DIR"
uv run python examples/scripts/lora_example.py \
  --data-path "$PROJECT_DIR/data/generated/adapter_train.jsonl" \
  --ckpt-output-dir "$PROJECT_DIR/outputs/hermes_adapter_lora_v1" \
  --model-path Qwen/Qwen3.5-0.8B-Instruct \
  --dataset-type chat_template \
  --field-messages messages \
  --qlora \
  --lora-r 32 \
  --lora-alpha 64 \
  --learning-rate 1e-4 \
  --num-epochs 3 \
  --max-seq-len 4096 \
  --micro-batch-size 2 \
  --effective-batch-size 32
```

If multi-GPU data parallel is used, launch with `torchrun` per training_hub docs.

## Phase 7 - Post-Training Validation

1. Serve adapter model checkpoint (or merged model) on local endpoint.
2. Re-run benchmark:
   - `uv run python -m experiments.v15.benchmark_adapter_patches ... --output-json results/posttrain_benchmark.json`
3. Run held-out synthetic eval from `adapter_test.jsonl` and save `results/holdout_eval.json`.
4. Compare baseline vs post-train deltas.

Pass/fail thresholds for v1:

- valid_json_rate >= 99%
- apply_success_rate >= 95%
- qualified_success_rate >= 60%
- tool_qualified_success_rate >= 50%

## Phase 8 - Iteration Loop

If gates fail:

1. Bucket failures (decision errors, unsupported paths, arg-shape errors, noop patches).
2. Generate hard-negative augmentation focused on top failure buckets.
3. Continue training from best checkpoint for 1-2 short epochs.
4. Re-evaluate and record metric movement.

## Execution Checklist

- [ ] Raw Hermes data downloaded and versioned
- [ ] Canonical examples built and validated
- [ ] Synthetic adapter dataset generated (train/val/test)
- [ ] Patch-apply quality gates all green
- [ ] Baseline benchmark captured
- [ ] LoRA training run completed
- [ ] Post-train benchmark + holdout eval captured
- [ ] Go/no-go decided against thresholds
