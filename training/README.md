# Training data

Released data:

- [HyWe-Hybrid-SFT-Data](https://huggingface.co/datasets/yzxjb/HyWe-Hybrid-SFT-Data)
  contains 3,735 trajectories rendered as 8,055 segments, with 45,045 original
  screenshots in tar shards. Extract `images_shards/*.tar` at the dataset root
  before resolving `images/...` references.
- [HyWe-Search-SFT-Data](https://huggingface.co/datasets/yzxjb/HyWe-Search-SFT-Data)
  contains the 128K multi-turn conversion of XYZ-Aquila-SFT trajectories.
  The current source snapshot has **10,012 JSONL samples**. Download
  `train_aquila_v2_multiturn_128k.jsonl`.

These mirror the ModelScope releases `yzxjb11/HybridWebTraj-v1` and
`yzxjb11/aquila-sft-v2-react`. Original source cards are retained in the HF
repositories. The raw training files and screenshot archives are unchanged.
The original search data card describes 7,000 input trajectories and 10,413
rendered samples; the current ModelScope file contains 10,012 rows. The mirror
preserves the file exactly and records the observed count separately rather
than modifying it to match the older card.

Download the released files and extract the screenshot archives:

```bash
hf download yzxjb/HyWe-Hybrid-SFT-Data --repo-type dataset --local-dir data/hybrid
hf download yzxjb/HyWe-Search-SFT-Data --repo-type dataset --local-dir data/search
for shard in data/hybrid/images_shards/*.tar; do tar -xf "$shard" -C data/hybrid; done
```

The hybrid samples are in `data/hybrid/data/train.jsonl`; image paths resolve
relative to `data/hybrid`. The search samples are in
`data/search/train_aquila_v2_multiturn_128k.jsonl`.

## Export newly collected trajectories

```bash
source configs/collector.env
TAGS="demo" bash training/build_sft_data.sh
```

Input is one or more `runs/train_<TAG>.opure.jsonl` files produced by collector
Stage 4. The export launcher:

1. Combines batches and renders screenshots for the O-side coordinate/spin judges.
2. Masks coordinate errors and O-side repeated actions; S-side masks use
   mechanical duplicate-query/URL checks. `answer` and `switch` remain supervised.
3. Aligns system prompts with the current inference prompt.
4. Writes `runs/sft/hybrid.jsonl`, copies referenced screenshots under
   `runs/sft/images/`, and writes `hybrid_images.jsonl` with source provenance.

Each sample has `messages`, trajectory/segment metadata, and per-assistant
`loss` values. Context messages are conditioning; targets are the segment's
assistant turns. The same `react_v2` renderer/folding logic is used at inference.
The optional `AQUILA_SRCS` variable converts original XYZ-Aquila-SFT files into
`search_aquila.jsonl`. Released search data is already converted.

## Saved training configuration

`training_config.json` records the portable values audited from the released
checkpoint's saved arguments. The paper used full LM fine-tuning with the
vision encoder/projector frozen, BF16, one configured epoch, sequence length
131,072, effective batch size 32, learning rate 2e-5 (minimum 2e-6), 5% warmup,
cosine decay, Adam betas 0.9/0.95, weight decay 0.1, and seed 42.

The training stack was Megatron via ms-swift 4.5.0.dev0 with tensor/expert
parallelism 8. The exact GPU model, elapsed training time, and source sampling
proportions were not recorded in the saved arguments. The saved iteration is
564; `save_steps=565` is an interval, not an audited count of completed updates.
This release provides model weights, data, and preprocessing/export code; it
does not claim that rerunning a generic SFT command reproduces the checkpoint
bit for bit.
