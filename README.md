# QAMR-CRS

Official implementation of **QAMR-CRS: Query-Aware Multimodal Routing for
Conversational Recommendation**.

QAMR-CRS routes knowledge-graph, co-occurrence, text, and image evidence
according to the dialogue context and mentioned entities. The routed evidence
is converted into prompt prefixes for item recommendation.

## Setup

```bash
conda env create -f environment.yml
conda activate qamr-crs
export PYTHONNOUSERSITE=1
```

The experiments use local snapshots of DialoGPT-small and RoBERTa-base:

```text
hf_models/DialoGPT-small/
hf_models/roberta-base/
```

Place the processed datasets in:

```text
rec_data/redial/
rec_data/inspired/
```

Dataset files, pretrained models, checkpoints, and logs are not included in
the repository.

## Recommendation

Pretrain the ReDial prompt encoder:

```bash
scripts/run_pretrain_redial.sh
```

Train and evaluate QAMR-CRS:

```bash
scripts/run_rec_redial.sh
scripts/run_rec_inspired.sh
```

Each training run selects the checkpoint with the lowest validation loss and
evaluates the test split once from that checkpoint.

## Reproducibility

Record the environment and command:

```bash
scripts/collect_run_info.sh experiments/run_info.txt \
  scripts/run_rec_redial.sh --fp16 --num_workers 4
```

Parse recommendation metrics:

```bash
scripts/parse_rec_results.py "log/*.log"
```

Utilities for matched-seed summaries, ablations, and routing analysis are
available under `scripts/` and `tools/`.

## Acknowledgements

This implementation builds on MSCRS and related conversational recommendation
research. Please cite the corresponding papers and repositories when using
this code.
