# Experiment Records

Keep compact run metadata and parsed summaries in this directory. Raw logs,
datasets, model snapshots, and checkpoints should remain outside git.

Each reported run should record:

- commit hash and command line
- package versions and GPU model
- dataset split and preprocessing source
- output and raw log paths
- checkpoint selection rule
- parsed metrics

Select checkpoints using validation data only. Report test metrics once from
the selected checkpoint.
