# Negishi cluster build

This package splits ELFNet graph extraction into independent Slurm array tasks,
then merges the task outputs and creates the train/validation/test splits once
across the whole dataset. The same parent phase therefore cannot leak across
splits just because samples ran in different tasks.

## Before submitting

The repository already uses Negishi and Slurm. The defaults use account
`andre176`, partition `cpu`, and QOS `normal`; confirm the account with `slist`
and change `negishi.env` if your current allocation differs. Negishi requires
an explicit account, partition, QOS, and time limit for jobs. The requested
resources/time are in the `.slurm` files.

Copy the repository scripts and the ELFNet source tree to a path visible from
Negishi compute nodes. By default the source is expected under the repository
at `elfnet_data/extracted/dft-reference/dft-reference-elfs-50000-v1`. If you
put the data elsewhere, edit `ELFNET_SOURCE_ROOT` in `negishi.env`. The source
tree is about 21 GB, so check `myquota` and use the cluster's recommended
transfer method. Set `ELFNET_RUN_ROOT` to scratch for working shards, and set
`ELFNET_FINAL_OUTPUT` to persistent project storage if available. Negishi
scratch is temporary and purged; do not leave the only copy of the completed
dataset there.

If the repository and data are only on your workstation, transfer the project
folder to Negishi. For example, run this locally after replacing the account
name and remote destination with your actual values:

```bash
LOCAL_REPO="$PWD"
NEGISHI_USER="your-rcac-username"
REMOTE_REPO="/scratch/negishi/$NEGISHI_USER/electride-ml-predictor"
ssh "$NEGISHI_USER@negishi.rcac.purdue.edu" "mkdir -p '$REMOTE_REPO'"
rsync -avh --partial --info=progress2 --exclude='.git/' \
  "$LOCAL_REPO/" \
  "$NEGISHI_USER@negishi.rcac.purdue.edu:$REMOTE_REPO/"
```

If the data already lives on Negishi, transfer only the changed code or clone
the repository there and set `ELFNET_SOURCE_ROOT` to the existing dataset.
On Negishi, check storage and account access with `myquota` and `slist` before
the full run.

## Environment

From the repository on Negishi, set up a private Conda environment with the
cluster's supported `conda` module:

```bash
bash elf_void_analysis/cluster/negishi/setup_environment.sh
```

## Submit in two steps

First submit the manifest job. It indexes POSCAR/ELFCAR pairs once and records
their hashes for provenance by default:

```bash
bash elf_void_analysis/cluster/negishi/submit_prepare.sh
```

Wait until its job finishes successfully (`squeue -u "$USER"` or `jobinfo
<job-id>`). Then submit the extraction array and dependent merge/validation job:

```bash
bash elf_void_analysis/cluster/negishi/submit_array.sh
```

The manifest hashes source files once by default for provenance. If metadata
storage throughput makes this too slow, set `ELFNET_NO_HASHES=1` in
`negishi.env`; this omits source integrity hashes but leaves extraction intact.

Defaults are 250 samples per task and at most 32 concurrent tasks. The array
size is calculated from the prepared manifest, so it adapts to missing or
invalid source pairs. Change `ELFNET_SAMPLES_PER_TASK` or
`ELFNET_ARRAY_CONCURRENCY` in `negishi.env` before submission to tune throughput.
Each task requests one CPU core, 4 GB memory, and one hour. The code is
single-process per task; the array provides the parallelism. Merge and dataset
validation run after the array terminates. If a task crashes, merge fails with
the missing task directory so that incomplete output is not presented as a
complete dataset.

The chunk size balances scheduler overhead with restart cost: roughly 200 jobs
for 50,000 rows at the defaults. A task output is kept intact so a completed
task can be skipped if the submission is repeated; use a new `ELFNET_RUN_ID`
for a clean rebuild. Failed partial task folders are intentionally not deleted
automatically.

## Outputs and monitoring

```text
$ELFNET_RUN_ROOT/
├── manifest.parquet
├── manifest_report.json
├── chunks/task-00000/                 # one independent extraction task
├── logs/                              # Slurm stdout/stderr
└── dataset/                           # merged HDF5 shards, manifest, splits,
                                       # build report, and validation report
```

Monitor with `squeue -u "$USER"`. Inspect logs in `$ELFNET_RUN_ROOT/logs/`.
The final `build_report.json` and `validation.json` record processed/failed
counts and validation results. The final split table is regenerated globally
from all successful rows, keeping each `family_id` together.

## Why this is faster

The old builder ran samples in one Python process. The array assigns independent
groups of 250 samples to concurrent CPU jobs. Based on the local 10-sample
benchmark (~4.65 seconds per sample), 50,000 samples imply ~65 serial CPU-hours.
With 32 array tasks running concurrently, ideal processing time is ~2 hours;
filesystem throughput, queue delay, varying ELFCAR sizes, manifest indexing,
and the merge/validation pass add overhead. A practical estimate is several
hours, subject to the cluster allocation and queue.

## Restarting or resubmitting

Completed task folders are skipped. A folder containing a partial result stops
the corresponding task rather than being overwritten. Inspect and move/delete
only that specific failed task folder before resubmitting, or choose a fresh
`ELFNET_RUN_ID`. The merge step requires a report and manifest from every
expected task and will refuse an incomplete run.
