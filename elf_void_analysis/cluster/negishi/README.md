# Negishi cluster build

This package builds the ELFNet graph dataset on one exclusive 128-core Negishi
CPU node. It launches up to 128 independent workers on that node; each worker
processes one or more 250-sample chunks. A final job merges the chunk outputs
and creates family-safe train/validation/test splits across the full dataset.

## Before submitting

The repository already uses Negishi and Slurm. Set `ELFNET_SLURM_ACCOUNT` in
your shell or in an ignored `negishi.local.env` file before sourcing
`negishi.env`. Confirm the allocation with `slist`. Negishi requires an
explicit account, partition, QOS, and time limit for jobs. The requested
resources/time are in the local `.slurm` files.

The repository can stay in home while the 21 GB ELFNet source tree and all
working outputs stay in scratch. The default source path is
`$RCAC_SCRATCH/dft-reference-elfs-50000-v1`, matching the recommended `scp`
destination. Check `myquota` before transferring. Set `ELFNET_FINAL_OUTPUT` to
persistent project storage if available. Negishi scratch is temporary and
purged; do not leave the only copy of the completed dataset there.

If the source data is only on your workstation, copy just the source tree into
Negishi scratch from your local machine:

```bash
scp -r "/path/to/dft-reference-elfs-50000-v1" \
  "your-rcac-username@negishi.rcac.purdue.edu:/scratch/negishi/your-rcac-username/"
```

On Negishi, check storage and account access with `myquota` and `slist` before
the full run. A full node requests 128 cores, so the selected account must have
access to that many CPU cores.

## Environment

From the repository on Negishi, set up a private Conda environment with the
cluster's supported `conda` module:

```bash
bash elf_void_analysis/cluster/negishi/setup_environment.sh
```

## Submit in three stages

First submit the manifest job. It indexes POSCAR/ELFCAR pairs once and records
their hashes for provenance by default:

```bash
bash elf_void_analysis/cluster/negishi/submit_prepare.sh
```

Wait until its job finishes successfully (`squeue -u "$USER"` or `jobinfo
<job-id>`). Then submit the single-node extraction job and dependent
merge/validation job:

```bash
bash elf_void_analysis/cluster/negishi/submit_node.sh
```

The manifest hashes source files once by default for provenance. If metadata
storage throughput makes this too slow, set `ELFNET_NO_HASHES=1` in
`negishi.env`; this omits source integrity hashes but leaves extraction intact.

The node job requests one exclusive 128-core CPU node, with all node memory,
for up to four hours. It starts 128 single-core workers. The workers divide
the manifest's 250-sample chunks by worker number, so for 50,000 structures
about 200 chunks are processed in two waves. Merge and dataset validation run
after the node job terminates. If a worker fails before writing a chunk report,
the merge detects the missing output instead of presenting an incomplete
dataset as complete.

Each chunk is saved independently, so completed chunks can be skipped on a
resubmission. Use a new `ELFNET_RUN_ID` for a clean rebuild. Failed partial
chunk folders are intentionally not deleted automatically.

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

## Expected runtime

Based on the local 10-sample benchmark (~4.65 seconds per sample), 50,000
samples imply about 65 serial CPU-hours. At 128 workers, ideal processing time
is about 31 minutes; filesystem throughput, varying ELFCAR sizes, manifest
indexing, and merge/validation add overhead. Queue time is separate. This
estimate assumes the account can schedule a full node.

## Restarting or resubmitting

Completed chunk folders are skipped. A folder containing a partial result
stops that chunk rather than being overwritten. Inspect the failed chunk and
choose a fresh `ELFNET_RUN_ID` for a clean rebuild. The merge step requires a
report and manifest from every expected chunk and refuses an incomplete run.
