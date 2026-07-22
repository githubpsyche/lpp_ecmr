# Pooled stagnation fits

This is an experimental rerun of the same 16 pooled models and three
independent restarts per model used by `work/pooled_model_runs/`. It changes
only the differential-evolution stopping rule:

- stop after a complete 100-generation window in which the best NLL improves
  by less than `0.01`;
- continue when the improvement is exactly `0.01`;
- never exceed 1,000 generations.

The standard population-spread rule remains the repository default. These
fits use the explicit `stagnation` campaign, a distinct run tag, and a
separate run directory. They do not replace canonical fits, simulations,
figures, or manuscript results.

No fitting notebooks are duplicated here. The cluster tasks call the same
executable 16-model registry and restart runner as the standard campaign.

## Stage and submit on CSD3

After the lpp_ecmr and jaxcmr changes have been committed, pushed, and pulled
into clean CSD3 checkouts, run these commands in the authenticated CSD3 web
shell:

```bash
cd ~/workspace/lpp_ecmr
RUN_DIR="$(bash work/pooled_stagnation_fits/stage_remote_run.sh)"
bash "$RUN_DIR/submit_fit_restarts.sh" 16 "$RUN_DIR"
```

The staging command copies exact source snapshots and the fitting dataset into
one timestamped directory under `~/lpp_ecmr_runs/`. The submission command
runs a preflight, then submits array tasks `0-47` with at most 16 running at
once.

## Check completion

Reconnect to the web shell and run:

```bash
RUN_DIR=~/lpp_ecmr_runs/lpp-stagnation-YYYYMMDD-HHMMSS
squeue -j "$(cat "$RUN_DIR/submission.txt")"
find "$RUN_DIR/restarts" -maxdepth 1 -name '*_restart_*.json' | wc -l
```

Completion means the Slurm array is no longer queued or running and the count
is exactly 48. Email is not required for this check.

Do not run reduction with `--install` into `work/pooled_model_runs/`. Once all
48 files are complete, archive the run's `restarts/` directory and download it
through Open OnDemand Files. Reduction, comparison with the standard fits,
and any adoption decision happen locally in a separate output directory.

After the archive has been downloaded and verified locally, the timestamped
remote run directory is disposable.
