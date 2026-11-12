# Repository sync rule

This project has a checkout on the Negishi cluster and a local checkout, both using `origin` (the GitHub repository for this project).

After any significant change intended to be kept, commit and push it from the checkout where it was made, then fast-forward pull the other checkout before reporting the change complete. This applies in either direction. Check `git status` and fetch first. If either checkout has uncommitted work or the branches diverge, inspect and reconcile them without discarding commits or force-pushing. Report any sync failure and the unsynced checkout.

Keep datasets, model runs, checkpoints, and Slurm logs outside the repository unless the user explicitly asks to version a specific artifact.
