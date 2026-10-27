# Repository sync rule

This project has a Negishi checkout at `/home/weilana/electride-ml-predictor` and a local checkout at `/Users/andrewweiland/Documents/ChatGPT/Electride ML Prediction/electride-ml-predictor`. Both use `origin` (`git@github.com:weilana30/electride-ml-predictor.git`).

After any significant change intended to be kept, commit and push it from the checkout where it was made, then fast-forward pull the other checkout before reporting the change complete. This applies in either direction. Check `git status` and fetch first. If either checkout has uncommitted work or the branches diverge, inspect and reconcile them without discarding commits or force-pushing. Report any sync failure and the unsynced checkout.

Keep datasets, model runs, checkpoints, and Slurm logs outside the repository unless the user explicitly asks to version a specific artifact.
