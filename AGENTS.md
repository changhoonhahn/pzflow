# Working rules

- Before starting a task or resuming after new user instructions, fetch origin and inspect the current branch, working tree, and upstream divergence. Always check for GitHub updates before doing work. If the check fails, say so; never claim the checkout is current.
- Fetch and check again before committing/pushing. Fast-forward clean branches when safe; preserve user edits and never overwrite remote changes or force-push without explicit authorization.
- Keep the tracked directories limited to bin/, nb/, tex/, and src/pzflow/. Root configuration files are allowed. Put executable workflows in bin/, reusable functions in src/pzflow/, and notebook source in nb/.
- Keep data, results, figures, checkpoints, executed notebook outputs, caches, and archives local and ignored. Do not re-add them to Git. Save executed notebooks in results/notebooks_executed/.
- Keep code minimal. Use the single requirements.txt and setup.py; avoid duplicate wrapper/implementation layers and redundant notebook generators.
- Maintain data provenance and reproducibility notes in bin/data/. Preserve the user's concise README.

- Keep the active workflow to bin/lsst_like.py, bin/match_cosmos.py, bin/train_flows.py, and bin/run_eazy.py. All analyses use data/parent.npz. Retain only one small shared utility file, src/pzflow/utils.py, plus package initialization. Ask before adding more shared modules.
