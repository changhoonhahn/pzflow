# Working rules

- Before starting a task or resuming after new user instructions, fetch origin and inspect the current branch, working tree, and upstream divergence. Always check for GitHub updates before doing work. If the check fails, say so; never claim the checkout is current.
- Fetch and check again before committing/pushing. Fast-forward clean branches when safe; preserve user edits and never overwrite remote changes or force-push without explicit authorization.
- Keep the tracked directories limited to bin/, nb/, tex/, and src/pzflow/. Root configuration files are allowed. Put executable workflows in bin/, reusable functions in src/pzflow/, and notebook source in nb/.
- Keep data, results, figures, checkpoints, executed notebook outputs, caches, and archives local and ignored. Do not re-add them to Git. Save executed notebooks in results/notebooks_executed/.
- Keep code minimal. Use the single requirements.txt and setup.py; avoid duplicate wrapper/implementation layers and redundant notebook generators.
- Maintain data provenance and reproducibility notes in bin/data/. Preserve the user's concise README.

- Keep the active workflow to bin/lsst_like.py, bin/match_cosmos.py, bin/train_flows.py, bin/run_eazy.py, and bin/compare_som.py. All analyses use data/parent.npz. Retain only one small shared utility file, src/pzflow/utils.py, plus package initialization. Ask before adding more shared modules.

## Default Python style

Use these conventions for new code and when editing existing code. They are
inferred from 15 source files reviewed on 2026-11-20 in five public repositories,
with recent scientific code in px2cosmo weighted most heavily. The repositories
vary: floody.ai2 uses more formal documentation and more double quotes. The rules
below resolve that variation for pzflow; they are not claims of a uniform upstream
standard. Preserve working behavior and public interfaces during style changes.

### Structure and naming

- Write direct scientific code: a readable sequence of load, select, transform,
  fit, evaluate, and save. Use short functions for distinct operations and keep
  configuration close to the calculation. Avoid wrapper layers, generic pipeline
  frameworks, and classes that only hold configuration. Classes are appropriate
  for stateful models, samplers, and neural networks.
- Use snake_case for functions, arguments, and ordinary variables, and PascalCase
  for classes. Retain conventional scientific symbols where clear: z, flux, err,
  pdf, theta, mu, sigma. Use descriptive short names such as dat_dir, n_train,
  i_train, train_mask, and zspec. Do not rename established interfaces just to
  impose this convention.
- Keep imports at the top, grouped as standard library, numerical/third-party,
  and project imports. Use numpy as np, matplotlib.pyplot as plt, torch, and
  torch.nn as nn. An established module alias such as util as UT is fine when
  useful; do not invent aliases for every module.
- Put command-line execution behind a main guard. Use argparse for configurable
  scripts; a separate parse_args() is useful when argument setup is substantial.
  Keep the existing workflow scripts and the single shared utils.py.

### Syntax and formatting

- Use four spaces for indentation, two blank lines between top-level definitions,
  and blank lines between scientific steps. One logical statement per line;
  avoid semicolon-packed assignments and compressed control flow.
- Prefer single-quoted strings, including dictionary keys, and f-strings for new
  interpolated messages and paths. Preserve existing % formatting when editing a
  small region. Use double quotes when they avoid escaping.
- Use spaces around assignments, comparisons, and boolean operators, and after
  commas. Keep keyword arguments compact: axis=1, dtype=np.float32. Compact
  arithmetic such as 10**x or (zphot-zspec)/(1+zspec) is fine when readable;
  space longer expressions and parenthesize array comparisons before & or |.
- Wrap long calls and expressions in parentheses, with continuation arguments
  aligned or indented clearly. Aim for roughly 100 columns without mechanically
  reflowing readable formulas. Small groups of related assignments may align =
  signs for readability; do not require alignment throughout a file.
- Type hints are optional. Default to ordinary function signatures for scientific
  scripts, and document units/shapes where needed. Do not add typing machinery,
  dataclasses, or configuration schemas solely for style.

### Scientific implementation and documentation

- Prefer explicit NumPy/PyTorch array operations, boolean masks, slicing, and
  named intermediate quantities. Use loops for genuinely sequential work or
  memory-bounded batches. Make array shapes, axis choices, and units clear.
- Use a short module docstring explaining the scientific purpose. Give functions
  concise docstrings; document parameters, returns, shapes, and units for complex
  interfaces using a consistent local format. Triple single quotes are the
  default for new docstrings; keep an existing file's convention when practical.
- Use brief comments describing the scientific step, assumption, or reason:
  # load data, # select training sample, # normalize PDFs. Add equations and
  references when they explain a non-obvious calculation. Short section markers
  such as # -- training -- are fine; avoid decorative comment blocks and
  comments that merely repeat the next line.
- Use assertions for internal invariants and explicit exceptions for invalid
  user inputs or missing data. Keep validation focused on scientific correctness.
- Record seeds, selection cuts, noise definitions, and source provenance explicitly.
  Preserve pzflow's existing reproducibility rules. Keep paths configurable and
  use context managers for files; machine-specific paths and legacy API calls in
  upstream examples are not defaults to copy.
- Preserve the agreed scientific plotting style: serif/LaTeX labels, axes width
  1.5, major ticks of length 5 and width 1.5, and frameless legends. Magnitude
  comparisons use actual bin-median magnitudes, galaxy counts as text, and
  fill_between uncertainty bands with alpha=0.5.

### Source examples

Reviewed snapshots (links pin the actual source, rather than mutable branches):

- [px2cosmo training script](https://github.com/changhoonhahn/px2cosmo/blob/4cd50fa3e7875a985d86839a3b28b92dd9f05330/bin/mock/nde_likelihood.py)
  and [Deep Set model](https://github.com/changhoonhahn/px2cosmo/blob/4cd50fa3e7875a985d86839a3b28b92dd9f05330/src/px2cosmo/deepset.py).
- [provabgs scientific utilities](https://github.com/changhoonhahn/provabgs/blob/f50f274ec8b1caf011c70fac13903c53c567ad5e/src/provabgs/util.py).
- [SEDflow data preparation](https://github.com/changhoonhahn/SEDflow/blob/4561ecfe3a38cc4c25df263d971a87e8a83f88ce/src/sedflow/train.py).
- [pySpectrum plotting script](https://github.com/changhoonhahn/pySpectrum/blob/6c8615b3aa7033a17b3dda494b1c8b1652b50078/run/bk_demo.py).
- [floody.ai2 dataset pipeline](https://github.com/changhoonhahn/floody.ai2/blob/5eaa031d7183ca7124ffd4ea225e771bf8e0a58d/bin/data/ai2_dataset.py).
