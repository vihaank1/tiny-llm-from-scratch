# Update notes

## Implemented

- Renamed the training entry point from inspect_data.py to train.py; importing it no longer trains.
- Retained the hand-built model and fixed inactive dropout and duplicate branch computation.
- Added real output, an evidence-based historical loss curve, exact parameter/corpus counts,
  nats-to-BPC conversion, measured uniform/unigram/bigram baselines, and source attribution.
- Added temperature, top-k, seed, prompt, checkpoint, and length options to generation.
- Added training configuration flags, safe output naming, persistent best checkpoints,
  metrics CSV, environment metadata, and fixed-prompt sample history.
- Added an all-target evaluator and a plotter for CSV/console logs.
- Added an MIT license for code/documentation and separate corpus source/terms notes.
- Added seven regression tests and verified a two-update training/save/load/generation smoke run.

## Interpretation changes

The old source allocated dropout but did not apply it. Historical experiments described
as dropout runs therefore did not exercise dropout in the retrieved implementation.
Current defaults actually apply it; historical quality metrics do not describe the new
training procedure. Evaluation-mode compatibility with old checkpoints is retained.

Training no longer performs an uncounted diagnostic update before step 0. Evaluation
uses fixed random windows and an isolated generator, and sample generation preserves
the training RNG state. The new default seed is 42. These changes intentionally make
future results easier to reproduce; they do not reproduce the old random run exactly.

## Evidence not available

The final expanded-corpus checkpoint, complete training curve, run duration, training
seed, and original Python/chip details were not supplied. No substitutes are presented
as measurements. The included curve is explicitly the earlier single-book experiment.
Run evaluate.py with the actual checkpoint to obtain a matched baseline comparison;
new runs record the previously missing evidence automatically.

No GitHub Release, repository visibility change, push, or profile pinning was performed.
The private update package contains no checkpoint, raw terminal log, or personal name.
