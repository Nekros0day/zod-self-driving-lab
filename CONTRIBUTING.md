# Contributing

Contributions should make the benchmark easier to trust, reproduce, or explain.

1. Create a focused branch and keep raw ZOD data, access links, caches, and
   checkpoints outside the repository.
2. Reproduce the issue with synthetic data where possible.
3. Add a test before changing geometry, timestamp alignment, split logic,
   multiple shooting, spectral operators, masks, thresholds, or metrics.
4. Run `python -m ruff check src tests scripts` and `python -m pytest`.
5. State whether the change can alter a manifest hash, target convention, metric
   definition, or released model output. These are benchmark-breaking changes.
6. For an experiment contribution, include the resolved config, code/split/data
   hashes, seed, hardware, runtime, checkpoint rule, and all planned results,
   including negative ones.

Notebook code may demonstrate package APIs but must not become the only place an
algorithm exists. Reusable behavior belongs under `src/zod_driveformer` with a
data-free test.
