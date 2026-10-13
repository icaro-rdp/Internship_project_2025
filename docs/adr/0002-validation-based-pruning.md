---
status: proposed
---

# Migrate Experiment 1B channel pruning to validation split to prevent test-set leakage

To address Reviewer 1 Major Comment 6 (#4), we ensure channel and unit pruning across all experiments is conducted strictly on validation data rather than the test partition. In the original Experiment 1 pipeline (`experiment_one.py`), channel importance ranking and greedy pruning evaluated candidate removals on `test_loader`. While Experiment 3 previously resolved this by pruning on `val_loader` (developed on branch `test-pruning-on-validation` and merged into `main`), Experiment 1B still contains the legacy test-set evaluation. We align Experiment 1B with Experiment 3 by computing importance scores and greedy pruning decisions on `val_loader`, reserving the global test set exclusively for final out-of-sample evaluation.

## Context & Problem

In Reviewer 1 Major Comment 6 (#4), the reviewer noted:
> *"The explanations appear to be generated after label-informed channel pruning on the same 'test' images being explained. This introduces test-set leakage into the explanation pipeline, because the labels of the images being explained influence the channel selection used to generate their attribution maps. The resulting maps therefore explain a test-optimized model rather than the original out-of-sample predictor. To preserve the out-of-sample interpretation, channel selection should be conducted exclusively on training/validation data and fixed before explanation of the test set."*

In Experiment 1B, `FeatureMapsPruner` received `dataloader=test_loader`. This optimized channel selection against the very test partition used to evaluate generalization and generate attribution maps.

## Decisions

- **Reconstruct `val_loader` per variant in Experiment 1B.** Because each variant's data split is completely deterministic (derived from `seed = 42 + variant_idx`), we reconstruct the exact 10% validation split for each variant during Exp 1B without needing to retrain the Exp 1A baseline weights.
- **Pass `dataloader=val_loader` to `FeatureMapsPruner`.** Both importance score computation and greedy pruning iterations will evaluate candidate channel removals on the validation split.
- **Evaluate pruned weights exclusively on the held-out `test_loader`.** Once pruned model weights are saved, the testing phase evaluates both baseline and pruned models on the untouched global test partition.
- **Status is `proposed` pending GPU server verification.** Before triggering an Exp 1B rerun on GPU Server 1 or 2, we must inspect the GPU server storage / SMB share (`cimec-storage/tmp_uri`) to verify whether validation-pruned Experiment 1 weights and evaluation tables already exist from previous runs on the `test-pruning-on-validation` branch.

## Considered Options

- **Keep pruning on test set and reframe as "in-sample feature attribution" (Rejected):** This fails to address the reviewer's concern and undermines claims regarding out-of-sample generalization.
- **Prune on training set (Rejected):** Pruning on the training set risks removing channels that generalize well but appear redundant on the training distribution. The validation set is the standard target for model selection and regularization.
- **Prune on validation set (Chosen):** Matches standard machine learning best practices and guarantees complete consistency between Experiment 1 and Experiment 3.

## Consequences

- **Exp 1A weights remain untouched:** The 10 baseline variants per architecture do not need retraining; only the pruning pass and subsequent test evaluations are affected.
- **If results do not exist on the GPU server:** Exp 1B pruning and testing must be executed on GPU Server 1 (RTX 3090) or Server 2 (RTX 4090) across the 6 existing backbones (and the incoming CLIP backbone).
- **Manuscript alignment:** Completely resolves Issue #4 and strengthens the methodological response to Reviewer 1.
