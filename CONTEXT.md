# Image Authenticity Prediction

Predicting human judgments of visual authenticity and technical quality in AI-generated imagery, investigating whether deep neural representations align with human perceptual criteria and whether post-hoc model explanations are identifiable.

## Language

### Behavioral & Psychometric Constructs

**Authenticity Rating**:
Subjective Mean Opinion Score (MOS) measuring the degree to which human observers judge an image to be a real, natural photograph rather than computer-generated.
_Avoid_: Realness score, truth score, credibility

**Quality Rating**:
Subjective Mean Opinion Score (MOS) measuring human judgment of visual fidelity, technical sharpness, and rendering clarity.
_Avoid_: Aesthetics score, resolution score, fidelity

**Correspondence Rating**:
Subjective Mean Opinion Score (MOS) measuring the degree to which an image semantically matches the textual prompt used for its generation.
_Avoid_: Text-image alignment, prompt relevance, text match

**Human Noise Ceiling**:
The theoretical upper bound on empirical correlation with human ratings, defined as the square root of the Spearman–Brown-corrected split-half reliability.
_Avoid_: Empirical ceiling, r-max, human agreement limit

**Model Prediction Reliability**:
The mean pairwise Pearson correlation between prediction vectors generated across independently initialized instances of a given architecture.
_Avoid_: Model stability, seed variance, run repeatability

**Combined Correlation Ceiling**:
The estimated upper bound on observable model–human correlation ($r_{\max} \approx \sqrt{r_{\text{human}} \cdot r_{\text{model}}}$) integrating measurement noise from both human observers and model instantiations.
_Avoid_: Total ceiling, joint noise limit, adjusted correlation

### Dataset & Stimulus Sampling

**Kept Dataset**:
The subset of 1,368 photorealistic natural-scene images (57 prompts × 6 generative models × 4 images) retained for model training, testing, and behavioral comparison.
_Avoid_: Filtered dataset, valid dataset, in-distribution set

**Discarded Dataset**:
The subset of 1,032 non-photorealistic images (43 prompts × 6 models × 4 images) excluded from the experimental corpus due to artificial or non-natural prompt criteria.
_Avoid_: Bad images, outlier images, trash set

**Prompt-Level Exclusion**:
A metadata-driven filtering criterion that removes entire prompt cohorts based on generation intent rather than selecting individual image outputs.
_Avoid_: Post-hoc filtering, image culling, manual curation

**Perceptual Dissociation**:
A stimulus exhibiting substantial divergence between normalized quality and authenticity ratings ($|z_Q - z_A| \ge 1.0$), demonstrating non-redundancy between constructs.
_Avoid_: Annotation error, label noise, misfit

**Quantile Decile**:
An equal-frequency 10-bin partition ($\approx 10\%$ of stimuli per bin) used to analyze conditional rating distributions without tail sample-size distortion.
_Avoid_: Equal-interval bin, score bracket

### Modeling & Training Regimes

**Frozen Backbone**:
A pretrained vision network whose convolutional or transformer weights remain fixed while only a downstream regression head is adapted.
_Avoid_: Fixed encoder, pretrained trunk, base network

**Self-Supervised Objective**:
A pretraining regime (e.g., Barlow Twins) that learns representations by minimizing redundancy across augmented views without human class labels.
_Avoid_: Barlow Twins architecture, Barlow Twins model, unsupervised backbone

**Regression Head**:
A lightweight multi-layer perceptron attached to a frozen backbone that maps extracted feature embeddings to a scalar authenticity score.
_Avoid_: Readout layer, adapter network, prediction head, linear probe

**Model Variant**:
An independently trained instance of an architecture with a distinct random seed governing parameter initialization and data partitioning.
_Avoid_: Seed run, checkpoint, iteration, training trial

**Validation-Based Pruning**:
A feature selection protocol using Sequential Backward Selection (SBS) on validation data to identify essential readout channels before testing.
_Avoid_: Test-set pruning, in-sample channel selection, feature drop

### Quality vs. Authenticity Disentanglement

**Authenticity-Specific Variance**:
The residual association between predicted and observed authenticity remaining after linearly partialling out quality ratings ($r(A, \hat{A} \mid Q)$).
_Avoid_: Unique authenticity, pure authenticity, non-quality variance

**Quality-Shared Variance**:
The mutual linear covariance between perceived quality and authenticity reflecting overlapping perceptual criteria or shared low-level rendering cues.
_Avoid_: Quality confound, proxy variance, quality artifact

### Explainability & Attribution

**Attribution Heatmap**:
A spatial importance map projecting positive and negative pixel-level or region-level contributions to a model's predicted scalar rating.
_Avoid_: Saliency map, explanation map, relevance map, attention map

**Grad-CAM**:
A gradient-based attribution method that weights target feature maps by global-average-pooled output gradients to localize influential regions.
_Avoid_: Activation map, gradient heatmap, CAM

**Multi-scale Pixel Masking (MPM)**:
A model-agnostic attribution method evaluating the prediction change caused by zero-masking image patches across multiple spatial scales ($\sigma \in \{3, 17, 65\}$).
_Avoid_: Occlusion map, sliding patch mask, perturbation saliency

**LIME Explanation**:
A perturbation-based explanation that estimates local region contributions via an interpretable linear surrogate fitted over segmented superpixels.
_Avoid_: Superpixel mask, surrogate map, local explanation

**Surrogate Fidelity**:
The coefficient of determination ($R^2$) quantifying how accurately a local surrogate model reproduces base network predictions over perturbed inputs.
_Avoid_: Surrogate accuracy, LIME score, explanation validity

### Consistency & Identifiability

**Within-Architecture Consistency**:
The pairwise spatial concordance (correlation or Intersection over Union) of attribution heatmaps across independently trained variants of the same architecture.
_Avoid_: Intra-model consistency, seed stability, internal reliability

**Across-Architecture Consistency**:
The spatial correlation between architectural prototype heatmaps across distinct model backbones for the same image.
_Avoid_: Inter-model agreement, cross-backbone consistency, model consensus

**Architectural Prototype Heatmap**:
The mean attribution map produced by averaging the heatmaps of all trained variants of a given architecture for a single image.
_Avoid_: Average heatmap, centroid map, architecture consensus

**Rashomon Effect (Non-Identifiability)**:
The occurrence where distinct vision architectures achieve comparable behavioral prediction accuracy while producing divergent, irreconcilable attribution heatmaps.
_Avoid_: Model multiplicity, explanatory ambiguity, attribution discordance

### Ensembling & Aggregation

**Bagging Ensemble**:
An aggregated model whose prediction is the arithmetic mean of scalar outputs across all trained variants and architectures.
_Avoid_: Average ensemble, bootstrap aggregating, mean voting

**Stacking Meta-Learner**:
A level-1 linear regression model trained via out-of-fold cross-validation on base-model predictions to optimize behavioral score estimation.
_Avoid_: Blending, meta-regressor, stacked generalization

**Consensus Predictor**:
An ensemble model combining diverse architectural representations to achieve superior behavioral alignment, evaluated with model-agnostic explanation tools.
_Avoid_: Aggregate model, consensus explanation, unified model
