1) rdm row-wise to understand the importance scores x each image instead of batch
2) predicted vs ground truth labels we have it
   - starting from this point 
   - indipendent contribution of the error to any individual image
   - for each individual image which increase vs decrease the distance to the ground truth
3) looking at barlow twins featrue extraction
   - we can use the same approach?
   - understand.
4) Documentation for each of the steps.

POINT 2:

A form of feature ablation or perturbation analysis per object. Here's how:
1. For a single object, loop over all features (or groups of features, like feature maps if they come
from conv layers):
For each feature fi:
• Temporarily set fi = 0 (or to a baseline value).
• Recompute the prediction: ý-i
• Compute the new error: |ý-i - Ytrue|
• Compare with the original error: |ý — Ytrue |
This tells you whether removing that feature improved or worsened the prediction.
2. Rank features for that object based on how much their removal increased or decreased the
prediction error.
This gives you an individual-level feature attribution.