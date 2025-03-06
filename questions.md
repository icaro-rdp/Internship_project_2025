1) does the feature extraction has to be done at the penultimate fc layer of the pre-trained model or the 512x7x7 flattened last conv feature ? also for our case, in which the fc has learned classification head for another task? Do we want to use the finetuned? Both?

## Similarities

2) can u clarify this for me? We define 'similarity' in the 'quality' or 'realness' spaces as differences in values between any pair of images? normal difference, quadratic difference, cosine similarity, etc?

3) Similarity btw embedding spaces, do we take cosine sim?

# What i did
## What You Did in the File

1. **Created RSM-based analysis framework**: You developed a system that compares representation similarity matrices from CNN features with quality difference matrices.

2. **Implemented feature extraction**: You extract CNN features from the last layer of VGG16, maintaining the spatial information (shape: n_samples × 512 × 7 × 7), TBD if you want to use the penultimate layer.

3. **Built similarity matrices**: You calculate cosine similarity between feature vectors to create a model-based RSM.

4. **Created quality difference matrices**: You compute DIFFERENCES (which diff?) between quality scores to create a quality-based RSM.

5. **Measured alignment**: You use Spearman correlation between these matrices as your "fit" metric, with higher correlation indicating better alignment.

6. **Analyzed channel importance**: You systematically removed each of the 512 channels to measure their impact on the alignment between RSMs.

7. **Implemented multiple pruning strategies**:
   - Threshold-based pruning (removing channels above impact threshold, Removing only the negative impact channels)
   - Greedy search (incrementally removing channels that improve alignment)

9. **Evaluated pruning effectiveness**: You measured alignment improvement after pruning, with your greedy approach showing the best improvement.

"---------------------------------------------------------------"

Responses to the questions:

- analys of the distribution on terms of quality labels.
- can we prune rmse and understand the impact in terms of matrices?

Another topic:
- One direction we can think of is using the knowledge we obtain from which image sections are less important for Quality (or Authenticity) to simplify an image so it can be better compressed.  It also allows setting up a simple psychology experiment seeing whether removing information from areas of the image that our method identifies as less important for Quality has a weaker impact on quality ratings than removing information from areas of the image that our method identifies as more important.