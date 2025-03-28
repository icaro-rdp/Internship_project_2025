1. Feature Map Extraction
For each image in your dataset, pass it through the neural network and extract the activation maps from the deepest convolutional layer (in our case 512 per image)
2. Define Hollow Square Templates
Use the same height and width of the feature map, and construct hollow sqare templates that are binary masks. The length of each side of the square should be  50%, 60%, 70%, 80%, 90% of the feature map height/width).  For example, of the activation maps are 14x14, the smallest square will be 7x7 centered.
The square is open, i.e., only the edges are marked — top, bottom, left, and right — not the interior.
3. Match Templates to Feature Maps
For each feature map (i.e., each channel) in each image:
Compare the activation map to each of the square templates using a similarity measure (e.g., normalized cross-correlation or cosine similarity).
This yields one match score per square size.
From these, record:
The maximum match score (i.e., best-fitting square size) (Store Pearson correlation) AND
The corresponding size ratio that produced the maximum score
4. Aggregate Results Across Images
For each feature map across the dataset:
Collect the best match scores for all images (Correlation and size)
Analyze the distribution of these scores:
A feature map that often has high scores may be consistently detecting hollow squares.
A feature map with a bimodal or heavy-tailed distribution may only activate in the presence of the artifact.