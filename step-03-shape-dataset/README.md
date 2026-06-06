# Step 03 — Shape Dataset Generation

This step focuses on generating synthetic training data for a tiny diffusion model.

Instead of manually drawing images, we now create images programmatically using NumPy arrays and randomness.

## Concepts Covered

* Procedural image generation
* Synthetic datasets
* Random shape creation
* Random positioning
* Random sizing
* Gaussian noise addition
* Clean vs noisy image pairs
* Dataset diversity
* Training data intuition

## Files

* `generate_shapes.py` → generates random geometric shapes with noise

## Key Learning

AI models learn from large collections of examples.
Instead of downloading huge datasets, we generate our own simple training images using code.

This step introduces the idea of creating:

* clean images
* noisy images
* randomized samples

which later become training inputs for the diffusion model.

## Important Insight

Randomness in dataset generation helps the model learn general visual patterns instead of memorizing a single shape or position.
