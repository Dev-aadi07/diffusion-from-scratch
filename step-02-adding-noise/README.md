# Step 02 — Adding Noise

This step focuses on understanding the forward diffusion process by gradually corrupting images using random noise.

## Concepts Covered

* Random noise generation
* Gaussian noise
* Forward diffusion intuition
* Iterative noise addition
* Pixel corruption
* Noise strength control
* Image clipping
* Float vs integer image representations

## Files

* `add_noise.py` → adding random noise to an image
* `forward_diffusion.py` → simulating gradual diffusion over multiple steps

## Key Learning

Diffusion models work by gradually adding noise to images until the original structure disappears.
Instead of learning to generate images directly, the model later learns how to reverse this corruption process step-by-step.

## Important Insight

The diffusion process is easier to reverse when noise is added gradually rather than instantly destroying the image.
