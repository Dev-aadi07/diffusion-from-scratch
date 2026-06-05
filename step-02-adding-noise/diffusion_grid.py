import numpy as np
import matplotlib.pyplot as plt

image = np.zeros((100, 100), dtype=np.float32)

# image[30:70, 30:70] = 255

# Circle parameters
center_x, center_y = 50, 50
radius = 20

Y, X = np.ogrid[:100, :100]
mask = (X - center_x)**2 + (Y - center_y)**2 <= radius**2

image[mask] = 255

current_image = image.copy()

fig, axes = plt.subplots(1, 20, figsize=(15, 3))

for step in range(20):
  noise = np.random.randn(100, 100)*70
  
  current_image = current_image+noise
  current_image = np.clip(current_image, 0, 255)
  
  axes[step].imshow(current_image, cmap='gray')
  
  axes[step].set_title(f"Step {step}")
  axes[step].axis("off")

plt.show()