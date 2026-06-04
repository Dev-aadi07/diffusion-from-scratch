import numpy as np
import matplotlib.pyplot as plt

image = np.zeros((100, 100), dtype=np.float32)

image[30:70, 30:70] = 255

current_image = image.copy()

for  step in range(10):
  
  noise = np.random.randn(100, 100)*10
  
  current_image = current_image+noise
  
  current_image = np.clip(current_image, 0, 255)
  
  plt.imshow(current_image, cmap='gray')
  plt.title(f"Step {step}")
  plt.show()