import numpy as np
import matplotlib.pyplot as plt

image = np.zeros((100, 100), dtype=np.uint8)
image[30:70, 30:70] = 120

noise = np.random.randint(
  0,50,
  (100, 100),
  dtype=np.uint8
)

noisy_image = image+noise
plt.imshow(noisy_image, cmap='gray')
plt.show()