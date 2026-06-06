import numpy as np
import matplotlib.pyplot as plt

image = np.zeros((64, 64), dtype=np.float32)

x = np.random.randint(5, 20)
y = np.random.randint(5, 20)

size = np.random.randint(5, 10)

image[y:y+size, x:x+size] = 255


# adding noise
noise = np.random.randn(64, 64)*20
noisy_image = image+noise
noisy_image = np.clip(noisy_image, 0, 255)


plt.imshow(noisy_image, cmap='gray')
plt.show()