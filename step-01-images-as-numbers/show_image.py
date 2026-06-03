import numpy as np
import matplotlib.pyplot as plt

# Grayscale image, 1 number = 1 pixel brightness
# image = np.array([
#     [0, 50, 100],
#     [150, 200, 255],
#     [30, 120, 220]
# ])
# plt.imshow(image, cmap='gray')
# plt.show()

# Colorful pixels
# image = np.array([
#   [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
#   [[255, 255, 0], [255, 255, 255], [0, 0, 0]],
#   [[255, 0, 255], [0, 255, 255], [120, 120, 120]]
# ], dtype=np.uint8)
# plt.imshow(image)
# plt.show()


# NumPy tensor creation
image = np.zeros((100, 100, 3), dtype=np.uint8)
image[20:80, 20:80] = [255, 0, 0]
plt.imshow(image)
plt.show()