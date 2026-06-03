import numpy as np
import matplotlib.pyplot as plt

image = np.random.randint(
    0, 256,
    (300, 300, 3),
    dtype=np.uint8
)

plt.imshow(image)
plt.show()