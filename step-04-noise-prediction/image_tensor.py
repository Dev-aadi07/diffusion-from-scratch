import numpy as np
import torch

image = np.zeros((4, 4), dtype=np.float32)

image[1:3, 1:3] = 255

print(image)

tensor_image = torch.tensor(image)

print(tensor_image)

flat_image = tensor_image.flatten()

print(flat_image)

print(flat_image.shape)