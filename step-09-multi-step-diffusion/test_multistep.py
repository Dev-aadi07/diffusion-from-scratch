import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

model = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(8, 1, 3, padding=1)
)

model.load_state_dict(
    torch.load("multistep_model.pth")
)

model.eval()

image = np.zeros((32, 32), dtype=np.float32)

x = np.random.randint(8, 16)
y = np.random.randint(8, 16)

size = 8

image[y:y+size, x:x+size] = 1.0

noise_strength = 0.20

noise = np.random.randn(32, 32).astype(np.float32) * noise_strength

noisy_image = image + noise

input_tensor = torch.tensor(
    noisy_image,
    dtype=torch.float32
).unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    predicted_noise = model(input_tensor)

predicted_noise = predicted_noise.squeeze().numpy()

denoised_image = noisy_image - predicted_noise

fig, axes = plt.subplots(1, 4, figsize=(12, 3))

axes[0].imshow(noisy_image, cmap="gray")
axes[0].set_title("Noisy")

axes[1].imshow(noise, cmap="gray")
axes[1].set_title("Real Noise")

axes[2].imshow(predicted_noise, cmap="gray")
axes[2].set_title("Predicted Noise")

axes[3].imshow(denoised_image, cmap="gray")
axes[3].set_title("Denoised")

for ax in axes:
    ax.axis("off")

plt.show()