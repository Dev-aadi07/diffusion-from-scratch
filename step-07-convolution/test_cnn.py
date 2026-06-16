import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(8, 1, kernel_size=3, padding=1),
    nn.Sigmoid()
)

model.load_state_dict(
    torch.load("cnn_model.pth")
)

model.eval()

image = np.zeros((32, 32), dtype=np.float32)

x = np.random.randint(8, 16)
y = np.random.randint(8, 16)

size = 8

image[y:y+size, x:x+size] = 1.0

noise = np.random.randn(32, 32) * 0.2

noisy_image = image + noise

input_tensor = torch.tensor(
    noisy_image,
    dtype=torch.float32
).unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    prediction = model(input_tensor)

prediction_image = prediction.squeeze().numpy()

fig, axes = plt.subplots(1, 3, figsize=(9, 3))

axes[0].imshow(noisy_image, cmap="gray")
axes[0].set_title("Noisy")

axes[1].imshow(prediction_image, cmap="gray")
axes[1].set_title("Prediction")

axes[2].imshow(image, cmap="gray")
axes[2].set_title("Clean")

for ax in axes:
    ax.axis("off")

plt.show()