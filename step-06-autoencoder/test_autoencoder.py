import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

model = nn.Sequential(
    nn.Linear(1024, 512),
    nn.ReLU(),

    nn.Linear(512, 256),
    nn.ReLU(),

    nn.Linear(256, 512),
    nn.ReLU(),

    nn.Linear(512, 1024),
    nn.Sigmoid()
)

model.load_state_dict(
    torch.load("square_model.pth")
)

model.eval()

image = np.zeros((32, 32), dtype=np.float32)

x = np.random.randint(8, 16)
y = np.random.randint(8, 16)


# size = np.random.randint(4, 10)
size = 8

image[y:y+size, x:x+size] = 1.0

noise = np.random.randn(32, 32) * 0.2

noisy_image = image + noise

noisy_tensor = torch.tensor(
    noisy_image.flatten(),
    dtype=torch.float32
).unsqueeze(0)

with torch.no_grad():

    prediction = model(noisy_tensor)

print(prediction.min().item())
print(prediction.max().item())
print(prediction.mean().item())

prediction_image = (
    prediction
    .reshape(32,32)
    .numpy()
)

print(prediction.std().item())

prediction_image = np.clip(
    prediction_image,
    0,
    1
)


fig, axes = plt.subplots(
    1,
    3,
    figsize=(9, 3)
)

axes[0].imshow(
    noisy_image,
    cmap="gray"
)
axes[0].set_title("Noisy")

axes[1].imshow(
    prediction_image,
    cmap="gray"
)
axes[1].set_title("Prediction")

axes[2].imshow(
    image,
    cmap="gray"
)
axes[2].set_title("Clean")

for ax in axes:
    ax.axis("off")

plt.show()