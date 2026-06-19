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
    torch.load("../step-09-multi-step-diffusion/multistep_model.pth")
)

model.eval()

image = np.random.randn(
    32,
    32
).astype(np.float32)

snapshots = [image.copy()]

for step in range(20):

    input_tensor = torch.tensor(
        image,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        predicted_noise = model(input_tensor)

    predicted_noise = (
        predicted_noise
        .squeeze()
        .numpy()
    )

    image = image - 0.3 * predicted_noise

    if step in [0, 4, 9, 19]:
        snapshots.append(image.copy())

fig, axes = plt.subplots(
    1,
    len(snapshots),
    figsize=(15, 3)
)

titles = [
    "Start",
    "Step 1",
    "Step 5",
    "Step 10",
    "Step 20"
]

for ax, img, title in zip(
    axes,
    snapshots,
    titles
):
    ax.imshow(img, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.show()