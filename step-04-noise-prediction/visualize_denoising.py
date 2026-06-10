import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

image = np.zeros((32, 32), dtype=np.float32)
image[10:22, 10:22] = 1.0

noise = np.random.randn(32, 32) * 0.2

noisy_image = image + noise

clean_tensor = torch.tensor(
    image.flatten(),
    dtype=torch.float32
).unsqueeze(0)

noisy_tensor = torch.tensor(
    noisy_image.flatten(),
    dtype=torch.float32
).unsqueeze(0)

model = nn.Linear(1024, 1024)

loss_function = nn.MSELoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01
)

for step in range(300):

    output = model(noisy_tensor)

    loss = loss_function(
        output,
        clean_tensor
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    prediction = model(noisy_tensor)

prediction_image = (
    prediction
    .reshape(32, 32)
    .numpy()
)

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