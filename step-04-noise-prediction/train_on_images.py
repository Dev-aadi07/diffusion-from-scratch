import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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

for step in range(100):

    output = model(noisy_tensor)

    loss = loss_function(
        output,
        clean_tensor
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if step % 10 == 0:
        print(
            f"Step {step}: {loss.item()}"
        )