import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

images = []
targets = []

noise_levels = [0.05, 0.10, 0.15, 0.20]

for _ in range(2000):

    image = np.zeros((32, 32), dtype=np.float32)

    x = np.random.randint(8, 16)
    y = np.random.randint(8, 16)

    size = 8

    image[y:y+size, x:x+size] = 1.0

    noise_strength = np.random.choice(noise_levels)

    noise = np.random.randn(32, 32).astype(np.float32) * noise_strength

    noisy_image = image + noise

    images.append(noisy_image)
    targets.append(noise)

images = torch.tensor(
    np.array(images),
    dtype=torch.float32
).unsqueeze(1)

targets = torch.tensor(
    np.array(targets),
    dtype=torch.float32
).unsqueeze(1)

model = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1),
    nn.ReLU(),

    nn.Conv2d(8, 1, 3, padding=1)
)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

for epoch in range(200):

    prediction = model(images)

    loss = criterion(
        prediction,
        targets
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: {loss.item()}")

torch.save(
    model.state_dict(),
    "multistep_model.pth"
)