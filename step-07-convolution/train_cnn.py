import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

images = []
targets = []

for _ in range(1000):

    image = np.zeros((32, 32), dtype=np.float32)

    x = np.random.randint(8, 16)
    y = np.random.randint(8, 16)

    size = 8

    image[y:y+size, x:x+size] = 1.0

    noise = np.random.randn(32, 32) * 0.2

    noisy_image = image + noise

    images.append(noisy_image)
    targets.append(image)

images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),
    nn.ReLU(),

    nn.Conv2d(8, 1, kernel_size=3, padding=1),
    nn.Sigmoid()
)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

for epoch in range(300):

    predictions = model(images)

    loss = criterion(
        predictions,
        targets
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 5 == 0:
        print(
            f"Epoch {epoch}: {loss.item()}"
        )

torch.save(
    model.state_dict(),
    "cnn_model.pth"
)