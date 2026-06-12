import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

images = []
noisy_images = []

for _ in range(1000):

    image = np.zeros((32, 32), dtype=np.float32)

    x = np.random.randint(8, 16)
    y = np.random.randint(8, 16)
    
    # size = np.random.randint(4, 10)
    size=8

    image[y:y+size, x:x+size] = 1.0

    noise = np.random.randn(32, 32) * 0.2

    noisy_image = image + noise

    images.append(image.flatten())

    noisy_images.append(noisy_image.flatten())

clean_tensor = torch.tensor(
    np.array(images),
    dtype=torch.float32
)

noisy_tensor = torch.tensor(
    np.array(noisy_images),
    dtype=torch.float32
)

model = nn.Sequential(
    nn.Linear(1024, 2048),
    nn.ReLU(),
    nn.Linear(2048, 1024),
    nn.Sigmoid()
)

loss_function = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)

for epoch in range(500):

    output = model(noisy_tensor)

    loss = loss_function(
        output,
        clean_tensor
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
    "square_model.pth"
)