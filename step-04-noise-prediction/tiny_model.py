import torch
import torch.nn as nn

model = nn.Linear(4, 2)

input_data = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

target = torch.tensor([[0.5, 1.0]])

output = model(input_data)

loss_function = nn.MSELoss()

loss = loss_function(output, target)

print(output)

print(loss)