import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(4, 2)

input_data = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

target = torch.tensor([[0.2, 0.1]])

loss_function = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)

for step in range(10):
  output = model(input_data)
  
  loss = loss_function(output, target)
  
  optimizer.zero_grad()
  
  loss.backward()
  
  optimizer.step()
  
  print(f"Step {step}: Loss = {loss.item()}")