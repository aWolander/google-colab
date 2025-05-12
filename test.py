import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import random
import numpy as np

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
class DinoFullModel(nn.Module):
    def __init__(self):
      super().__init__()
      # Load the full DINO model (backbone + head)
      self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

      self.learning_rate = 1e-2
      self.epochs = 10

      self.loss_fn = nn.CrossEntropyLoss()
      self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

      self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        self.optimizer, T_max=self.epochs
      )

    def forward(self, x):
      return self.model(x)

    def train_model(self, dataloader):
      size = len(dataloader.dataset)
      # Set the model to training mode - important for batch normalization and dropout layers
      # Unnecessary in this situation but added for best practices
      self.train()
      for epoch in range(self.epochs):
        current = 0
        print(f"-------------------------------\nEpoch {epoch+1}\n-------------------------------")
        for (X, y) in dataloader:
          # Compute prediction and loss
          X = X.to(device)
          y = y.to(device)
          pred = self.model(X).to(device)
          loss = self.loss_fn(pred, y)

          # Backpropagation
          loss.backward()
          self.optimizer.step()
          self.optimizer.zero_grad()
          current += len(X)
          if current % 5000 == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            loss = loss.item()

        self.scheduler.step()



    def test_model(self, dataloader):
      self.model.eval()
      total_loss = 0.0
      total_correct = 0
      total_samples = 0

      with torch.no_grad():
        for inputs, targets in dataloader:
          inputs, targets = inputs.to(device), targets.to(device)

          outputs = self.model(inputs)
          loss = self.loss_fn(outputs, targets)
          total_loss += loss.item() * inputs.size(0)  # total loss, not average

          # Get predicted class
          preds = outputs.argmax(dim=1)
          total_correct += (preds == targets).sum().item()
          total_samples += targets.size(0)

      avg_loss = total_loss / total_samples
      accuracy = total_correct / total_samples

      #print(f"Validation Loss: {avg_loss:.4f}, Correct: {total_correct} out of {len(dataloader)}, Accuracy: {accuracy:.2%}")

      return avg_loss, total_correct, accuracy

def main():

  train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(CIFAR100, [0.7,0.15,0.15])

  train_dataloader = DataLoader(train_dataset, batch_size=100)
  test_dataloader = DataLoader(test_dataset, batch_size=100)
  validate_dataloader = DataLoader(validate_dataset, batch_size=100)

  model = DinoFullModel().to(device)
  #for param in model.parameters():
  #    print(param.shape)

  model.train_model(train_dataloader)
  model.test_model(test_dataloader)
  print("Done!")

if __name__=="__main__":
  main()