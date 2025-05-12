from __future__ import annotations
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import random
import numpy as np

class FL_client(DinoFullModel):
  def __init__(self):
    super().__init__()

  def __add__(self, other: FL_client) -> FL_client:
    assert isinstance(other, FL_client)
    temp_weights = []
    temp_biases = []
    for (name, self_param), (_,other_param) in zip(self.named_parameters(), other.named_parameters()):
      self_param.to(device)
      other_param.to(device)
      if 'weight' in name:
        temp_weights.append(self_param + other_param)
      elif 'bias' in name:
        temp_biases.append(self_param + other_param)
    temp_client = copy.deepcopy(self) # pointers??
    with torch.no_grad():
      i = 0
      for name, param in temp_client.model.named_parameters():
        if 'weight' in name:
          param.copy_(temp_weights[i])
          i += 1

        j = 0
        for name, param in temp_client.model.named_parameters():
          if 'bias' in name:
            param.copy_(temp_biases[j])
            j += 1
    # order?
    return temp_client

  def __mul__(self, multiplier: float|int) -> FL_client:
    assert isinstance(multiplier, float|int)
    temp_weights = []
    temp_biases = []
    for (name, self_param) in self.named_parameters():
      self_param.to(device)
      if 'weight' in name:
        temp_weights.append(self_param * multiplier)
      elif 'bias' in name:
        temp_biases.append(self_param * multiplier)

    temp_client = copy.deepcopy(self)
    #order?
    with torch.no_grad():
      i = 0
      for name, param in temp_client.model.named_parameters():
        if 'weight' in name:
          param.copy_(temp_weights[i])
          i += 1

      j = 0
      for name, param in temp_client.model.named_parameters():
        if 'bias' in name:
          param.copy_(temp_biases[j])
          j += 1
    return temp_client

  def __rmul__(self, multiplier: float|int) -> FL_client:
    return self.__mul__(multiplier)

  def __sub__(self, other: FL_client) -> FL_client:
    assert isinstance(other, FL_client)
    return self + (-other)



class FL_server():
  def __init__(self, K):
    self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(device)

    self.K = K
    self.training_rounds = 10

    self.loss_fn = nn.CrossEntropyLoss()
    self.clients = []
    for i in range(self.K):
      self.clients.append(FL_client()) # pointers?


  def forward(self, x):
    return self.model(x)

  def FedAvg(self, dataloaders_list, validate_dataloader, C):
    for round in range(self.training_rounds):
      m = max(C*self.K, 1)
      rand_set_clients = random.sample(range(self.K), m)
      client_model_sum = None
      m_t = 0
      for client_id in rand_set_clients:
        temp_client = self.clients[client_id]
        temp_dataset = dataloaders_list[client_id]

        print(f"----------- TRAIN LOOP FOR CLIENT {client_id}. -----------")
        temp_client.train_model(temp_dataset)
        n_k = len(temp_dataset)
        m_t += n_k
        # Easier than instantiating an empty model
        if client_model_sum is None:
          client_model_sum = n_k*temp_client
        else:
          client_model_sum = client_model_sum + n_k*temp_client

      client_model_sum = (1/m_t) * client_model_sum
      self.model = client_model_sum.model
      print(f"----------- VALIDATION FOR SERVER. -----------")
      self.validate(validate_dataloader)


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

    print(f"Validation Loss: {avg_loss:.4f}, Correct: {total_correct} out of {len(dataloader)}, Accuracy: {accuracy:.2%}")

    return avg_loss, total_correct, accuracy

