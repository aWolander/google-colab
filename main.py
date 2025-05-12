'''
TODO: 
Batch normalization, probably. I think I read somewhere that the facebook model needs this
Version control and checkpointing
Testing and hyperparameter tuning
gradient mask TaLoS thing. Also expanding this to FL.
'''

K=5
FL = FL_server(K)
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
CIFAR100 = datasets.CIFAR100(
  root="data",
  download=True,
  transform=ToTensor()
)
train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(CIFAR100, [0.7,0.15,0.15])


split_dataset = split_data_iid(train_dataset, K)

train_dataloader = [DataLoader(subset, batch_size=100) for subset in split_dataset]

test_dataloader = DataLoader(test_dataset, batch_size=100)
validate_dataloader = DataLoader(validate_dataset, batch_size=100)

FL.FedAvg(train_dataloader, validate_dataloader, 1)
FL.test_model(test_dataloader)
