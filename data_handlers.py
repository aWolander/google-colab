import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import random
import numpy as np
# vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

# print(vits16)

def main():
    K= 20
    N = 20
    CIFAR100 = datasets.CIFAR100(
      root="data",
      download=True,
      transform=ToTensor()
    )
    train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(CIFAR100, [0.7,0.15,0.15])
    client_datasets = split_data_non_iid(train_dataset, K, N)
    test_split(client_datasets)


def create_label_indexing(dataset):
    label_index = {}
    for i in range(100):
        label_index.update({i:[]})
    for index, (_, label) in enumerate(dataset):
        label_index[label].append(index)
    return label_index

def split_data_non_iid(dataset, K, N_c = 100):
    index_split = [ [] for _ in range(K) ]
    label_index = create_label_indexing(dataset)
    split_availible_labels = {}
    lablels_exhausted = [False]*K

    # Randomly choose which labels each client has access to
    for i in range(K):
        split_availible_labels[i] = random.sample(range(100), N_c)

    # stops if no more datapoints are left to assign to any client
    while not all(lablels_exhausted):
        for client in range(K):

            if not split_availible_labels[client]:
                lablels_exhausted[client] = True

            for label in split_availible_labels[client]:
                # if no indices are left for a label, remove the label
                if not label_index[label]:
                    split_availible_labels[client].remove(label)
                    continue

                index_split[client].append(label_index[label].pop())

    for client_indices in index_split:
        random.shuffle(client_indices)

    return [Subset(dataset, indices) for indices in index_split]

def split(l, n):
    split_list = []
    for i in range(0, n):
        split_list.append(l[i::n])
    return split_list

def split_data_iid(dataset, K):
    index_split = [[] for x in range(K)]
    label_index = create_label_indexing(dataset)

    for label in range(0,100):
        split_indices = split(label_index[label],K)
        for client in range(0, K):
            index_split[client] += split_indices[client]

    for client_indices in index_split:
        random.shuffle(client_indices)
    return [Subset(dataset, indices) for indices in index_split]

def test_split(client_datasets):
    bottom = np.zeros(100)
    for client_id, client_dataset in enumerate(client_datasets):
        occurrences = np.zeros(100)
        for datapoint in client_dataset:
            label = datapoint[1]
            occurrences[label] += 1
        print(occurrences)
        plt.bar(range(100), occurrences, bottom=bottom, label=client_id)
        plt.xlabel("Class label")
        plt.ylabel("Number of samples")
        bottom += occurrences

    plt.show()

main()

# def test_split_index(dataset):
#     K= 2
#     N = 5
#     client_data_split = split_data_non_iid(dataset, K, N)
#     for client_data in client_data_split:
#         occurences = {}
#         for index in client_data:
#             occurences[dataset[index][1]] = occurences.setdefault(dataset[index][1], 0) + 1
#         print(occurences)
#         plt.hist(occurences, stacked=True, bins=100)
#     plt.show()

# def split_data_non_iid(dataset, K, N_c=20, seed=42):
#     # as far I can tell N_c does exactly nothing
#     random.seed(seed)
#     label_index = create_label_indexing(dataset)
#     client_indices = [[] for _ in range(K)]

#     class_to_clients = {label: set() for label in range(100)} # why sets?
#     for label in range(100):
#         selected_clients = random.sample(range(K), k=max(1, K // 5)) # why define variable k? why K // 5?
#         class_to_clients[label].update(selected_clients)

#     # why are these two for loops?
#     client_to_classes = {client: set() for client in range(K)} # why do you need both client_to_classes and class_to_clients?
#     for label in range(100):
#         for client in class_to_clients[label]:
#             if len(client_to_classes[client]) < N_c: 
#                 client_to_classes[client].add(label)
    
#     # what
#     all_labels = list(range(100)) # ???
#     for client in range(K):
#         while len(client_to_classes[client]) < N_c:
#             label = random.choice(all_labels)
#             client_to_classes[client].add(label) # is this dict even used?
#             class_to_clients[label].add(client)

#     for label, indices in label_index.items():
#         random.shuffle(indices) # why? this is done at the end
#         clients = list(class_to_clients[label])
#         num_clients = len(clients)

#         # im pretty sure this is just the split(l,n) function
#         split_size = len(indices) // num_clients 
#         for i, client in enumerate(clients):
#             start = i * split_size
#             end = (i + 1) * split_size if i < num_clients - 1 else len(indices)
#             client_indices[client].extend(indices[start:end])

#     for indices in client_indices:
#         random.shuffle(indices)

#     return [Subset(dataset, indices) for indices in client_indices]