import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, MNIST


def get_datasets(dataset_name, dataset_dir, split=None):
    trans_mnist = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    trans_emnist = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    trains_cifar10 = transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    )
    trains_cifar100 = transforms.Normalize(
        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    )
    trans_cifar_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            trains_cifar10,
        ]
    )
    trans_cifar_test = transforms.Compose([transforms.ToTensor(), trains_cifar10])

    if dataset_name == "mnist":
        trainset = MNIST(dataset_dir, train=True, download=False, transform=trans_mnist)
        testset = MNIST(dataset_dir, train=False, download=False, transform=trans_mnist)
    elif dataset_name == "cifar10":
        trainset = CIFAR10(
            dataset_dir, train=True, download=False, transform=trans_cifar_train
        )
        testset = CIFAR10(
            dataset_dir, train=False, download=False, transform=trans_cifar_test
        )
    elif dataset_name == "cifar100":
        trainset = CIFAR100(
            dataset_dir, train=True, download=False, transform=trans_cifar_train
        )
        testset = CIFAR100(
            dataset_dir, train=False, download=False, transform=trans_cifar_test
        )
    elif dataset_name == "emnist":
        trainset = EMNIST(
            dataset_dir, train=True, download=False, split=split, transform=trans_emnist
        )
        testset = EMNIST(
            dataset_dir,
            train=False,
            download=False,
            split=split,
            transform=trans_emnist,
        )
    else:
        raise NotImplementedError("This dataset is not currently supported")
    return trainset, testset


def get_dataloader(trainset, testset, batch_size, num_workers=0, pin_memory=False):
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return trainloader, testloader