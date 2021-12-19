# coding: utf-8

import time

import torch.nn as nn
import torch.optim as optim
from models import MLP, CNN
from Trainer import Trainer
from datasets import get_datasets, get_dataloader

if __name__ == "__main__":
    dataset_name, dataset_fpath, batch_size = "cifar10", "~/.keras/datasets", 128
    trainset, testset = get_datasets(dataset_name, dataset_fpath)
    trainloader, testloader = get_dataloader(
        trainset, testset, num_workers=32, batch_size=batch_size, pin_memory=True
    )

    # model = MLP()
    model = CNN()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(model, optimizer, criterion, device="cuda:0")

    s1 = time.time()
    # 训练并验证模型
    trainer.train(trainloader, epochs=5)
    e1 = time.time()
    print("训练耗时: {}".format(e1 - s1))

    s2 = time.time()
    trainer.test(testloader)
    e2 = time.time()
    print("测试耗时: {}".format(e2 - s2))
