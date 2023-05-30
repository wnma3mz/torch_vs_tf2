# coding: utf-8

import time
import torch
import torch.nn as nn
import torch.optim as optim
from models import MLP, CNN
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet152, ResNet152_Weights
from Trainer import Trainer
from datasets import get_datasets, get_dataloader
from fixed_proj import Proj, HadamardProj
from imprint import imprint, weight_norm

import numpy as np
import random

def setup_seed(seed):
    """设置随机种子，以便于复现实验"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # tf.random.set_seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)

class MyTrainer(Trainer):
    def train(self, data_loader, epochs=1):
        self.model.train()
        self.is_train = True
        for _ in range(1, epochs + 1):
            with torch.enable_grad():
                loss, accuracy = self._iteration(data_loader)
            
            self.model.fc.weight.data = weight_norm(self.model.fc.weight.data)
            self.history_loss.append(loss)
            self.history_accuracy.append(accuracy)
        return loss, accuracy


if __name__ == "__main__":
    dataset_name, dataset_fpath, batch_size = "few_cifar10", "~/.keras/datasets", 128
    trainset, testset = get_datasets(dataset_name, dataset_fpath)
    trainloader, testloader = get_dataloader(
        trainset, testset, num_workers=32, batch_size=batch_size, pin_memory=True
    )

    # model = MLP()
    # model = CNN()
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # model = resnet34(weights=ResNet34_Weights.DEFAULT)
    # model = resnet152(weights=ResNet152_Weights.DEFAULT)
    # model.fc = Proj(model.fc.in_features, 10)
    
    # model.fc = HadamardProj(model.fc.in_features, 10)
    # model.fc.out_features = 10


    model = imprint(trainloader, model, 10, "cuda:0", False, model.fc.in_features)
    # model.fc = nn.Identity()
    # for name, params in model.named_parameters():
    #     if "fc" not in name:
    #         params.requires_grad = False

    # optimizer = optim.Adam(model.parameters(), lr=1e-2)    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, device="cuda:0")

    s2 = time.time()
    trainer.test(testloader)
    e2 = time.time()
    print("测试耗时: {}".format(e2 - s2))


    # s1 = time.time()
    # # 训练并验证模型
    # trainer.train(trainloader, epochs=1)
    # e1 = time.time()
    # print("训练耗时: {}".format(e1 - s1))

    # s2 = time.time()
    # trainer.test(testloader)
    # e2 = time.time()
    # print("测试耗时: {}".format(e2 - s2))
