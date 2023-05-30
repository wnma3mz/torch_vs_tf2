# coding: utf-8

import time
import torch
import torch.nn as nn
import torch.optim as optim
from models import MLP, CNN
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet152, ResNet152_Weights, vit_b_16, ViT_B_16_Weights
from Trainer import Trainer
from datasets import get_datasets, get_dataloader
from fixed_proj import Proj, HadamardProj
from imprint import imprint, weight_norm

import numpy as np
import random
from functools import partial
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

    num_class = 10
    dataset_name, dataset_fpath, batch_size = f"few_cifar{num_class}", "~/.keras/datasets", 16
    # dataset_name, dataset_fpath, batch_size = f"cifar{num_class}", "~/.keras/datasets", 16
    trainset, testset = get_datasets(dataset_name, dataset_fpath)
    trainloader, testloader = get_dataloader(
        trainset, testset, num_workers=4, batch_size=batch_size, pin_memory=False
    )
    # assert 1==2

    # model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    # model = imprint(trainloader, model, num_class, "cuda:0", False, 768)
    
    # model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # model = resnet34(weights=ResNet34_Weights.DEFAULT)
    model = resnet152(weights=ResNet152_Weights.DEFAULT)
    model.fc.out_features = num_class
    # model = imprint(trainloader, model, num_class, "cuda:0", False, model.fc.in_features)
    # model.fc = Proj(model.fc.in_features, 10)
    
    # model.fc = HadamardProj(model.fc.in_features, 10)
    # model.fc.out_features = 10

    # model.fc = nn.Identity()
    for name, params in model.named_parameters():
        if "fc" not in name:
            params.requires_grad = False

    # optimizer = optim.Adam(model.parameters(), lr=1e-2)    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, device="cuda:0")

    s2 = time.time()
    loss, accuracy = trainer.test(testloader)
    e2 = time.time()
    print("测试耗时: {}".format(e2 - s2), loss, accuracy)

    # s1 = time.time()
    # loss, accuracy = trainer.train(trainloader, epochs=10)
    # e1 = time.time()
    # print("训练耗时: {}".format(e1 - s1), loss, accuracy)

    # s2 = time.time()
    # loss, accuracy = trainer.test(testloader)
    # e2 = time.time()
    # print("测试耗时: {}".format(e2 - s2), loss, accuracy)
