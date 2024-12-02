import argparse
import os, sys
from os import path
import time
import copy
import torch
from torch import nn
import numpy as np
import random
import shutil
import torchvision.models as models
import torch.nn.functional as F
import pickle

import warnings
warnings.filterwarnings("ignore")


def seed_torch(seed=1029):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(999)
from utils_model_load import *
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as TF
# from gan_training.distributions import get_ydist, get_zdist
# from gan_training.config import (
#     load_config, build_models,)
from EWC import Net
import scipy.io as sio
ce_loss = nn.CrossEntropyLoss()
# main_path = './code_GAN_Memory/'


data_transforms = {
    'train1': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def select_task_path(task_id, is_test=False):
    if not is_test:
        if task_id == 0:
            train_path = train_path_all + '/train/fish/'
        elif task_id == 1:
            train_path = train_path_all + '/train/bird/'
        elif task_id == 2:
            train_path = train_path_all + '/train/snake/'
        elif task_id == 3:
            train_path = train_path_all + '/train/dog/'
        return train_path
    elif is_test:
        if task_id == 0:
            train_path = train_path_all + '/test1/fish/'
        elif task_id == 1:
            train_path = train_path_all + '/test1/bird/'
        elif task_id == 2:
            train_path = train_path_all + '/test1/snake/'
        elif task_id == 3:
            train_path = train_path_all + '/test1/dog/'
        return train_path



# -------------------------------------------------------------
# -------------------------------------------------------------
lamda_replay = 1 # 5
lamda_EWC = 1e4 # 500
batch_size= 36
N_task = 4
N_epoch = 1
N_labels = N_task * 6
do_method = 'StyleCL'#CAMGAN' #'MeRGAN'   # StyleCL  'joint' 'EWC'
# -------------------------------------------------------------
# -------------------------------------------------------------
print("do method ",do_method)



if do_method == 'CAMGAN':
    with open('results/cond/fish/camgan/00007-fish-cond-mirror-auto1-gamma4-resumecustom/network-snapshot-001440.pkl','rb') as modelFile:
        model = pickle.load(modelFile)
    G0 = model['G']
    with open('results/cond/fish/camgan/00007-fish-cond-mirror-auto1-gamma4-resumecustom/network-snapshot-001440.pkl','rb') as modelFile:
        model = pickle.load(modelFile)
    G1 = model['G']
    with open('results/cond/fish/camgan/00007-fish-cond-mirror-auto1-gamma4-resumecustom/network-snapshot-001440.pkl','rb') as modelFile:
        model = pickle.load(modelFile)
    G2 = model['G']
    with open('results/cond/fish/camgan/00007-fish-cond-mirror-auto1-gamma4-resumecustom/network-snapshot-001440.pkl','rb') as modelFile:
        model = pickle.load(modelFile)
    G3 = model['G']
    

elif do_method == 'MeRGAN':
    with open('pretrained_models/mergan_conditional_models/fish_mergan_cond.pkl','rb') as modelFile:
        model = pickle.load(modelFile)
    G0 = model['G']
    with open('pretrained_models/mergan_conditional_models/bird_mergan_cond.pkl','rb') as modelFile:
        model = pickle.load(modelFile)
    G1 = model['G']
    with open('pretrained_models/mergan_conditional_models/snake_mergan_cond.pkl','rb') as modelFile:
        model = pickle.load(modelFile)
    G2 = model['G']
    with open('pretrained_models/mergan_conditional_models/dogs_mergan_cond.pkl','rb') as modelFile:
        model = pickle.load(modelFile)
    G3 = model['G']

elif do_method == 'StyleCL':
    with open('pretrained_models/lifelong_classification/fish/network-snapshot-001400.pkl','rb') as modelFile:
        model = pickle.load(modelFile)
    G0 = model['G'].synthesis
    with open('pretrained_models/lifelong_classification/fish/network-snapshot-001400.pkl','rb') as modelFile:
        adap = pickle.load(modelFile)
    adaptor0 = adap['adaptor']

    with open('pretrained_models/lifelong_classification/birds/network-snapshot-002400.pkl','rb') as modelFile:
        model = pickle.load(modelFile)
    G1 = model['G'].synthesis
    with open('pretrained_models/lifelong_classification/birds/network-snapshot-002400.pkl','rb') as modelFile:
        adap = pickle.load(modelFile)
    adaptor1 = adap['adaptor']

    with open('pretrained_models/lifelong_classification/snake/network-snapshot-003600.pkl','rb') as modelFile:
        model = pickle.load(modelFile)
    G2 = model['G'].synthesis
    with open('pretrained_models/lifelong_classification/snake/network-snapshot-003600.pkl','rb') as modelFile:
        adap = pickle.load(modelFile)
    adaptor2 = adap['adaptor']

    with open('pretrained_models/lifelong_classification/dogs/network-snapshot-007600.pkl','rb') as modelFile:
        model = pickle.load(modelFile)
    G3 = model['G'].synthesis
    with open('pretrained_models/lifelong_classification/dogs/network-snapshot-007600.pkl','rb') as modelFile:
        adap = pickle.load(modelFile)
    adaptor3 = adap['adaptor']


def evalu_my(model, test_loader,test_task=-1 ):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        if test_task>=0:
            target = target+ test_task*6
        output = model(data)
        test_loss += ce_loss(output, target).data # sum up batch loss
        _, pred = output.data.max(1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    test_acc = 100.0 * correct.float() / len(test_loader.dataset)
    print('\nTest set task {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.5f}%)\n'.format(
        test_task, test_loss, correct, len(test_loader.dataset),
        test_acc))
    return test_acc

def generator(y,i_t,task_id,method):

    if method == "MeRGAN":
        xx = task_id
    else:
        xx = i_t

    if xx==0 :
        G = G0
    elif xx==1 :
        G = G1
    elif xx==2 :
        G = G2
    elif xx==3 :
        G = G3
    
    
    G.to(device)
    if method == 'CAMGAN':
        grid_z = torch.randn([y.shape[0], G.z_dim], device=device)
        grid_c = torch.tensor(np.array([np.array([np.eye(6)[int(t)]]) for t in y])).reshape((y.shape[0],6)).to(device)
        images = G(z=grid_z, c=grid_c , noise_mode='const')

    elif method == "MeRGAN":
        grid_z = torch.randn([y.shape[0], G.z_dim], device=device)
        grid_labels = torch.tensor(np.array([np.array([np.eye(6)[int(t)]]) for t in y])).reshape((y.shape[0],6)).to(device)
        grid_tasks = torch.tensor(np.array([np.array([np.eye(5)[int(i_t)]]) for t in range(len(y))])).reshape((y.shape[0],5)).to(device)
        grid_c = torch.cat((grid_tasks,grid_labels),1)
        images = G(z=grid_z, c=grid_c , noise_mode='const')

    elif method == 'StyleCL':
        if i_t==0 :
            adaptor = adaptor0
        elif i_t==1 :
            adaptor = adaptor1
        elif i_t==2 :
            adaptor = adaptor2
        elif i_t==3 :
            adaptor = adaptor3
        # print(adaptor)
        adaptor.to(device)
        grid_z = torch.randn(y.shape[0], 14,16).to(device)
        grid_c = torch.tensor(np.array([np.array([np.eye(6)[int(t)]]) for t in y])).reshape((y.shape[0],6)).to(device)
        w_s = adaptor(grid_z, grid_c)
        images = G(w_s)

    return images


train_path_all = 'conditional/data/'


test_path = 'conditional/data/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device = ",device)
# config_path = './configs/ImageNet_classify_53.yaml'
# config = load_config(config_path, 'configs/default.yaml')


test_dataset = datasets.ImageFolder('conditional/data/joint_test1', data_transforms['test'])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=0)
class_names = test_dataset.classes
# print(class_names)


classify_my = Net(nlabels=N_labels, device=device).to(device)
c_optimizer = torch.optim.Adam(classify_my.params, lr=1e-4)


acc_all_i = [[],[],[],[],[],[]]
acc_all = []
save_path = 'conditional/quant_results/'

for n, param in classify_my.feat.named_parameters():
    param.requires_grad = True


for task_id in range(N_task):
    print("--------Task number--------- " , task_id+1)
    # prepare dataloader
    n_c = batch_size
    n_p = batch_size

    train_path = select_task_path(task_id)
    train_dataset = datasets.ImageFolder(os.path.join(train_path), data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=n_c,
                                                shuffle=True, num_workers=0)
    
    if do_method== "joint" :
        IT = 0
        acc_task = []
        acc_task_i = []
        for epoch in range(N_epoch):
            for (x_cur, y_cur) in train_loader:
                y_cur = task_id*6 + y_cur
                classify_my.train()
                c_optimizer.zero_grad()
                x_cur, y_cur = x_cur.to(device), y_cur.to(device)
                logids_cur = classify_my(x_cur)
                loss_cur = ce_loss(logids_cur, y_cur)
                loss_cur.backward()
                if task_id > 0:
                    with torch.no_grad():
                        x_replay = []
                        y_replay = []
                        for i_t in range(task_id):
                            train_path_0 = select_task_path(i_t)
                            train_dataset_0 = datasets.ImageFolder(os.path.join(train_path_0),
                                                                 data_transforms['train'])
                            train_loader_0 = torch.utils.data.DataLoader(train_dataset_0, batch_size=n_c,
                                                                       shuffle=True, num_workers=0)
                            iter_0 = iter(train_loader_0)
                            x_replay0,y_replay0 = iter_0.__next__()
                            x_replay.append(x_replay0.to(device))
                            y_replay.append(6 * i_t + y_replay0.to(device))

                        x_replay = torch.cat(x_replay)
                        y_replay = torch.cat(y_replay)

                    logits_replay = classify_my(x_replay.detach())
                    loss_replay = ce_loss(logits_replay, y_replay)
                    (lamda_replay * loss_replay).backward()
                c_optimizer.step()
                IT +=1
                if IT%50==0:
                    print(IT)
                    with torch.no_grad():
                        test_acc_i=0.0
                        for i_t in range(task_id+1):
                            test_path_i = select_task_path(i_t, is_test=True)
                            test_dataset_i = datasets.ImageFolder(os.path.join(test_path_i), data_transforms['test'])
                            test_loader_i = torch.utils.data.DataLoader(test_dataset_i, batch_size=batch_size,
                                                                        shuffle=False, num_workers=0)

                            test_acc_i = evalu_my(classify_my, test_loader_i, test_task=i_t)
                            acc_all_i[i_t].append(test_acc_i.data.cpu())
                        test_acc = evalu_my(classify_my, test_loader)
                        acc_all.append(test_acc.data.cpu())
                        print('\nTest set task {}/ epoch {}: Accuracy: ({:.5f}% / {:.5f}%) LAMBDA: ({:.5f}%)\n'.format(
                            task_id, epoch, test_acc_i, test_acc, lamda_replay))

        if task_id>=1:
            lamda_replay = lamda_replay * 0.9
            
    elif do_method == 'EWC':
        fisher_estimation_sample_size = batch_size * 40
        IT = 0
        acc_task = []
        acc_task_i = []
        train_loader_fisher = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=0)
        for epoch in range(N_epoch):
            for (x_cur, y_cur) in train_loader:
                y_cur = task_id * 6 + y_cur
                classify_my.train()
                x_cur, y_cur = x_cur.to(device), y_cur.to(device)
                c_optimizer.zero_grad()
                output = classify_my(x_cur)
                objective_loss = ce_loss(output, y_cur)
                # Manual
                ewc_loss = classify_my.ewc_loss(task_id=task_id, lamda=lamda_EWC)
                loss = objective_loss + ewc_loss
                # print('loss ================= ', loss)
                loss.backward()
                c_optimizer.step()

                IT += 1
                if IT % 50 == 0:
                    with torch.no_grad():
                        test_acc_i = 0.0
                        for i_t in range(task_id + 1):
                            test_path_i = select_task_path(i_t, is_test=True)
                            test_dataset_i = datasets.ImageFolder(os.path.join(test_path_i), data_transforms['test'])
                            test_loader_i = torch.utils.data.DataLoader(test_dataset_i, batch_size=batch_size,
                                                                        shuffle=False, num_workers=0)
                            test_acc_i = evalu_my(classify_my, test_loader_i, test_task=i_t)
                            acc_all_i[i_t].append(test_acc_i.data.cpu())

                        test_acc = evalu_my(classify_my, test_loader)
                        acc_all.append(test_acc.data.cpu())
                        print('\nTest set task {}/ epoch {}: Accuracy: ({:.5f}% / {:.5f}%)\n'.format(
                            task_id, epoch, test_acc_i, test_acc))
        # Get fisher inf of parameters and consolidate it in the net
        classify_my.estimate_fisher(train_loader_fisher, fisher_estimation_sample_size, batch_size=batch_size, task_id=task_id)

    else:
        classifier_old = copy.deepcopy(classify_my).eval()
        IT = 0
        acc_task = []
        acc_task_i = []
        for epoch in range(N_epoch):
            print("epoch " , epoch)
            for (x_cur, y_cur) in train_loader:
                y_0 = y_cur.clone()
                y_cur = task_id*6 + y_cur
                classify_my.train()
                x_cur, y_cur = x_cur.to(device), y_cur.to(device)
                c_optimizer.zero_grad()

                x_replay = []
                y_replay = []
                x_replay.append(x_cur)
                y_replay.append(y_cur)
                if task_id > 0:
                    with torch.no_grad():
                        if do_method=='CAMGAN' or 'MeRGAN' or 'StyleCL':
                            # y_0 = y_cur.detach().clone()
                            for i_t in range(task_id):
                                y_replay0 = (y_0 + i_t * 6).to(device)
                                
                                
                                x_replay0 = generator(y_0 , i_t,task_id=task_id,method=do_method )

                                x_replay0 = F.interpolate(x_replay0, 224, mode='bilinear')
                                mu_0 = torch.tensor([0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(
                                    2).unsqueeze(3)
                                st_0 = torch.tensor([0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(
                                    2).unsqueeze(3)
                                x_replay0 = ((x_replay0 + 1.0) / 2.0 - mu_0) / st_0

                                x_replay.append(x_replay0)
                                y_replay.append(y_replay0)

                x_replay = torch.cat(x_replay)
                y_replay = torch.cat(y_replay)

                logits_S = classify_my(x_replay.detach())
                loss_replay = ce_loss(logits_S, y_replay)
                loss_replay.backward()
                c_optimizer.step()
                IT +=1
                
                if IT%50==0:
                    print("IT" , IT)
                    with torch.no_grad():
                        test_acc_i=0.0
                        for i_t in range(task_id+1):
                            test_path_i = select_task_path(i_t, is_test=True)
                            test_dataset_i = datasets.ImageFolder(os.path.join(test_path_i), data_transforms['test'])
                            test_loader_i = torch.utils.data.DataLoader(test_dataset_i, batch_size=batch_size,
                                                                        shuffle=False, num_workers=0)

                            test_acc_i = evalu_my(classify_my, test_loader_i, test_task=i_t)
                            acc_all_i[i_t].append(test_acc_i.data.cpu())
                        test_acc = evalu_my(classify_my, test_loader)
                        acc_all.append(test_acc.data.cpu())
                        print('\nTest set task {}/ epoch {}: Accuracy: ({:.5f}% / {:.5f}%) )\n'.format(
                            task_id, epoch, test_acc_i, test_acc))
                        # print('\nTest set task {}/ epoch {}: Accuracy: ({:.5f}% / {:.5f}%) LAMBDA: ({:.5f}%)\n'.format(
                        #     task_id, epoch, test_acc_i, lamda_replay))
                        # print('\nTest set task {}/ epoch {}: Accuracy: {:.5f}% \n'.format(task_id, epoch, test_acc_i))


ACC_all_0 = np.stack(acc_all_i[0])
ACC_all_1 = np.stack(acc_all_i[1])
ACC_all_2 = np.stack(acc_all_i[2])
ACC_all_3 = np.stack(acc_all_i[3])
# ACC_all_4 = np.stack(acc_all_i[4])
# ACC_all_5 = np.stack(acc_all_i[5])
ACC_all = np.stack(acc_all)

sio.savemat(save_path + do_method + '_quant_results.mat', {'ACC_all_0': ACC_all_0,
                                                               'ACC_all_1': ACC_all_1,
                                                               'ACC_all_2': ACC_all_2,
                                                               'ACC_all_3': ACC_all_3,
                                                            #    'ACC_all_4': ACC_all_4,
                                                            #    'ACC_all_5': ACC_all_5,
                                                               'ACC_all': ACC_all,
                                                               })








