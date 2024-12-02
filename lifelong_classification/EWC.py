import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable as V
from torch import autograd
import numpy as np
import torchvision.models as models

class Net(nn.Module):
    def __init__(self, nlabels=1000, device=None):
        super(Net, self).__init__()
        self.device = device


        self.feat = models.resnet18(pretrained=True)
        # self.feat = models.resnet18(pretrained=False)
        self.feat.fc = nn.Linear(512, nlabels)
        
        self.params = [param for n, param in self.feat.named_parameters() if 1]

        self.online=False
        self.gamma = 1.

    def forward(self, x):
        if x.shape[3]>224:
            h = F.interpolate(x, 224, mode='bilinear')
        else:
            h= x
        # h = F.interpolate(x, 224, mode='bilinear')
        h = self.feat(h)
        
        return h

    def estimate_fisher(self, dataset, sample_size, batch_size=32, task_id=0):
        # Get loglikelihoods from data
        est_fisher_info = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher_info[n] = p.detach().clone().zero_()

        mode = self.training
        self.eval()
        data_loader = dataset
        for index,(x,y) in enumerate(data_loader):
            #print(x.size(), y.size())
            # x = x.view(batch_size, -1)
            if index >= sample_size//batch_size:
                break

            x = x.to(self.device)
            y = y.to(self.device)

            loglikelihoods = F.log_softmax(self(x), dim=1)[range(batch_size), y.data].mean()
            # negloglikelihood = F.nll_loss(F.log_softmax(self(x), dim=1), y.data)
            self.zero_grad()
            loglikelihoods.backward()

            # Square gradients and keep running sum
            for n, p in self.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        est_fisher_info[n] += p.grad.detach() ** 2

        est_fisher_info = {n: p/index for n, p in est_fisher_info.items()}

        # Store new values in the network
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                # -mode (=MAP parameter estimate)
                self.register_buffer('{}_estimated_mean{}'.format(n, 9 if self.online else task_id),
                                     p.detach().clone())
                # -precision (approximated by diagonal Fisher Information matrix)
                if self.online and task_id>= 1:
                    existing_values = getattr(self, '{}_estimated_fisher9'.format(n))
                    est_fisher_info[n] += self.gamma * existing_values
                self.register_buffer(
                    '{}_estimated_fisher{}'.format(n, 9 if self.online else task_id),
                    est_fisher_info[n])
        # Set model back to its initial mode
        self.train(mode=mode)

    def ewc_loss(self, task_id=0, lamda=1e9):
        if task_id > 0:
            losses = []
            for task in range(0, task_id):
                for n, p in self.named_parameters():
                    # retrieve the consolidated mean and fisher information.
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        mean = getattr(self, '{}_estimated_mean{}'.format(n,9 if self.online else task))
                        fisher = getattr(self, '{}_estimated_fisher{}'.format(n,9 if self.online else task))
                        
                        fisher = self.gamma*fisher if self.online else fisher
                        
                        losses.append((fisher * (p-mean)**2).sum())
            return (lamda/2)*sum(losses)
        else:
            return torch.tensor(0., device=self.device)