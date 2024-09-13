# Train MMVAEplus model for Robot Actions dataset
import os
import argparse
import sys
import json
from pathlib import Path
import numpy as np
from torch import optim
import models
import objectives as objectives
from utils import Logger, Timer, save_model_light
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from statistics import mean
import wandb

# Argument parsing
parser = argparse.ArgumentParser(description='MMVAEplus Robot Actions Experiment')
parser.add_argument('--experiment', type=str, default='', metavar='E',
                    help='experiment name')
parser.add_argument('--obj', type=str, default='elbo', choices=['elbo', 'dreg'],
                    help='objective to use')
parser.add_argument('--K', type=int, default=1,
                    help='number of samples when resampling in the latent space')
parser.add_argument('--beta', type=float, default=1.0,
                    help='beta-VAE parameter')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='batch size for training (default: 64)')
parser.add_argument('--latent-dim-w', type=int, default=32,
                    help='dimension of latent space w')
parser.add_argument('--latent-dim-z', type=int, default=32,
                    help='dimension of latent space z')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--outputdir', type=str, default='./results',
                    help='output directory for results')
parser.add_argument('--datadir', type=str, default='./data',
                    help='directory where data is stored')
parser.add_argument('--priorposterior', type=str, default='Laplace', choices=['Normal', 'Laplace'],
                    help='distribution choice for prior and posterior')

# Args
args = parser.parse_args()
args.latent_dim_u = args.latent_dim_w + args.latent_dim_z

# Random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# CUDA setup
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# Get model class
modelC = getattr(models, 'RobotActions')
model = modelC(args).to(device)

# Set experiment name if not set
if not args.experiment:
    args.experiment = model.modelName

# Set up run path
runId = f"{args.latent_dim_w}_{args.latent_dim_z}_{args.beta}_{args.seed}"
experiment_dir = Path(os.path.join(args.outputdir, args.experiment, "checkpoints"))
experiment_dir.mkdir(parents=True, exist_ok=True)
runPath = os.path.join(str(experiment_dir), runId)
if os.path.exists(runPath):
    shutil.rmtree(runPath)
os.makedirs(runPath)
sys.stdout = Logger(f'{runPath}/run.log')
print('Expt:', runPath)
print('RunID:', runId)

NUM_VAES = len(model.vaes)

# Save args to run
with open(f'{runPath}/args.json', 'w') as fp:
    json.dump(args.__dict__, fp)
torch.save(args, f'{runPath}/args.rar')

# WandB setup
wandb.login()
wandb.init(
    project=args.experiment,
    config=args,
    name=runId
)

# Optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=1e-3, amsgrad=True)

# Data loaders
train_loader, test_loader = model.getDataSets(args.batch_size, device=device)

# Training function
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = [d.to(device) for d in data]
        optimizer.zero_grad()
        
        if args.obj == 'elbo':
            loss = -objectives.elbo(model, data, args.K, args.beta)
        elif args.obj == 'dreg':
            loss = -objectives.dreg(model, data, args.K, args.beta)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data[0])}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader):.4f}')
    wandb.log({"train_loss": train_loss / len(train_loader)}, step=epoch)

# Test function
def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = [d.to(device) for d in data]
            if args.obj == 'elbo':
                loss = -objectives.elbo(model, data, args.K, args.beta)
            elif args.obj == 'dreg':
                loss = -objectives.dreg(model, data, args.K, args.beta)
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    print(f'====> Test set loss: {test_loss:.4f}')
    wandb.log({"test_loss": test_loss}, step=epoch)

# Main training loop
if __name__ == '__main__':
    with Timer('MMVAEplus Robot Actions') as t:
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            test(epoch)
            
            if epoch % 10 == 0:
                # Save model checkpoint
                save_model_light(model, optimizer, epoch, runPath)
                
                # Generate samples
                gen_samples = model.generate_unconditional(N=10)
                for j in range(NUM_VAES):
                    wandb.log({f'Generations/m{j}': wandb.Histogram(gen_samples[j].cpu().numpy())}, step=epoch)

    print(f"Total time taken: {t.interval:.2f} seconds")
    wandb.finish()