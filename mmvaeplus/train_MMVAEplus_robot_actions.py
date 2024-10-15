# Train MMVAEplus on Robot Actions dataset
import os
import shutil
import argparse
import sys
import json
from pathlib import Path
import numpy as np
import torch
from torch import optim
import models
import objectives as objectives
from utils import Logger, Timer, save_model_light, move_to_device
from utils import unpack_data_robot_actions as unpack_data
import wandb
import pickle

parser = argparse.ArgumentParser(description='MMVAEplus Robot Actions')
parser.add_argument('--experiment', type=str, default='', metavar='E',
                    help='experiment name')
parser.add_argument('--obj', type=str, default='dreg', choices=['elbo', 'dreg'],
                    help='objective to use')
parser.add_argument('--K', type=int, default=10,
                    help='number of samples when resampling in the latent space')
parser.add_argument('--batch-size', type=int, default=16384, metavar='N',
                    help='batch size for data')
parser.add_argument('--epochs', type=int, default=50, metavar='E',
                    help='number of epochs to train')
parser.add_argument('--latent-dim-w', type=int, default=32, metavar='L',
                    help='latent dimensionality (default: 20)')
parser.add_argument('--latent-dim-z', type=int, default=64, metavar='L',
                    help='latent dimensionality (default: 20)')
parser.add_argument('--print-freq', type=int, default=50, metavar='f',
                    help='frequency with which to print stats (default: 0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA use')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed')
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--llik_scaling_action', type=float, default=5.0,
                    help='likelihood scaling factor for actions')
parser.add_argument('--datadir', type=str, default='./data',
                    help='Directory where data is stored')
parser.add_argument('--outputdir', type=str, default='../outputs',
                    help='Output directory')
parser.add_argument('--priorposterior', type=str, default='Normal', choices=['Normal', 'Laplace'],
                    help='distribution choice for prior and posterior')
parser.add_argument('--faive_enc_hidden_dim', type=int, default=32,
                    help='encoder hidden dimension for faive')
parser.add_argument('--faive_dec_hidden_dim', type=int, default=32,
                    help='decoder hidden dimension for faive')
parser.add_argument('--mano_enc_hidden_dim', type=int, default=36,
                    help='encoder hidden dimension for mano')
parser.add_argument('--mano_dec_hidden_dim', type=int, default=36,
                    help='decoder hidden dimension for mano')
parser.add_argument('--gripper_enc_hidden_dim', type=int, default=4,
                    help='encoder hidden dimension for gripper')
parser.add_argument('--gripper_dec_hidden_dim', type=int, default=4,
                    help='decoder hidden dimension for gripper')
parser.add_argument('--num_hidden_layers', type=int, default=2,
                    help='number of hidden layers for MLPs')
parser.add_argument('--cuda-device-id', type=int, default=0,
                    help='CUDA device ID')

# args
args = parser.parse_args()
args.latent_dim_u = args.latent_dim_w + args.latent_dim_z

# Random seed
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# CUDA stuff
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device(f"cuda:{args.cuda_device_id}" if args.cuda else "cpu")
print(device)

modelC = getattr(models, 'MMVAEplus_RobotActions')
model = modelC(args).to(device)

# Set experiment name if not set
if not args.experiment:
    args.experiment = model.modelName

# WandB
wandb.login()

wandb.init(
    project=args.experiment,
    config=args,
    # mode="disabled" 
)

runId = wandb.run.name
# Set up run path
experiment_dir = Path(os.path.join(args.outputdir, args.experiment, "checkpoints"))
experiment_dir.mkdir(parents=True, exist_ok=True)
runPath = os.path.join(str(experiment_dir), runId)
if os.path.exists(runPath):
    shutil.rmtree(runPath)
os.makedirs(runPath)
sys.stdout = Logger('{}/run.log'.format(runPath))
print('Expt:', runPath)
print('RunID:', runId)

NUM_VAES = len(model.vaes)

# Create path where to temporarily save data for evaluation
eval_path = os.path.join(args.datadir, 'eval_Robot_Actions' + (runPath.rsplit('/')[-1]))
datadirRobotActions = os.path.join(args.datadir, "Robot_Actions")

# save args to run
with open('{}/args.json'.format(runPath), 'w') as fp:
    json.dump(args.__dict__, fp)
torch.save(args, '{}/args.rar'.format(runPath))


# preparation for training
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=1e-3, amsgrad=True)

# Load Robot Actions dataset
train_loader, test_loader, dataset_stats = model.getDataLoaders(args.datadir, args.batch_size, device=device)

# save dataset stats to run
with open(f'{runPath}/dataset_stats.pkl', 'wb') as fp:
    pickle.dump(dataset_stats, fp)

objective = getattr(objectives,
                    ('m_' if hasattr(model, 'vaes') else '')
                    + args.obj)
t_objective = objective

def train(epoch):
    model.train()
    b_loss = 0
    for i, dataT in enumerate(train_loader):
        data = unpack_data(dataT, device=device)
        optimizer.zero_grad()
        loss = -objective(model, data, K=args.K)
        loss.backward()
        optimizer.step()
        b_loss += loss.item()
        if args.print_freq > 0 and i % args.print_freq == 0:
            print("iteration {:04d}: loss: {:6.3f}".format(i, loss.item() / args.batch_size))
    # Epoch loss
    epoch_loss = b_loss / len(train_loader.dataset)
    wandb.log({"Loss/train": epoch_loss}, step=epoch)
    print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, epoch_loss))

def test(epoch):
    b_loss = 0
    with torch.no_grad():
        for i, dataT in enumerate(test_loader):
            data = unpack_data(dataT, device=device)
            loss = -t_objective(model, data, K=args.K, test=True)
            b_loss += loss.item()
            if i == 0 and epoch % 1 == 0:
                # Compute cross-generations
                cg_actions = model.self_and_cross_modal_generation(data)
                # save samples to file
                with open(f'{runPath}/conditional_samples_{epoch}.pkl', 'wb') as f:
                    pickle.dump(move_to_device(cg_actions, device='cpu'), f)
                # for i in range(NUM_VAES):
                #     for j in range(NUM_VAES):
                #         wandb.log({'Cross_Generation/m{}/m{}'.format(i, j): wandb.Histogram(cg_actions[i][j])}, step=epoch)
    # Epoch test loss
    epoch_loss = b_loss / len(test_loader.dataset)
    wandb.log({"Loss/test": epoch_loss}, step=epoch)
    print('====>             Test loss: {:.4f}'.format(epoch_loss))
    return epoch_loss

if __name__ == '__main__':
    best_val_loss = float('inf')
    with Timer('MMVAEplus') as t:
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            if epoch % 1 == 0:
                val_loss = test(epoch)
                gen_samples = model.generate_unconditional(N=100)
                # save samples to file
                with open(f'{runPath}/unconditional_samples_{epoch}.pkl', 'wb') as f:
                    pickle.dump(move_to_device(gen_samples, device='cpu'), f)
                # for j in range(NUM_VAES):
                    # wandb.log({'Generations/m{}'.format(j): wandb.Histogram(gen_samples[j])}, step=epoch)
            if epoch % 10 == 0:
                save_model_light(model, runPath + '/model_'+str(epoch)+'.rar')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model_light(model, runPath + '/model_best.rar')

 