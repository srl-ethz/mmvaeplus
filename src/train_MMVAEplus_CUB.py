# Train MMVAEplus on CUB Image-Captions dataset
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
from utils import Logger, Timer, save_model_light
from utils import unpack_data_cubIC as unpack_data
import wandb
from test_functions_CUB import calculate_inception_features_for_gen_evaluation, calculate_fid

parser = argparse.ArgumentParser(description='MMVAEplus Imgae-Captions')
parser.add_argument('--experiment', type=str, default='', metavar='E',
                    help='experiment name')
parser.add_argument('--obj', type=str, default='dreg', choices=['elbo', 'dreg'],
                    help='objective to use')
parser.add_argument('--K', type=int, default=10,
                    help='number of samples when resampling in the latent space')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='batch size for data')
parser.add_argument('--epochs', type=int, default=300, metavar='E',
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
parser.add_argument('--llik_scaling_sent', type=float, default=5.0,
                    help='likelihood scaling factor sentences')
parser.add_argument('--datadir', type=str, default='/local/home/palumboe/data',
                    help=' Directory where data is stored and samples used for FID calculation are saved')
parser.add_argument('--outputdir', type=str, default='../outputs',
                    help='Output directory')
parser.add_argument('--inception_path', type=str, default='/local/home/palumboe/data/pt_inception-2015-12-05-6726825d.pth',
                    help='Path to inception module for FID calculation')
parser.add_argument('--priorposterior', type=str, default='Normal', choices=['Normal', 'Laplace'],
                    help='distribution choice for prior and posterior')


# args
args = parser.parse_args()
args.latent_dim_u = args.latent_dim_w + args.latent_dim_z


# Random seed
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# CUDA stuff
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
print(device)

modelC = getattr(models, 'MMVAEplus_CUB_Image_Captions')
model = modelC(args).to(device)

# Set experiment name if not set
if not args.experiment:
    args.experiment = model.modelName

# Set up run path
runId = str(args.latent_dim_w) + '_' + str(args.latent_dim_z) + '_' + str(args.beta) + '_' + str(args.seed)
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

# Creat path where to temporarily save images to compute FID scores
fid_path = os.path.join(args.datadir, 'fids_CUB_Image_Captions' + (runPath.rsplit('/')[-1]))
datadirCUB = os.path.join(args.datadir, "CUB_Image_Captions")

# save args to run
with open('{}/args.json'.format(runPath), 'w') as fp:
    json.dump(args.__dict__, fp)
# -- also save object because we want to recover these for other things
torch.save(args, '{}/args.rar'.format(runPath))

# WandB

wandb.login()

wandb.init(
    # Set the project where this run will be logged
    project=args.experiment,
    # Track hyperparameters and run metadata
    config=args,
    # Run name
    name=runId
)


# preparation for training
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=1e-3, amsgrad=True)


# Load CUB Image-Captions
train_loader, test_loader = model.getDataLoaders(args.batch_size,  device=device)
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
    # model.eval()
    b_loss = 0
    with torch.no_grad():
        for i, dataT in enumerate(test_loader):
            data = unpack_data(dataT, device=device)
            loss = -t_objective(model, data, K=args.K, test=True)
            b_loss += loss.item()
            if i == 0  and epoch % 1 == 0:
                # Compute cross-generations
                cg_imgs = model.self_and_cross_modal_generation(data, 10, 10)
                for i in range(NUM_VAES):
                    for j in range(NUM_VAES):
                        wandb.log({'Cross_Generation/m{}/m{}'.format(i, j): wandb.Image(cg_imgs[i][j])}, step=epoch)
    # Epoch test loss
    epoch_loss = b_loss / len(test_loader.dataset)
    wandb.log({"Loss/test": epoch_loss}, step=epoch)
    print('====>             Test loss: {:.4f}'.format(epoch_loss))

def calculate_fid_routine(datadirCUB, fid_path, num_fid_samples, epoch):
    """ Calculate FID scores for unconditional and conditional generation """
    total_cond = 0
    # Create new directories for conditional FIDs
    for j in [0]:
        if os.path.exists(os.path.join(fid_path, 'test', 'm{}'.format(j))):
            shutil.rmtree(os.path.join(fid_path, 'test', 'm{}'.format(j)))
            os.makedirs(os.path.join(fid_path, 'test', 'm{}'.format(j)))
        else:
            os.makedirs(os.path.join(fid_path, 'test', 'm{}'.format(j)))
        if os.path.exists(os.path.join(fid_path, 'random', 'm{}'.format(j))):
            shutil.rmtree(os.path.join(fid_path, 'random', 'm{}'.format(j)))
            os.makedirs(os.path.join(fid_path, 'random', 'm{}'.format(j)))
        else:
            os.makedirs(os.path.join(fid_path, 'random', 'm{}'.format(j)))
        for i in [0, 1]:
            if os.path.exists(os.path.join(fid_path, 'm{}'.format(i), 'm{}'.format(j))):
                shutil.rmtree(os.path.join(fid_path, 'm{}'.format(i), 'm{}'.format(j)))
                os.makedirs(os.path.join(fid_path, 'm{}'.format(i), 'm{}'.format(j)))
            else:
                os.makedirs(os.path.join(fid_path, 'm{}'.format(i), 'm{}'.format(j)))
    with torch.no_grad():
        # Generate unconditional fid samples
        for tranche in range(num_fid_samples // 100):
            kwargs_uncond = {
                'savePath': fid_path,
                'tranche': tranche
            }
            model.generate_unconditional(N=100, coherence_calculation=False, fid_calculation=True, **kwargs_uncond)
        # Generate conditional fid samples
        for i, dataT in enumerate(test_loader):
            data = unpack_data(dataT, device=device)

            if total_cond < num_fid_samples:
                model.self_and_cross_modal_generation_for_fid_calculation(data, fid_path, i)
                model.save_test_samples_for_fid_calculation(data, fid_path, i)
                total_cond += data[0].size(0)
        calculate_inception_features_for_gen_evaluation(args.inception_path, device,
                                                        fid_path, datadirCUB)
        # FID calculation
        modality_target = 'm{}'.format(0)
        file_activations_real = os.path.join(fid_path, 'test',
                                             'real_activations_{}.npy'.format(modality_target))
        feats_real = np.load(file_activations_real)
        file_activations_randgen = os.path.join(fid_path, 'random',
                                                    modality_target + '_activations.npy')
        feats_randgen = np.load(file_activations_randgen)
        fid_randval = calculate_fid(feats_real, feats_randgen)
        wandb.log({"FID/Random/{}".format(modality_target): fid_randval}, step=epoch)
        fid_condgen_target_list = []
        for modality_source in ['m{}'.format(m) for m in [0,1]]:
            file_activations_gen = os.path.join(fid_path, modality_source,
                                                modality_target + '_activations.npy')
            feats_gen = np.load(file_activations_gen)
            fid_val = calculate_fid(feats_real, feats_gen)
            wandb.log({"FID/{}/{}".format(modality_source, modality_target): fid_val}, step=epoch)
            fid_condgen_target_list.append(fid_val)
    # Clean up
    if os.path.exists(fid_path):
        shutil.rmtree(fid_path)
        os.makedirs(fid_path)

if __name__ == '__main__':
    with Timer('MMVAEplus') as t:
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            if epoch % 25 == 0:
                test(epoch)
                # model.eval()
                gen_samples = model.generate_unconditional(N=100, coherence_calculation=False, fid_calculation=False)
                for j in range(NUM_VAES):
                    wandb.log({'Generations/m{}'.format(j): wandb.Image(gen_samples[j])}, step=epoch)
                calculate_fid_routine(datadirCUB, fid_path, 10000, epoch)
            if epoch % 25 == 0:
                save_model_light(model, runPath + '/model_'+str(epoch)+'.rar')

