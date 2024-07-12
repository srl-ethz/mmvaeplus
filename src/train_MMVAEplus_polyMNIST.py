# Train MMVAEplus model PolyMNIST dataset
import os
import argparse
import sys
import json
from pathlib import Path
import numpy as np
from torch import optim
import models
import objectives as objectives
from utils import Logger, Timer, save_model_light, NonLinearLatent_Classifier, get_10_polymnist_samples, ClfImg
from utils import unpack_data_PM as unpack_data
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from statistics import mean
from test_functions_polyMNIST import calculate_inception_features_for_gen_evaluation, calculate_fid,classify_latent_representations
import wandb
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Subset

# Argument to parse
parser = argparse.ArgumentParser(description='MMVAEplus PolyMNIST Experiment')
parser.add_argument('--experiment', type=str, default='', metavar='E',
                    help='experiment name')
parser.add_argument('--obj', type=str, default='elbo', choices=['elbo', 'dreg'],
                    help='objective to use')
parser.add_argument('--K', type=int, default=1,
                    help='number of samples when resampling in the latent space')
parser.add_argument('--llik_scaling', type=float, default=0.)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=250)
parser.add_argument('--latent-dim-w', type=int, default=32,
                    help='latent modality-specific dimensionality')
parser.add_argument('--latent-dim-z', type=int, default=32,
                    help='latent shared dimensionality')
parser.add_argument('--print-freq', type=int, default=50,
                    help='frequency with which to print stats')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA use')
parser.add_argument('--seed', type=int, default=2,
                    help='random seed')
parser.add_argument('--beta', type=float, default=2.5,
                    help='beta hyperparameter in VAE objective')
parser.add_argument('--datadir', type=str, default='/local/home/palumboe/data',
                    help=' Directory where data is stored and samples used for FID calculation are saved')
parser.add_argument('--outputdir', type=str, default='../outputs',
                    help='Output directory')
parser.add_argument('--inception_path', type=str, default='/local/home/palumboe/data/pt_inception-2015-12-05-6726825d.pth',
                    help='Path to inception module for FID calculation')
parser.add_argument('--pretrained-clfs-dir-path', type=str, default='/local/home/palumboe/data/trained_clfs_polyMNIST',
                    help="Path to directory containing pre-trained digit classifiers for each modality")
parser.add_argument('--priorposterior', type=str, default='Laplace', choices=['Normal', 'Laplace'],
                    help='distribution choice for prior and posterior')

# Args
args = parser.parse_args()
flags_clf_lr = {'latdimz': args.latent_dim_z,
                'latdimw': args.latent_dim_w}
args.latent_dim_u = args.latent_dim_w + args.latent_dim_z

# Random seed
# https://pytorch.org/docs/stable/notes/randomness.html
# torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# CUDA stuff
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# Get model class
modelC = getattr(models, 'MMVAEplus_PolyMNIST_5modalities')
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

# Create path where to temporarily save images to compute FID scores
fid_path = os.path.join(args.datadir, 'fids_PM_' + (runPath.rsplit('/')[-1]))
datadirPM = os.path.join(args.datadir, "PolyMNIST")

# Save args to run
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


# Optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=1e-3, amsgrad=True)

# Data loaders
train_dataset, test_dataset = model.getDataSets(args.batch_size, device=device)
# Load validation and test indices

kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)


# Set up objective (IWAE/DReG)
objective = getattr(objectives,
                    ('m_' if hasattr(model, 'vaes') else '')
                    + args.obj)
t_objective = objective # Test objective (same as training)


# Loading pre-trained digit classifiers
clfs = [ClfImg() for idx, modal in enumerate(model.vaes)]
needs_conversion = not args.cuda
conversion_kwargs = {'map_location': lambda st, loc: st} if needs_conversion else {}
for idx, vae in enumerate(model.vaes):
    clfs[idx].load_state_dict(
        torch.load(os.path.join(args.pretrained_clfs_dir_path, "pretrained_img_to_digit_clf_m" + str(idx)),
                   **conversion_kwargs), strict=False)
    clfs[idx].eval()
    if args.cuda:
        clfs[idx].cuda()


def train(epoch):
    """
    Training function
    """
    model.train()
    b_loss = 0
    # Iterate over the data
    for i, dataT in enumerate(train_loader):
        # Unpack data
        data, labels_batch = unpack_data(dataT, device=device)
        optimizer.zero_grad()
        # Compute loss and backprop
        loss = -objective(model, data, K=args.K)
        loss.backward()
        # Optimizer step
        optimizer.step()
        # Get batch loss
        b_loss += loss.item()
        # Printing
        if args.print_freq > 0 and i % args.print_freq == 0:
            print("iteration {:04d}: loss: {:6.3f}".format(i, loss.item() / args.batch_size))
    # Epoch loss
    epoch_loss = b_loss / len(train_loader.dataset)
    wandb.log({"Loss/train": epoch_loss}, step=epoch)
    print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, epoch_loss))

def test(epoch):
    """
    Test function
    """
    # Set eval mode
    model.eval()
    b_loss = 0
    with torch.no_grad():
        # Get selected test samples for qualitative results
        test_selected_samples = get_10_polymnist_samples(test_loader.dataset,
                                                     num_testing_images=test_loader.dataset.__len__(), device=device)
        # Loop over the dataloader
        for i, dataT in enumerate(test_loader):
            # Unpack data
            data, _ = unpack_data(dataT, device=device)
            # Compute test loss
            loss = -t_objective(model, data, K=args.K, test=True)
            # Get batch loss
            b_loss += loss.item()
            # At first iteration save qualitative results
            if i == 0:
                # Compute cross-generations
                cg_imgs = model.self_and_cross_modal_generation(test_selected_samples, 10, 10)
                for i in range(NUM_VAES):
                    for j in range(NUM_VAES):
                        # print(cg_imgs[i][j].size())
                        wandb.log({'Cross_Generation/m{}/m{}'.format(i, j): wandb.Image(cg_imgs[i][j])}, step=epoch)
    # Epoch test loss
    epoch_loss = b_loss / len(test_loader.dataset)
    wandb.log({"Loss/test": epoch_loss}, step=epoch)
    print('====>             Test loss: {:.4f}'.format(epoch_loss))


def cross_coherence():
    """
    Compute cross coherence.
    """
    # Set eval mode
    model.eval()
    # Initialize matrix of coherences
    corrs = [[0 for idx, modal in enumerate(model.vaes)] for idx, modal in enumerate(model.vaes)]
    total = 0
    with torch.no_grad():
        # Iterate over test loader
        for i, dataT in enumerate(test_loader):
            # Unpack data
            data, targets = unpack_data(dataT, device)  # needs to be sent to device
            # Update total number of samples
            total += targets.size(0)
            # Test-time forward pass
            _, px_us, _ = model.self_and_cross_modal_generation_forward(data)
            # Update cross-coherences
            for idx_srt, srt_mod in enumerate(model.vaes):
                for idx_trg, trg_mod in enumerate(model.vaes):
                    clfs_results = torch.argmax(clfs[idx_trg](px_us[idx_srt][idx_trg].mean.squeeze(0)), dim=-1)
                    corrs[idx_srt][idx_trg] += (clfs_results == targets).sum().item()
        # Normalize cross-coherences
        for idx_trgt, vae in enumerate(model.vaes):
            for idx_strt, _ in enumerate(model.vaes):
                corrs[idx_strt][idx_trgt] = corrs[idx_strt][idx_trgt] / total
        # Compute average cross-coherences for each target modality (exclude self-reconstructions)
        means_target = [0 for idx, modal in enumerate(model.vaes)]
        for idx_target, _ in enumerate(model.vaes):
            means_target[idx_target] = mean(
                [corrs[idx_start][idx_target] for idx_start, _ in enumerate(model.vaes) if idx_start != idx_target])
        # Return matrix of cross-coherences, averages by targed modality, and overall average
    return corrs, means_target, mean(means_target)


def linear_latent_classification(clf_lr):
    """ Linear latent classification."""
    # Set eval mode
    model.eval()
    lr_acc_all_z = []
    lr_acc_m0_z, lr_acc_m1_z, lr_acc_m2_z, lr_acc_m3_z, lr_acc_m4_z = [], [], [], [], []
    lr_acc_m0_w, lr_acc_m1_w, lr_acc_m2_w, lr_acc_m3_w, lr_acc_m4_w = [], [], [], [], []
    lr_acc_m0_u, lr_acc_m1_u, lr_acc_m2_u, lr_acc_m3_u, lr_acc_m4_u = [], [], [], [], []
    accuracies_lr = {}
    with torch.no_grad():
        # Iterate over data
        for i, dataT in enumerate(test_loader):
            # Unpack data
            data, targets = unpack_data(dataT, device)
            b_size = data[0].size(0)
            labels_batch = nn.functional.one_hot(targets, num_classes=10).float()
            labels = labels_batch.cpu().data.numpy().reshape(b_size, 10)
            # Latent classification
            if clf_lr is not None:
                latent_reps = []
                # Get latent representation for each modality
                for v, vae in enumerate(model.vaes):
                    with torch.no_grad():
                        qu_x_params = vae.enc(data[v])
                        us_v = vae.qu_x(*qu_x_params).rsample()
                    ws_v, zs_v = torch.split(us_v, [args.latent_dim_w, args.latent_dim_z], dim=-1)
                    latent_reps.append([us_v.cpu().data.numpy(), ws_v.cpu().data.numpy(), zs_v.cpu().data.numpy()])
                # Classify from latents
                accuracies = classify_latent_representations(clf_lr, latent_reps, labels, split=True)
                # Averages for logging
                lr_acc_m0_u.append(np.mean(accuracies['m0_u']))
                lr_acc_m1_u.append(np.mean(accuracies['m1_u']))
                lr_acc_m2_u.append(np.mean(accuracies['m2_u']))
                lr_acc_m3_u.append(np.mean(accuracies['m3_u']))
                lr_acc_m4_u.append(np.mean(accuracies['m4_u']))

                lr_acc_m0_w.append(np.mean(accuracies['m0_w']))
                lr_acc_m1_w.append(np.mean(accuracies['m1_w']))
                lr_acc_m2_w.append(np.mean(accuracies['m2_w']))
                lr_acc_m3_w.append(np.mean(accuracies['m3_w']))
                lr_acc_m4_w.append(np.mean(accuracies['m4_w']))

                lr_acc_m0_z.append(np.mean(accuracies['m0_z']))
                lr_acc_m1_z.append(np.mean(accuracies['m1_z']))
                lr_acc_m2_z.append(np.mean(accuracies['m2_z']))
                lr_acc_m3_z.append(np.mean(accuracies['m3_z']))
                lr_acc_m4_z.append(np.mean(accuracies['m4_z']))
                lr_acc_all_z.append(np.mean(accuracies['all']))

        # Log accuracies
        accuracies_lr['m0_u'] = mean(lr_acc_m0_u)
        accuracies_lr['m1_u'] = mean(lr_acc_m1_u)
        accuracies_lr['m2_u'] = mean(lr_acc_m2_u)
        accuracies_lr['m3_u'] = mean(lr_acc_m3_u)
        accuracies_lr['m4_u'] = mean(lr_acc_m4_u)

        accuracies_lr['m0_w'] = mean(lr_acc_m0_w)
        accuracies_lr['m1_w'] = mean(lr_acc_m1_w)
        accuracies_lr['m2_w'] = mean(lr_acc_m2_w)
        accuracies_lr['m3_w'] = mean(lr_acc_m3_w)
        accuracies_lr['m4_w'] = mean(lr_acc_m4_w)

        accuracies_lr['m0_z'] = mean(lr_acc_m0_z)
        accuracies_lr['m1_z'] = mean(lr_acc_m1_z)
        accuracies_lr['m2_z'] = mean(lr_acc_m2_z)
        accuracies_lr['m3_z'] = mean(lr_acc_m3_z)
        accuracies_lr['m4_z'] = mean(lr_acc_m4_z)

        accuracies_lr['_mean_u'] = mean([accuracies_lr['m{}_u'.format(n)] for n in range(5)])
        accuracies_lr['_mean_w'] = mean([accuracies_lr['m{}_w'.format(n)] for n in range(5)])
        accuracies_lr['_mean_z'] = mean([accuracies_lr['m{}_z'.format(n)] for n in range(5)])

        accuracies_lr['z_all'] = mean(lr_acc_all_z)
    return accuracies_lr

def unconditional_coherence():
    """ Compute unconditional coherence"""
    # Set eval mode
    model.eval()
    # Set counts
    correct = 0
    total = 0
    with torch.no_grad():
        # Iterate over data (just to have the same number of samples considered as in conditional coherence)
        for i, dataT in enumerate(test_loader):
            # Unpack data
            data, targets = unpack_data(dataT, device)
            b_size = data[0].size(0)
            # labels_batch = nn.functional.one_hot(targets, num_classes=10).float()
            # labels = labels_batch.cpu().data.numpy().reshape(b_size, 10)
            # Joint generation
            uncond_gens = model.generate_unconditional(N=b_size, coherence_calculation=True, fid_calculation=False)
            uncond_gens = [elem.to(device) for elem in uncond_gens]
            clfs_resultss = []
            # Compute coherence across generated modalities
            for idx_trg, trg_mod in enumerate(model.vaes):
                clfs_results = torch.argmax(clfs[idx_trg](uncond_gens[idx_trg]), dim=-1)
                if idx_trg == 0:
                    total += b_size
                clfs_resultss.append(clfs_results)
            clfs_resultss_tensor = torch.stack(clfs_resultss, dim=-1)
            for dim in range(clfs_resultss_tensor.size(0)):
                if torch.unique(clfs_resultss_tensor[dim, :]).size(0) == 1:
                    correct += 1

        # Normalize unconditional coherence
        uncond_coherence = correct / total

    return uncond_coherence


def calculate_fid_routine(datadirPM, fid_path, num_fid_samples, epoch):
    """ Calculate FID scores for unconditional and conditional generation """
    total_cond = 0
    # Create new directories for conditional FIDs
    for j in [0, 1, 2, 3, 4]:
        if os.path.exists(os.path.join(fid_path, 'random', 'm{}'.format(j))):
            shutil.rmtree(os.path.join(fid_path, 'random', 'm{}'.format(j)))
            os.makedirs(os.path.join(fid_path, 'random', 'm{}'.format(j)))
        else:
            os.makedirs(os.path.join(fid_path, 'random', 'm{}'.format(j)))
        for i in [0, 1, 2, 3, 4]:
            if os.path.exists(os.path.join(fid_path, 'm{}'.format(j), 'm{}'.format(i))):
                shutil.rmtree(os.path.join(fid_path, 'm{}'.format(j), 'm{}'.format(i)))
                os.makedirs(os.path.join(fid_path, 'm{}'.format(j), 'm{}'.format(i)))
            else:
                os.makedirs(os.path.join(fid_path, 'm{}'.format(j), 'm{}'.format(i)))
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
            data, _ = unpack_data(dataT, device=device)
            if total_cond < num_fid_samples:
                model.self_and_cross_modal_generation_for_fid_calculation(data, fid_path, i)
                total_cond += data[0].size(0)
        calculate_inception_features_for_gen_evaluation(args.inception_path, device,
                                                        fid_path, datadirPM)
        # FID calculation
        fid_randm_list = []
        fid_condgen_list = []
        for modality_target in ['m{}'.format(m) for m in range(5)]:
            file_activations_real = os.path.join(args.datadir, 'PolyMNIST', 'test',
                                                 'real_activations_{}.npy'.format(modality_target))
            feats_real = np.load(file_activations_real)
            file_activations_randgen = os.path.join(fid_path, 'random',
                                                    modality_target + '_activations.npy')
            feats_randgen = np.load(file_activations_randgen)
            fid_randval = calculate_fid(feats_real, feats_randgen)
            wandb.log({"FID/Random/{}".format(modality_target): fid_randval}, step=epoch)
            fid_randm_list.append(fid_randval)
            fid_condgen_target_list = []
            for modality_source in ['m{}'.format(m) for m in range(5)]:
                file_activations_gen = os.path.join(fid_path, modality_source,
                                                    modality_target + '_activations.npy')
                feats_gen = np.load(file_activations_gen)
                fid_val = calculate_fid(feats_real, feats_gen)
                wandb.log({"FID/{}/{}".format(modality_source, modality_target): fid_val}, step=epoch)
                fid_condgen_target_list.append(fid_val)
            fid_condgen_list.append(mean(fid_condgen_target_list))
        mean_fid_condgen = mean(fid_condgen_list)
        mean_fid_randm = mean(fid_randm_list)
        wandb.log({"FID/random_meanall": mean_fid_randm}, step=epoch)
        wandb.log({"FID/condgen_meanll": mean_fid_condgen}, step=epoch)
    # Clean up
    if os.path.exists(fid_path):
        shutil.rmtree(fid_path)
        os.makedirs(fid_path)


def train_clf_lr(dl):
    """
    Train linear classifier on latent representations
    """
    latent_rep = {'m0': {'us': [], 'zs': [], 'ws': []},
                  'm1': {'us': [], 'zs': [], 'ws': []},
                  'm2': {'us': [], 'zs': [], 'ws': []},
                  'm3': {'us': [], 'zs': [], 'ws': []},
                  'm4': {'us': [], 'zs': [], 'ws': []}}
    labels_all = []
    for i, dataT_lr in enumerate(dl):
        data, labels_batch = unpack_data(dataT_lr, device=device)
        b_size = data[0].size(0)
        labels_batch = nn.functional.one_hot(labels_batch, num_classes=10).float()
        labels = labels_batch.cpu().data.numpy().reshape(b_size, 10);
        labels_all.append(labels)
        for v, vae in enumerate(model.vaes):
            # latent_rep['m{}'.format(v)]
            with torch.no_grad():
                qu_x_params = vae.enc(data[v])
                us_v = vae.qu_x(*qu_x_params).rsample()
            ws_v, zs_v = torch.split(us_v, [args.latent_dim_w, args.latent_dim_z], dim=-1)
            latent_rep['m{}'.format(v)]['us'].append(us_v.cpu().data.numpy())
            latent_rep['m{}'.format(v)]['zs'].append(zs_v.cpu().data.numpy())
            latent_rep['m{}'.format(v)]['ws'].append(ws_v.cpu().data.numpy())
            # latent_reps.append([zs_v.cpu().data.numpy(), ws_v.cpu().data.numpy(), us_v.cpu().data.numpy()])
    # print(labels_all[0].shape)
    labels_all = np.concatenate(labels_all, axis=0)
    gt = np.argmax(labels_all, axis=1).astype(int)
    clf_lr = dict();
    for v, vae in enumerate(model.vaes):
        latent_rep_u = np.concatenate(latent_rep['m{}'.format(v)]['us'], axis=0)
        latent_rep_w = np.concatenate(latent_rep['m{}'.format(v)]['ws'], axis=0)
        latent_rep_z = np.concatenate(latent_rep['m{}'.format(v)]['zs'], axis=0)
        # data_rep_uw, data_rep_w, data_rep_u = data_k
        clf_lr_rep_u = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto', max_iter=1000)
        clf_lr_rep_z = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto', max_iter=1000)
        clf_lr_rep_w = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto', max_iter=1000)
        clf_lr_rep_u.fit(latent_rep_u, gt.ravel())
        clf_lr['m' + str(v) + '_' + 'u'] = clf_lr_rep_u
        clf_lr_rep_w.fit(latent_rep_w, gt.ravel())
        clf_lr['m' + str(v) + '_' + 'w'] = clf_lr_rep_w
        clf_lr_rep_z.fit(latent_rep_z, gt.ravel())
        clf_lr['m' + str(v) + '_' + 'z'] = clf_lr_rep_z
    return clf_lr


def non_linear_latent_classification(epochs, mod):
    """
    Classify latents non-linear
    """
    model.eval()

    classifier_u = NonLinearLatent_Classifier(args.latent_dim_u, 10).to(device)
    classifier_z = NonLinearLatent_Classifier(args.latent_dim_z, 10).to(device)
    classifier_w = NonLinearLatent_Classifier(args.latent_dim_w, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_u = optim.Adam(classifier_u.parameters(), lr=0.001)
    optimizer_z = optim.Adam(classifier_z.parameters(), lr=0.001)
    optimizer_w = optim.Adam(classifier_w.parameters(), lr=0.001)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss_u = 0.0
        running_loss_z = 0.0
        running_loss_w = 0.0
        total_iters_train = len(train_loader)
        print('\n====> Epoch: {:03d} '.format(epoch))
        for i, dataT in enumerate(train_loader):
            data, labels = unpack_data(dataT, device=device)
            # m1_batch, m2_batch, m3_batch, m4_batch, m5_batch = data_batch
            # labels_batch = nn.functional.one_hot(labels_batch, num_classes=10).float()
            data_batch = data[mod]
            # labels_batch = labels_batch.to(FLAGS.device);

            with torch.no_grad():
                qu_x_params = model.vaes[mod].enc(data_batch)
                us = model.vaes[mod].qu_x(*qu_x_params).rsample()
                ws, zs = torch.split(us, [args.latent_dim_w, args.latent_dim_z], dim=-1)

            optimizer_u.zero_grad()
            optimizer_z.zero_grad()
            optimizer_w.zero_grad()
            outputs_u = classifier_u(us)
            outputs_z = classifier_z(zs)
            outputs_w = classifier_w(ws)
            loss_u = criterion(outputs_u, labels)
            loss_z = criterion(outputs_z, labels)
            loss_w = criterion(outputs_w, labels)
            loss_u.backward()
            loss_z.backward()
            loss_w.backward()
            optimizer_u.step()
            optimizer_z.step()
            optimizer_w.step()
            # print statistics
            running_loss_u += loss_u.item()
            running_loss_z += loss_z.item()
            running_loss_w += loss_w.item()
        print('Finished Training, calculating test loss...')

        classifier_u.eval()
        classifier_z.eval()
        classifier_w.eval()
        total = 0
        correct_u = 0
        correct_z = 0
        correct_w = 0
        with torch.no_grad():
            for i, dataT in enumerate(test_loader):
                data, labels = unpack_data(dataT, device=device)
                data_batch = data[mod]
                with torch.no_grad():
                    qu_x_params = model.vaes[mod].enc(data_batch)
                    us = model.vaes[mod].qu_x(*qu_x_params).rsample()
                    ws, zs = torch.split(us, [args.latent_dim_w, args.latent_dim_z], dim=-1)

                outputs_u = classifier_u(us)
                outputs_z = classifier_z(zs)
                outputs_w = classifier_w(ws)
                _, predicted_u = torch.max(outputs_u.data, 1)
                _, predicted_z = torch.max(outputs_z.data, 1)
                _, predicted_w = torch.max(outputs_w.data, 1)
                total += labels.size(0)
                correct_u += (predicted_u == labels).sum().item()
                correct_z += (predicted_z == labels).sum().item()
                correct_w += (predicted_w == labels).sum().item()
        print('The classifier correctly classified {} out of {} examples with u. Accuracy: '
              '{:.2f}%'.format(correct_u, total, correct_u / total * 100))
        print('The classifier correctly classified {} out of {} examples with z . Accuracy: '
              '{:.2f}%'.format(correct_z, total, correct_z / total * 100))
        print('The classifier correctly classified {} out of {} examples with w. Accuracy: '
              '{:.2f}%'.format(correct_w, total, correct_w / total * 100))
        wandb.log({"Latclassacc_nl_u/m{}".format(mod): (correct_u / total * 100)}, step=epoch)
        wandb.log({"Latclassacc_nl_z/m{}".format(mod): (correct_z / total * 100)}, step=epoch)
        wandb.log({"Latclassacc_nl_w/m{}".format(mod): (correct_w / total * 100)}, step=epoch)




if __name__ == '__main__':
    with Timer('MMVAEplus') as t:
        for epoch in range(1, args.epochs + 1):
            train(epoch) # Train the model
            if epoch % 25 == 0:
                # Generate samples
                gen_samples = model.generate_unconditional(N=100, coherence_calculation=False, fid_calculation=False)
                for j in range(NUM_VAES):
                    wandb.log({'Generations/m{}'.format(j) :  wandb.Image(gen_samples[j])}, step=epoch)

                # Test function (contains also cross-generations)
                test(epoch)
                # Train latent classfier
                clf_lr = train_clf_lr(train_loader)
                # Compute cross-coherence
                cors, means_tgt, ccmeanall = cross_coherence()
                wandb.log({"Conditional_coherence_meanall": ccmeanall}, step=epoch)
                for i in range(NUM_VAES):
                    wandb.log({"Conditional_coherence_target_m{}".format(i): means_tgt[i]}, step=epoch)
                    for j in range(NUM_VAES):
                        wandb.log({"Conditional_coherence_m{}xm{}".format(i, j): cors[i][j]}, step=epoch)
                # Calculate unconditional coherence and linear latent classification accuracies
                uncond_coher = unconditional_coherence()
                accuracies_lc = linear_latent_classification(clf_lr)
                wandb.log({"Unconditional_coherence": uncond_coher}, step=epoch)
                for key in accuracies_lc:
                    wandb.log({"Latclassacc_" + key: accuracies_lc[key]}, step=epoch)
                # Calculate nonlinear latent classification accuracies c
                for mod_clf_nl in range(NUM_VAES):
                    non_linear_latent_classification(1, mod_clf_nl)
                # Save checkpoint (light)
                save_model_light(model, runPath + '/model_' + str(epoch) + '.rar')
                # # Calculate FID scores
        calculate_fid_routine(datadirPM, fid_path, 10000, epoch)


