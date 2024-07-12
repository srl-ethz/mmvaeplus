import os
import numpy as np
import glob
from fid.inception import InceptionV3
from fid.fid_score import get_activations
from fid.fid_score import calculate_frechet_distance

def calculate_inception_features_for_gen_evaluation(inception_state_dict_path, device, dir_fid_base, datadir, dims=2048, batch_size=128):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx], path_state_dict=inception_state_dict_path)
    model = model.to(device)

    # for moddality_num in range(0):
    moddality_num = 0
    moddality = 'm{}'.format(moddality_num)
    filename_act_real_calc = os.path.join(dir_fid_base, 'test','real_activations_{}.npy'.format(moddality))
    if not os.path.exists(filename_act_real_calc):
        files_real_calc = glob.glob(os.path.join(dir_fid_base, 'test', moddality, '*' + '.png'))
        act_real_calc = get_activations(files_real_calc, model, device, batch_size, dims, verbose=False)
        np.save(filename_act_real_calc, act_real_calc)

    for prefix  in ['random', 'm0', 'm1']:
        dir_gen = os.path.join(dir_fid_base, prefix)
        if not os.path.exists(dir_gen):
            raise RuntimeError('Invalid path: %s' % dir_gen)
        # for modality in ['m{}'.format(m) for m in range(5)]:
        modality = 'm{}'.format(0)
        files_gen = glob.glob(os.path.join(dir_gen, modality, '*' + '.png'))
        filename_act = os.path.join(dir_gen,
                                       modality + '_activations.npy')
        act_rand_gen = get_activations(files_gen, model, device, batch_size, dims, verbose=False)
        np.save(filename_act, act_rand_gen)

def calculate_fid(feats_real, feats_gen):
    mu_real = np.mean(feats_real, axis=0)
    sigma_real = np.cov(feats_real, rowvar=False)
    mu_gen = np.mean(feats_gen, axis=0)
    sigma_gen = np.cov(feats_gen, rowvar=False)
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    return fid;

def calculate_fid_dict(feats_real, dict_feats_gen):
    dict_fid = dict();
    for k, key in enumerate(dict_feats_gen.keys()):
        feats_gen = dict_feats_gen[key];
        dict_fid[key] = calculate_fid(feats_real, feats_gen);
    return dict_fid;

def get_clf_activations(flags, data, model):
    model.eval();
    act = model.get_activations(data);
    act = act.cpu().data.numpy().reshape(flags.batch_size, -1)
    return act;
