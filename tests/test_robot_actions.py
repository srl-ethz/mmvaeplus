import numpy as np
import os
import torch
import pickle
from mmvaeplus.models.mmvaeplus_robot_actions import build_model, RobotActions
from action_retargeting.data.dataloader import get_hand_dataloader

def generate_from_mano(model: RobotActions, mano_params: torch.Tensor, dataset_stats: dict, device: torch.device = torch.device('cpu')):
    '''
    Conditional generation from MANO parameters.

    model: MMVAEPlus robot actions model
    mano_params: sequence of MANO parameters (B, 45)
    device: device to run the model on
    Returns: sequence of self- and cross-reconstructed data (gc_angles, mano_params, simple_gripper)
    '''

    # Convert MANO parameters to the required format
    mano_params = torch.tensor(mano_params, dtype=torch.float32).to(device)
    hand_pose_stats, gc_angles_stats, simple_gripper_stats = unwrap_dataset_stats(dataset_stats, device)

    mano_params = normalize_data(mano_params, *hand_pose_stats)
   
    # 3x3 grid of gc_angles, mano_params, simple_gripper with conditional generations
    # entries have shave (1, B, modality_dim)
    outputs = model.targeted_generation(mano_params, 1, 'all')

    gc_angles_cond = outputs[1][0]
    mano_params_cond = outputs[1][1]
    simple_gripper_cond = outputs[1][2]

    # unnormalize the data
    mano_params_in = unnormalize_data(mano_params, *hand_pose_stats)
    gc_angles_cond = unnormalize_data(gc_angles_cond, *gc_angles_stats)
    mano_params_cond = unnormalize_data(mano_params_cond, *hand_pose_stats)
    simple_gripper_cond = unnormalize_data(simple_gripper_cond, *simple_gripper_stats)

    # get all conditional generations from mano_params
    mano_params_in = mano_params_in.cpu().numpy()
    gc_angles_cond = gc_angles_cond.squeeze(0).cpu().numpy()
    mano_params_cond = mano_params_cond.squeeze(0).cpu().numpy()
    simple_gripper_cond = simple_gripper_cond.squeeze(0).cpu().numpy()

    ret_dict = {
        'mano_params_inputs': mano_params_in,
        'gc_angles_outputs': gc_angles_cond,
        'mano_params_outputs': mano_params_cond,
        'simple_gripper_outputs': simple_gripper_cond
    }

    return ret_dict

def unnormalize_data(data, mean, std):
    return data * std + mean

def normalize_data(data, mean, std):
    return (data - mean) / std

def move_to_torch_dev(data, device):
    if isinstance(data, np.ndarray):
        return torch.tensor(data, device=device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [move_to_torch_dev(item, device) for item in data]
    elif isinstance(data, dict):
        return {key: move_to_torch_dev(value, device) for key, value in data.items()}
    else:
        raise ValueError(f'Invalid data type: {type(data)}')

def unwrap_dataset_stats(dataset_stats, device):
    dataset_stats = move_to_torch_dev(dataset_stats, device)
    hand_pose_mean = dataset_stats['hand_pose']['mean']
    hand_pose_std = dataset_stats['hand_pose']['std']
    gc_angles_mean = dataset_stats['faive_angles']['mean']
    gc_angles_std = dataset_stats['faive_angles']['std']
    simple_gripper_params_mean = dataset_stats['onedof_pose']['mean']
    simple_gripper_params_std = dataset_stats['onedof_pose']['std']
    return [hand_pose_mean, hand_pose_std], [gc_angles_mean, gc_angles_std], [simple_gripper_params_mean, simple_gripper_params_std]
   

def load_mano_params(filepath):
    '''
    Load MANO parameters from a file.
    Should be a dict with keys 'pose', 'shape', 'trans', 'rot'
    Shapes:
    pose: (B, 45)
    shape: (B, 10)
    trans: (B, 3)
    rot: (B, 3)

    shape, trans, rot may just be zeros depending on the data source.
    '''

    data = np.load(filepath, allow_pickle=True)
    mano_params = np.array([entry['pose'] for entry in data])
    return mano_params

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_name = 'divine-dawn-44'
    model_epoch = 'best'

    model_path = f'/home/erbauer/vaes/mmvaeplus/outputs/RobotActions_1/checkpoints/{run_name}/'
    mano_filepath = '/mnt/data1/erbauer/grab_test_data/rhand_params.npy'
    out_base_path = model_path
    os.makedirs(out_base_path, exist_ok=True)

    model, dataset_stats = build_model(model_path, model_epoch, device)
    model.eval()
    mano_params = load_mano_params(mano_filepath)

    out_dict = generate_from_mano(model, mano_params, dataset_stats, device)

    # save out_dict
    with open(os.path.join(out_base_path, f'out_dict_{run_name}_{model_epoch}.pkl'), 'wb') as f:
        pickle.dump(out_dict, f)

    print(f'Saved to {os.path.join(out_base_path, f"out_dict_{run_name}_{model_epoch}.pkl")}')
