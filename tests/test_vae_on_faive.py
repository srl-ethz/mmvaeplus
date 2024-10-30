import os
import torch
from dataset_parser.faive.h5_tools import load_topic_arrays_h5
import numpy as np
import argparse

from mmvaes.mmvaeplus import MMVAEPlusWrapper
from dataset_parser.mano_utils import HandKinematicsInverse


def export_gc_angles(episode_file_path):
    topic_arrays = load_topic_arrays_h5(episode_file_path, verbose=True, selected_topics=["/faive/policy_output", "/ingress/mano"])
    gc_angles = topic_arrays["/faive/policy_output"]["message"]
    mano_data = topic_arrays["/ingress/mano"]["message"]

    mano_local_rep = HandKinematicsInverse().forward(torch.from_numpy(mano_data)).detach().cpu().numpy()
    return gc_angles, mano_local_rep

def export_mano_data():
    mano_path = '/mnt/data1/erbauer/grab_test_data/rhand_params_watch_lift.npy'
    # list of dicts
    mano_data = np.load(mano_path, allow_pickle=True)
    pose_data = [d['pose'] for d in mano_data]
    pose_data = np.stack(pose_data)
    print(pose_data.shape)
    return pose_data


def conditional_generation(input_data, mmvae_wrapper, modality):
    # encode gc_angles to latents
    input_data = input_data.squeeze()

    print(input_data.shape)
    original_input_data = input_data.copy()
    
    # Create zeros array with correct shape
    zeros = np.zeros((input_data.shape[0], 6))
    
    # Concatenate zeros and gc_angles
    input_data = np.concatenate([zeros, input_data], axis=1)
    
    # generate latents
    encoded_actions = mmvae_wrapper.encode_data(input_data, modality=modality)
    latents = encoded_actions[:, 6:]

    # decode latents to gc_angles
    decoded_data = mmvae_wrapper.decode_data(latents, modality=modality).detach().cpu().numpy()
    return (original_input_data, decoded_data)

def test_vae_on_mano(args):
    pose_data = export_mano_data()
    model_path = os.path.join(args.mmvaeplus_model_path, args.run_name)
    mmvae_wrapper = MMVAEPlusWrapper(model_path, args.mmvaeplus_model_epoch, scaling_method='standard')
    outputs = conditional_generation(pose_data, mmvae_wrapper, 'mano_params')
    np.save(args.output_file_path + 'mano_vae_test.npy', outputs)
    print(f'Saved to {args.output_file_path + "mano_vae_test.npy"} using {args.run_name} model')
    
def test_vae_on_faive(args, include_mano=False):
    gc_angles, mano_local_rep = export_gc_angles(args.episode_file_path)
    print(mano_local_rep.shape)
    model_path = os.path.join(args.mmvaeplus_model_path, args.run_name)
    mmvae_wrapper = MMVAEPlusWrapper(model_path, args.mmvaeplus_model_epoch, scaling_method='standard')
    os.makedirs(args.output_file_path, exist_ok=True)

    if include_mano:
        mano_outputs = conditional_generation(mano_local_rep, mmvae_wrapper, 'mano_params')
        np.save(args.output_file_path + 'mano_vae_test.npy', mano_outputs)
        print(f'Saved to {args.output_file_path + "mano_vae_test.npy"} using {args.run_name} model')

    gc_angles_outputs = conditional_generation(gc_angles, mmvae_wrapper, 'gc_angles')
    np.save(args.output_file_path + 'gc_angles_vae_test.npy', gc_angles_outputs)
    print(f'Saved to {args.output_file_path + "gc_angles_vae_test.npy"} using {args.run_name} model')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export GC angles from a Faive episode file.")
    parser.add_argument("--episode_file_path", default='/home/erik/msthesis/data/episode_0_success.h5', type=str,
                        help="Path to the episode file.")
    parser.add_argument("--output_file_path", default='/home/erik/msthesis/data/tests/', type=str,
                        help="Path to the output file.")
    parser.add_argument("--mmvaeplus_model_path", type=str, default='/home/erik/msthesis/encoders/mmvaeplus/outputs/RobotActions_1/checkpoints/',
                        help="Path to the mmvaeplus model.")
    parser.add_argument("--run_name", type=str, default='stoic-blaze-59',
                        help="Name of the run.")
    parser.add_argument("--mmvaeplus_model_epoch", type=str, default='best',
                        help="Epoch of the mmvaeplus model.")
    args = parser.parse_args()

    # test_vae_on_mano(args)
    test_vae_on_faive(args, include_mano=True)