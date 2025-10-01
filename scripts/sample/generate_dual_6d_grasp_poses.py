import copy
import configargparse

import scipy.spatial.transform
import numpy as np
from se3dif.models.loader import load_model
from se3dif.samplers import DualGrasp_AnnealedLD
from se3dif.utils import to_numpy, to_torch
import torch.nn.functional as F

import torch
import os
import trimesh
import random
from icecream import ic

def seed_all(seed=28):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
def parse_args():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--n_grasps', type=str, default='200')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--model', type=str, default='cgdf_v1')
    p.add_argument('--input', type=str, required=True)
    p.add_argument('--debug_mode', action='store_true', default=False)

    opt = p.parse_args()
    return opt


def get_approximated_grasp_diffusion_field(p, args, device='cpu', seed=28):
    model_params = args.model
    batch = int(args.n_grasps)
    ## Load model
    model_args = {
        'device': device,
        'pretrained_model': model_params,
        'use_attention': False,
    }
    print(model_args)
    model = load_model(model_args)
    # Reproduce
    model = model.eval().to(torch.float32)
    context = to_torch(p[None,...], device).to(torch.float32)
    model.set_latent(context, batch=batch)

    ########### 2. SET SAMPLING METHOD #############
    generator = DualGrasp_AnnealedLD(model, batch=batch, T=200, T_fit=25, k_steps=1, device=device, seed=seed)

    return generator, model

def mean_center_and_normalize(mesh):
    mean = mesh.sample(1000).mean(0)
    mesh.vertices -= mean
    scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
    mesh.apply_scale(1/(2*scale))
    return mesh


def sample_pointcloud(input_path=None, return_transform=False):
    mesh = trimesh.load(input_path)
    mesh = mean_center_and_normalize(mesh)
    # mesh.export('mesh.obj')
    scaling = 1

    # sample point cloud
    P = mesh.sample(1000)    
    P = P * scaling
    
    # apply random rotation
    # sampled_rot = scipy.spatial.transform.Rotation.random()
    sampled_rot = scipy.spatial.transform.Rotation.from_euler('z', np.random.uniform(0, 2 * np.pi), degrees=False)
    rot = sampled_rot.as_matrix()
    # rot = np.eye(3)

    P = np.einsum('mn,bn->bm', rot, P)
    P *= 8.
    P_mean = np.mean(P, 0)
    P += -P_mean

    H = np.eye(4)
    H[:3,:3] = rot
    mesh.apply_transform(H)
    mesh.apply_scale(8.)
    H = np.eye(4)
    H[:3,-1] = -P_mean
    mesh.apply_transform(H)
    
    print(f"Max point of P: {np.max(P, axis=0)} | mean: {P_mean}")


    if return_transform:
        return P, mesh, rot, scaling
    
    return P, mesh


def main(get_args=True, input_dict=None):
    seed=128
    seed_all(seed)

    if get_args:
        args = parse_args()
    else:
        args = configargparse.ArgumentParser()

        # Add arguments to the parser
        args.add_argument('--n_grasps', type=int)
        args.add_argument('--device', type=str)
        args.add_argument('--input', type=str)
        args.add_argument('--model', type=dict)  # or customize based on model structure,
        args.add_argument('--debug_mode', action='store_true', default=False)

        # Parse into a namespace using known values
        args = args.parse_args(args=[], namespace=configargparse.Namespace(
            n_grasps=input_dict['n_grasps'],
            device=input_dict['device'],
            input=input_dict['input'],
            model=input_dict['model'],
            save_path=input_dict['save_path']
        ))
        
    n_grasps = int(args.n_grasps)
    device = args.device
    input_path = args.input

    ## Set Model and Sample Generator ##
    
    # P, mesh, rot, scaling = sample_pointcloud(input_path, return_transform=True)
    # generator, model = get_approximated_grasp_diffusion_field(P, args, device, seed)
    # ...existing code...
    P, mesh, rot, scaling = sample_pointcloud(input_path, return_transform=True)
    generator, model = get_approximated_grasp_diffusion_field(P, args, device, seed)
    
    # Running the model
    dual = True
    # model.eval()
    save_path = True
    if save_path:
        if args.debug_mode:
            print('*************** Debug mode ***************')
            H_, traj, t = generator.sample_debug(dual=dual, save_path=save_path)
        else:
            # H_, traj, t = generator.sample(dual=dual, save_path=save_path)
            H_, traj, t, energies, force_closures, collisions = generator.sample(dual=dual, save_path=save_path)
    else:
        H_, t = generator.sample(dual=dual, save_path=save_path)

    ic(traj.shape)
    
    H = H_.reshape(-1,4,4)
    if not dual:
        H1, H2 = H[:n_grasps, ...], H[n_grasps:, ...]
        e = model(H1=H1, H2=H2, k1=t, k2=t, batch=1).squeeze()
    else:
        # energy = model(perturbed_H, random_t.unsqueeze(0).repeat(1,2).reshape(-1), batch=batch, dual=True)
        e = model(H, t, batch=1, dual=dual).flatten()
    
    mask = e < 0 # emperically set threshold, changes with different ckpt.
    # H = H[mask]
    H_dual = H.clone().reshape(-1, 2, 4, 4)
    print(H_dual.shape)

    print(f"Generated {H_dual.shape[0]} valid grasps")

    # H[..., :3, -1] *=1/8.
    H_dual[..., :3, -1] *=1/8. * 1/scaling
    
    traj[..., :3, -1] *=1/8. * 1/scaling

    P *=1/8 * 1/scaling
    mesh = mesh.apply_scale(1/8)
    
    mesh.export('./temp/only_output_mesh.obj')
    torch.save(H_dual, './temp/ouput_dual_grasps.pt')
    torch.save(e, './temp/output_energy.pt')
    torch.save(traj.detach().cpu().numpy(), './temp/grasps_traj.pt')
    torch.save(model.pred_label.detach().cpu().numpy(), './temp/grasp_scores.pt')
    torch.save(P, './temp/point_cloud.pt')
    torch.save(model.collision_pred.detach().cpu().numpy(), './temp/collision_scores.pt')
    torch.save(rot, './temp/rotation.pt')
    
    torch.save(energies, './temp/energies.pt')
    torch.save(force_closures, './temp/force_closures.pt')
    torch.save(collisions, './temp/collisions.pt')
    
if __name__ == "__main__":
    main(get_args=True, input_dict=None)