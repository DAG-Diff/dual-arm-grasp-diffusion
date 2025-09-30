import numpy as np
import h5py 
import os 
from tqdm import tqdm
import trimesh
from scipy.spatial.transform import Rotation as R
from trimesh.collision import CollisionManager
import random
import concurrent.futures 
import argparse

gripper = trimesh.load('./gripper.obj')

def seed_all(seed=28):
    np.random.seed(seed)
    random.seed(seed)

def random_T(in_T):
    T = np.eye(4)
    rot = R.from_euler('y', np.random.uniform(-np.pi, np.pi, 1), degrees=False).as_matrix()
    rot = R.from_euler('z', np.random.uniform(-np.pi/6, np.pi/6, 1), degrees=False).as_matrix() @ rot
    t = in_T[:3, 1] * np.random.uniform(0, 0.03, 1)
    T[:3, :3] = rot
    T[:3, 3] = t
    return T


def batch_random_T(in_T, N):
    # Random rotation around y-axis
    y_angles = np.random.uniform(-np.pi, np.pi, N)
    R_y = R.from_euler('y', y_angles, degrees=False).as_matrix()  # (N, 3, 3)

    # Random rotation around z-axis
    z_angles = np.random.uniform(-np.pi/6, np.pi/6, N)
    R_z = R.from_euler('z', z_angles, degrees=False).as_matrix()  # (N, 3, 3)

    # Compose final rotation: R_z @ R_y
    R_final = np.einsum('nij,njk->nik', R_z, R_y)  # (N, 3, 3)

    # Translation along the in_T's y-axis
    t_dir = in_T[:, :3, 1]  # (3,)
    t_mags = np.random.uniform(0, 0.02, N)  # (N,)
    t_final = t_mags[:, None] * t_dir[None, :]  # (N, 3)

    # Combine into full transformation matrices
    T_batch = np.tile(np.eye(4), (N, 1, 1))  # (N, 4, 4)
    T_batch[:, :3, :3] = R_final
    T_batch[:, :3, 3] = t_final

    return T_batch  # (N, 4, 4)

def collision_checker(mesh, gripper):
    collision = CollisionManager()
    collision.add_object('mesh', mesh)
    collision.add_object('gripper', gripper)
    return collision.in_collision_internal()

def main():
    
    seed_all(seed=28)
    
    args = argparse.ArgumentParser()
    args.add_argument('--num_workers', type=int, default=1, help='Number of workers for parallel processing')
    args.add_argument('--save_path', type=str)
    args = args.parse_args()
    
    GRASP_PATH = '/scratch/dualarm/DG16M/dg16m/grasps'
    MESH_PATH = '/scratch/dualarm/DG16M/dg16m/meshes'
    # SAVE_PATH = '/scratch/dualarm/DG16M/dg16m/negative_colliding_grasps'
    SAVE_PATH = args.save_path
    
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    num_files = len(os.listdir(GRASP_PATH))
    for idx in tqdm(range(num_files)):
        grasp_file = os.path.join(GRASP_PATH, os.listdir(GRASP_PATH)[idx])
        mesh_file = os.path.join(MESH_PATH, os.path.basename(grasp_file).replace('.h5', '.obj'))

        mesh = trimesh.load(mesh_file)
        grasp_file = h5py.File(grasp_file, 'r')
        grasps = grasp_file['grasps/grasps'][:].reshape(-1, 4, 4)
        random.shuffle(grasps)
        
        colliding_grasps = []

        T = batch_random_T(grasps, len(grasps))
        grasp_perturbed = grasps.copy()

        grasp_perturbed[:, :3, :3] = grasp_perturbed[:, :3, :3] @ T[:, :3, :3]
        grasp_perturbed[:, :3, 3] = grasp_perturbed[:, :3, 3] + T[:, :3, 3]

        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            grippers_now = [gripper.copy().apply_transform(g) for g in grasp_perturbed]
            results = list(executor.map(collision_checker,
                                        [mesh] * len(grasp_perturbed),
                                        grippers_now))

        colliding_grasps = np.where(np.array(results) == True)[0]
        colliding_grasps = grasp_perturbed[colliding_grasps]
        
        
        np.save(os.path.join(SAVE_PATH, f"{os.path.basename(mesh_file).replace('.obj', '')}.npy"), 
                colliding_grasps)
        
        with open(os.path.join(SAVE_PATH, '../colliding_grasps_info.txt'), 'a') as f:
            f.write(f'{os.path.basename(mesh_file)}: {len(colliding_grasps)}\n')
        
        print(f'Saved {len(colliding_grasps)} colliding grasps for {os.path.basename(mesh_file)}')
        
if __name__ == "__main__":
    main()