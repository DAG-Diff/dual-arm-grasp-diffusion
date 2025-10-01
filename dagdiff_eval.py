import os
from tqdm import tqdm
import scripts.sample.generate_dual_6d_grasp_poses as generate_dual_6d_grasp_poses 
import numpy as np
import random
import torch

def seed_all(seed=28):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    print(f"Seed set to {seed}")

def main():
    
    # seed_all()
    
    TEST_MESHES = '/scratch/dualarm/DG16M/test_final.txt'
    mesh_files = open(TEST_MESHES, 'r').readlines()
    mesh_files = [line.strip() for line in mesh_files if line.strip()]
    random.shuffle(mesh_files)
    mesh_files = mesh_files[:50]
    
    for mesh_name in tqdm(mesh_files, colour='blue'):
        mesh_name = os.path.join('/scratch/dualarm/DA2_15mar/meshes_scaled/', mesh_name)
        print(f"Processing mesh: {mesh_name}")
        
        generate_dual_6d_grasp_poses.main(
            get_args=False,
            input_dict={
                'n_grasps': 300,
                'input': mesh_name,
                'model': 'dual_arm_params',
                'device': 'cuda:0',
                'save_path': '/scratch/dualarm/collision_refined_7sep'
            }
        )
        
        
if __name__ == "__main__":
    main()
    
