import glob
import copy
import time

import numpy as np
import trimesh

from scipy.stats import special_ortho_group

import os
import torch

from torch.utils.data import DataLoader, Dataset
import json
import pickle
import h5py
from se3dif.utils import get_data_src

from se3dif.utils import to_numpy, to_torch, get_grasps_src
# from mesh_to_sdf.surface_point_cloud import get_scan_view, get_hq_scan_view
# from mesh_to_sdf.scan import ScanPointcloud
from tqdm import tqdm
from icecream import ic


import os, sys

import logging
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)
from scipy.spatial.transform import Rotation as R
from trimesh.collision import CollisionManager

class DG16MGrasps:
    def __init__(self, grasp_path, mesh_path, scales, single_arm=False):
        self.grasp_path = grasp_path
        self.mesh_path = mesh_path
        data = h5py.File(self.grasp_path, 'r')
        self.mesh_name = self.mesh_path.split('/')[-1].split('.')[0]
        self.mesh_scale = scales[self.mesh_name]
        self.single_arm = single_arm
        self.positive_negative = True
        
        if single_arm:
            self.grasps = self.load_grasps_for_collision_prediction(data, self.mesh_name)
        else:
            self.grasps = self.load_grasps(data)
        self.mesh = self.load_mesh()        
        
    def load_grasps_for_collision_prediction(self, data, mesh_name):
        positive_grasps = data['grasps/grasps'][()].reshape(-1, 4, 4)
        negative_grasps = np.load(f'/scratch/dualarm/DG16M/dg16m/negative_colliding_grasps/{mesh_name}.npy').reshape(-1, 4, 4)
        # if len(positive_grasps) > 5000:
        #     # print(f'{len(positive_grasps)} positive grasps for {mesh_name}, taking 5000')
        #     indices_to_take = np.random.choice(positive_grasps.shape[0], 5000, replace=False)
        #     positive_grasps = positive_grasps[indices_to_take]
        
        self.labels = np.concatenate((
            np.ones(positive_grasps.shape[0], dtype=np.float32),
            np.zeros(negative_grasps.shape[0], dtype=np.float32)
        ))
        
        # print(f'Loaded {positive_grasps.shape} positive grasps and {negative_grasps.shape} negative grasps for {mesh_name}')
        grasps = np.concatenate((positive_grasps, negative_grasps), axis=0)
        
        perm = np.random.permutation(len(grasps))
        grasps = grasps[perm]
        self.labels = self.labels[perm]
        return grasps
        

    def load_grasps(self, data):
        grasps = data['grasps/grasps'][()]
        if self.single_arm:
            grasps = grasps.reshape(-1, 4, 4)
            self.labels = np.ones(grasps.shape[0], dtype=np.float32)
            return grasps
        
        positive_indices = data['grasps/fc_passing_indices'][()]
        negative_indices = data['grasps/fc_failed_indices'][()]
        positive_grasps = grasps[positive_indices]
        negative_grasps = grasps[negative_indices]
        if self.positive_negative: 
            grasps = np.concatenate((positive_grasps, negative_grasps), axis=0)
            self.labels = np.concatenate((np.ones(positive_grasps.shape[0]), 
                                 np.zeros(negative_grasps.shape[0])), axis=0)
        else:
            grasps = grasps[positive_indices]
            self.labels = np.ones(positive_grasps.shape[0])
        return grasps
    
    def load_mesh(self):
        mesh = trimesh.load(self.mesh_path)
        # mesh.apply_translation(-mesh.centroid)
        mesh.apply_scale(self.mesh_scale)
        return mesh
    
class DG16MPointcloudSDFDataset(Dataset):
    def __init__(self, 
                 grasps_dir, 
                 meshes_dir, 
                 sdf_dir, 
                 meshes_to_take,
                 n_grasps=64, 
                 n_points=1000,
                 single_arm=False):
        super().__init__()
        self.n_grasps = n_grasps
        self.n_points = n_points
        self.grasps_dir = grasps_dir
        self.meshes_dir = meshes_dir
        self.sdf_dir = sdf_dir
        self.single_arm = single_arm
        self.meshes_to_take = meshes_to_take        
        # print(self.grasps_dir)
        
        # self.grasp_files = os.listdir(self.grasps_dir)
        self.grasp_files = [i.replace('.obj', '.h5') for i in self.meshes_to_take]
        self.grasp_objects = []
        
        # self.scales = json.load('./scales.json')
        with open('./scales.json', 'r') as f:
            self.scales = json.load(f)
        
        for grasp_file in tqdm(self.grasp_files, desc='Loading grasps'):
            if not grasp_file.endswith('.h5'):
                continue
            try:
                grasp_path = os.path.join(self.grasps_dir, grasp_file)
                mesh_path = os.path.join(self.meshes_dir, grasp_file.replace('.h5', '.obj'))
                # print(grasp_path, mesh_path)
                grasp_object = DG16MGrasps(grasp_path, mesh_path, self.scales, self.single_arm)
                if grasp_object.grasps.shape[0] > 0:
                    self.grasp_objects.append(grasp_object)
            except Exception as e:
                # print(f'Error loading {grasp_file} | {e}')
                continue
                                    
        np.random.shuffle(self.grasp_objects)
        
        # train_size = int(0.95 * len(self.grasp_objects))
        # test_size = len(self.grasp_objects) - train_size
        
        # self.train_grasp_objects, self.val_grasp_objects = torch.utils.data.random_split(self.grasp_objects, [train_size, test_size])
        # self.n_samples = len(self.train_grasp_objects)
        self.n_samples = len(self.grasp_objects)
        self.custom_scale = 8.0
        self.gripper = trimesh.load('./gripper.obj')
        
    def get_sdf(self, grasp_object):
        sdf_path = os.path.join(self.sdf_dir, grasp_object.mesh_name + '.json')
        with open(sdf_path, 'rb') as f:
            sdf_dict = pickle.load(f)
        
        loc = sdf_dict['loc']
        scale = sdf_dict['scale']
        sdf_points = (sdf_dict['xyz'] + loc) * scale * grasp_object.mesh_scale
        indices_to_take = np.random.choice(sdf_points.shape[0], self.n_points, replace=False)
        sdf_points = sdf_points[indices_to_take]
        sdf = sdf_dict['sdf'][indices_to_take] * scale * grasp_object.mesh_scale
        return sdf_points, sdf
        
    def __len__(self):
        return self.n_samples
   
    def create_random_T(self):
        T = np.eye(4)
        while True:
            rot = np.random.randn(3) * 15
            if np.linalg.norm(rot) > 10:
                T[:3, :3] = R.from_euler('xyz', rot, degrees=True).as_matrix()
                break
        # T[:3, 3] = np.random.uniform(-0.001, 0.001, size=3) * 8
        return T
    
    def collision_checker(self, mesh, grasp_left=None, grasp_right=None):
        cm = CollisionManager()
        cm.add_object('mesh', mesh)
        if grasp_left is not None:
            cm.add_object('gripper_left', grasp_left)
        if grasp_right is not None:
            cm.add_object('gripper_right', grasp_right)
        return cm.in_collision_internal()
      
    # def __getitem__(self, idx):
    #     grasp_object = self.grasp_objects[idx]
    #     pcd = grasp_object.mesh.sample(self.n_points) # [n_points, 3]
    #     grasps = grasp_object.grasps
    #     indices_to_take = np.random.choice(grasps.shape[0], self.n_grasps, replace=not grasps.shape[0] > self.n_grasps)
    #     grasps = grasps[indices_to_take] # [n_grasps, 2, 4, 4]
    #     labels = grasp_object.labels[indices_to_take] # [n_grasps]\
        
    def __getitem__(self, idx):
        grasp_object = self.grasp_objects[idx]
        pcd = grasp_object.mesh.sample(self.n_points) # [n_points, 3]
        grasps = grasp_object.grasps
        indices_to_take = np.random.choice(grasps.shape[0], self.n_grasps, replace=not grasps.shape[0] > self.n_grasps)
        grasps = grasps[indices_to_take] # [n_grasps, 2, 4, 4]
        labels = grasp_object.labels[indices_to_take] # [n_grasps]
        
        sdf_points, sdf = self.get_sdf(grasp_object)
        
        # scale everything by self.custom_scale
        pcd = pcd * self.custom_scale
        sdf_points = sdf_points * self.custom_scale
        sdf = sdf * self.custom_scale 
        grasps[..., :3, -1] = grasps[..., :3, -1] * self.custom_scale
        
        pcd_mean = np.mean(pcd, axis=0)
        pcd = pcd - pcd_mean
        sdf_points = sdf_points - pcd_mean
        # pcd = pcd - sdf_mean
        
        # random_R = special_ortho_group.rvs(3)
        random_R = R.from_euler('z', np.random.uniform(0, 2 * np.pi), degrees=False).as_matrix()
        # random_R = np.eye(3)
        random_T = np.eye(4)
        random_T[:3, :3] = random_R
        
        pcd = np.einsum('mn,bn->bm',random_R, pcd)
        sdf_points = np.einsum('mn,bn->bm',random_R, sdf_points)
        if self.single_arm:
            grasps = np.einsum('mn,bnk->bmk', random_T, grasps)
        else: 
            grasps = np.einsum('mn,bonk->bomk', random_T, grasps)
            
        # if self.single_arm:
        #     H_grasps = np.einsum('mn,bnk->bmk', H, H_grasps)
        # else:
        #     H_grasps = np.einsum('mn,bonk->bomk', H, H_grasps)
        
        
        res = {
            'visual_context': torch.from_numpy(pcd).float(), # B, 1000, 3
            'x_sdf': torch.from_numpy(sdf_points).float(), # B, 1000, 3
            'scale': torch.Tensor([self.custom_scale]).float(),
            'x_ene_pos': torch.from_numpy(grasps).float() # B, 64, 4, 4
        }
        
        gt = {
            'sdf': torch.from_numpy(sdf).float(),
            'labels': torch.from_numpy(labels).float(),
        }
        
        return res, gt
    
    def __getitem__2(self, idx):
        grasp_object = self.grasp_objects[idx]
        pcd = grasp_object.mesh.sample(self.n_points)  # [n_points, 3]
        grasps = grasp_object.grasps                   # [total_grasps, 2, 4, 4]
        labels = grasp_object.labels                   # [total_grasps], binary (0 or 1)
        # pcd = grasp_object.mesh.sample(self.n_points) # [n_points, 3]
        # grasps = grasp_object.grasps
        # indices_to_take = np.random.choice(grasps.shape[0], self.n_grasps, replace=not grasps.shape[0] > self.n_grasps)
        # grasps = grasps[indices_to_take] # [n_grasps, 2, 4, 4]
        # labels = grasp_object.labels[indices_to_take] # [n_grasps]
        # print(labels.sum(), labels.shape)

        # Separate indices for each class
        pos_indices = np.where(labels == 1)[0]
        neg_indices = np.where(labels == 0)[0]

        # Number of samples per class (half of n_grasps)
        n_each = self.n_grasps // 2

        # Adjust if there are fewer positives or negatives than n_each
        n_pos = min(n_each, len(pos_indices))
        n_neg = min(n_each, len(neg_indices))
        n_pos, n_neg = n_each, n_each

        # Sample from each class
        pos_sampled = np.random.choice(pos_indices, n_pos, replace=len(pos_indices) < n_pos)
        neg_sampled = np.random.choice(neg_indices, n_neg, replace=len(neg_indices) < n_neg)

        # Concatenate and shuffle
        final_indices = np.concatenate([pos_sampled, neg_sampled])
        np.random.shuffle(final_indices)

        grasps = grasps[final_indices]        # [n_selected_grasps, 2, 4, 4]
        labels = labels[final_indices]        # [n_selected_grasps]
        
        sdf_points, sdf = self.get_sdf(grasp_object)
        
        # scale everything by self.custom_scale
        pcd = pcd * self.custom_scale
        sdf_points = sdf_points * self.custom_scale
        sdf = sdf * self.custom_scale 
        grasps[..., :3, -1] = grasps[..., :3, -1] * self.custom_scale
        
        pcd_mean = np.mean(pcd, axis=0)
        pcd = pcd - pcd_mean
        sdf_points = sdf_points - pcd_mean
        # pcd = pcd - sdf_mean
        
        # random_R = special_ortho_group.rvs(3)
        random_R = R.from_euler('z', np.random.uniform(0, 2 * np.pi), degrees=False).as_matrix()
        # random_R = np.eye(3)
        random_T = np.eye(4)
        random_T[:3, :3] = random_R
        
        pcd = np.einsum('mn,bn->bm',random_R, pcd)
        sdf_points = np.einsum('mn,bn->bm',random_R, sdf_points)
        if self.single_arm:
            grasps = np.einsum('mn,bnk->bmk', random_T, grasps)
        else: 
            grasps = np.einsum('mn,bonk->bomk', random_T, grasps)
            
        # if self.single_arm:
        #     H_grasps = np.einsum('mn,bnk->bmk', H, H_grasps)
        # else:
        #     H_grasps = np.einsum('mn,bonk->bomk', H, H_grasps)
        

        if self.single_arm:
            res = {
                'visual_context': torch.from_numpy(pcd).float(), # B, 1000, 3
                'scale': torch.Tensor([self.custom_scale]).float(),
                'x_ene_pos': torch.from_numpy(grasps).float(), # B, 64, 4, 4
            }
            
            gt = {
                'labels': torch.from_numpy(labels).float(),
            }
            
        
        # if self.single_arm:
        #     gripper_mesh = self.gripper.copy()
        #     gripper_mesh = gripper_mesh.apply_scale(self.custom_scale)
        #     object_mesh = grasp_object.mesh.copy()
        #     object_mesh = object_mesh.apply_scale(self.custom_scale)
        #     object_mesh = object_mesh.apply_translation(-pcd_mean)
            
        #     labels = np.ones(grasps.shape[0], dtype=np.float32) 
        #     for i in range(grasps.shape[0]):
        #         grasp_now = grasps[i].copy()
        #         if np.random.rand(1) > 0.5:
        #             random_T = self.create_random_T()
        #             grasp_now[:3, :3] = grasp_now[:3, :3] @ random_T[:3, :3]
        #             grasp_now[:3, 3] = grasp_now[:3, 3] + grasp_now[:3, :3][:, 1] * np.random.rand(1)/3
        #             # gripper_now = gripper_mesh.copy().apply_transform(grasp_now)
        #             # if self.collision_checker(object_mesh, grasp_left=gripper_now):
        #             labels[i] = 0.0     
        #             grasps[i] = grasp_now   
                        
        #     res = {
        #         'visual_context': torch.from_numpy(pcd).float(), # B, 1000, 3
        #         'scale': torch.Tensor([self.custom_scale]).float(),
        #         'x_ene_pos': torch.from_numpy(grasps).float() # B, 64, 4, 4
        #     }
            
        #     gt = {
        #         'labels': torch.from_numpy(labels).float(),
        #     }
            
        else:
            res = {
                'visual_context': torch.from_numpy(pcd).float(), # B, 1000, 3
                'x_sdf': torch.from_numpy(sdf_points).float(), # B, 1000, 3
                'scale': torch.Tensor([self.custom_scale]).float(),
                'x_ene_pos': torch.from_numpy(grasps).float(), # B, 64, 4, 4
                # 'mesh_names': [grasp_object.mesh_name]
            }
            
            gt = {
                'sdf': torch.from_numpy(sdf).float(),
                'labels': torch.from_numpy(labels).float(),
            }
        
        return res, gt

class DA2Grasps():
    def __init__(self, filename, single_arm = True):
        self.filename = filename
        scale = None
        if filename.endswith(".json"):
            data = json.load(open(filename, "r"))
            self.mesh_fname = data["object"].decode('utf-8')
            self.mesh_type = self.mesh_fname.split('/')[1]
            self.mesh_id = self.mesh_fname.split('/')[-1].split('.')[0]
            self.mesh_scale = data["object_scale"] if scale is None else scale
        elif filename.endswith(".h5"):
            data = h5py.File(filename, "r")
            self.mesh_fname = 'meshes/' + data["object/file"][()].decode('utf-8')
            # self.mesh_type = self.mesh_fname.split('/')[1]
            self.mesh_id = self.mesh_fname.split('/')[-1].split('.')[0]
            self.mesh_scale = data["object/scale"][()] if scale is None else scale
        else:
            raise RuntimeError("Unknown file ending:", filename)

        self.grasps, self.success = self.load_grasps(filename)
        # dual_negatives_path = os.path.join('/raid/t1/scratch/grasp_dif/da2-positive-neg-1024/',os.path.basename(filename)[:-3]+'.pickle')
        # neg_grasp_file = pickle.load(open(dual_negatives_path,'rb'))
        # self.ng_index = np.array(neg_grasp_file['negative_pairs'])

        self.quality_to_class()

        good_idxs = np.argwhere(self.success>=0.5)[:,0]
        bad_idxs  = np.argwhere(self.success<0.5)[:,0]
        self.good_grasps = self.grasps[good_idxs,...]
        self.bad_grasps  = self.grasps[bad_idxs,...]
        if single_arm:
            self.good_grasps = self.good_grasps.reshape(-1,4,4)
            self.bad_grasps = self.bad_grasps.reshape(-1,4,4)
    
    def quality_to_class(self):
        bin_boundaries = [0, 0.85, 1+1e-5]
        self.bin_indices = np.digitize(self.success, bin_boundaries) - 1

    def load_grasps(self, filename):
        """Load transformations and qualities of grasps from a JSON file from the dataset.

        Args:
            filename (str): HDF5 or JSON file name.

        Returns:
            np.ndarray: Homogenous matrices describing the grasp poses. 2000 x 4 x 4.
            np.ndarray: List of binary values indicating grasp success in simulation.
        """
        if filename.endswith(".json"):
            data = json.load(open(filename, "r"))
            T = np.array(data["transforms"])
            success = np.array(data["quality_flex_object_in_gripper"])
        elif filename.endswith(".h5"):
            data = h5py.File(filename, "r")
            T = np.array(data["grasps/transforms"])
            success = \
                0.5 * np.array(data["grasps/qualities/Force_closure"]) + \
                0.4 * np.array(data["grasps/qualities/Dexterity"]) + \
                0.1 * np.array(data["grasps/qualities/Torque_optimization"])
        else:
            raise RuntimeError("Unknown file ending:", filename)
        return T, success
    
    def load_mesh(self):
        mesh_path_file = os.path.join(get_data_src(), self.mesh_fname)

        mesh = trimesh.load(mesh_path_file,  file_type='obj', force='mesh')

        mesh.apply_scale(self.mesh_scale)
        if type(mesh) == trimesh.scene.scene.Scene:
            mesh = trimesh.util.concatenate(mesh.dump())
        return mesh


class DA2GraspsDirectory():
    def __init__(self, filename=get_grasps_src(), data_type='Mug', single_arm=True):

        self.grasps_files = sorted(glob.glob(filename + '/*.h5'))

        self.avail_obj = []
        for grasp_file in self.grasps_files:
            self.avail_obj.append(DA2Grasps(grasp_file, single_arm=single_arm))


class DA2AndSDFDataset(Dataset):
    'DataLoader for training DeepSDF Auto-Decoder model'
    def __init__(self, class_type='Mug', se3=False, phase='train', one_object=False,
                 n_pointcloud = 1000, n_density = 200, n_coords = 1500,
                 augmented_rotation=True, visualize=False, split = True , exp_feature=False):

        self.class_type = class_type
        self.data_dir = get_data_src()
        self.DA2_data_dir = self.data_dir

        self.grasps_dir = os.path.join(self.DA2_data_dir, 'grasps')
        self.sdf_dir = os.path.join(self.DA2_data_dir, 'sdf')

        self.generated_points_dir = os.path.join(self.DA2_data_dir, 'train_data')
        
        self.exp_feature = exp_feature
        
        grasps_files = sorted(glob.glob(self.grasps_dir+'/'+class_type+'/*.h5'))
        

        points_files = []
        sdf_files = []
        for grasp_file in grasps_files:
            g_obj = DA2Grasps(grasp_file)
            mesh_file = g_obj.mesh_fname
            txt_split = mesh_file.split('/')

            sdf_file = os.path.join(self.sdf_dir, class_type, txt_split[-1].split('.')[0]+'.json')
            point_file = os.path.join(self.generated_points_dir, class_type, '4_points', txt_split[-1]+'.npz')

            sdf_files.append(sdf_file)
            points_files.append(point_file)

        ## Split Train/Validation
        n = len(grasps_files)
        indexes = np.arange(0, n)
        self.total_len = n
        if split:
            idx = int(0.9 * n)
        else:
            idx = int(n)

        if phase == 'train':
            self.grasp_files = grasps_files[:idx]
            self.points_files = points_files[:idx]
            self.sdf_files = sdf_files[:idx]
            self.indexes = indexes[:idx]
        else:
            self.grasp_files = grasps_files[idx:]
            self.points_files = points_files[idx:]
            self.sdf_files = sdf_files[idx:]
            self.indexes = indexes[idx:]


        self.len = len(self.points_files)

        self.n_pointcloud = n_pointcloud
        self.n_density  = n_density
        self.n_occ = n_coords

        ## Variables on Data
        self.one_object = one_object
        self.augmented_rotation = augmented_rotation
        self.se3 = se3

        ## Visualization
        self.visualize = visualize
        self.scale = 8.

    def __len__(self):
        return self.len

    def _get_item(self, index):
        if self.one_object:
            index = 0

        index_obj = self.indexes[index]
        ## Load Files ##
        grasps_obj = DA2Grasps(self.grasp_files[index])
        sdf_file = self.sdf_files[index]
        with open(sdf_file, 'rb') as handle:
            sdf_dict = pickle.load(handle)

        ## PointCloud
        p_clouds = sdf_dict['pcl']
        rix = np.random.permutation(p_clouds.shape[0])
        p_clouds = p_clouds[rix[:self.n_pointcloud],:]

        ## Coordinates XYZ
        coords  = sdf_dict['xyz']
        rix = np.random.permutation(coords.shape[0])
        coords = coords[rix[:self.n_occ],:]

        ### SDF value
        sdf = sdf_dict['sdf'][rix[:self.n_occ]]
        grad_sdf = sdf_dict['grad_sdf'][rix[:self.n_occ], ...]

        ### Scale and Loc
        scale = sdf_dict['scale']
        loc = sdf_dict['loc']

        ## Grasps good/bad
        rix = np.random.randint(low=0, high=grasps_obj.good_grasps.shape[0], size=self.n_density)
        H_grasps = grasps_obj.good_grasps[rix, ...]
        rix = np.random.randint(low=0, high=grasps_obj.bad_grasps.shape[0], size=self.n_density)
        H_bad_grasps = grasps_obj.bad_grasps[rix, ...]

        ## Rescale Pointcloud and Occupancy Points ##
        coords = (coords + loc)*scale*grasps_obj.mesh_scale * self.scale
        p_clouds = (p_clouds + loc)*scale*grasps_obj.mesh_scale * self.scale

        sdf = sdf*scale*grasps_obj.mesh_scale * self.scale
        grad_sdf = -grad_sdf*scale*grasps_obj.mesh_scale * self.scale

        H_grasps[:,:-1,-1] = H_grasps[:,:-1,-1] * self.scale
        H_bad_grasps[:,:-1,-1] = H_bad_grasps[:,:-1,-1]*self.scale

        ## Random rotation ##
        if self.augmented_rotation:
            R = special_ortho_group.rvs(3)
            H = np.eye(4)
            H[:3,:3] = R

            coords = np.einsum('mn,bn->bm',R, coords)
            p_clouds = np.einsum('mn,bn->bm',R, p_clouds)

            H_grasps = np.einsum('mn,bnd->bmd', H, H_grasps)
            H_bad_grasps = np.einsum('mn,bnd->bmd', H, H_bad_grasps)

            grad_sdf = np.einsum('mn,bn->bm', R, grad_sdf)


        # Visualize
        if self.visualize:
            ## 3D matplotlib ##
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(p_clouds[:,0], p_clouds[:,1], p_clouds[:,2], c='r')

            n = 10
            x = coords[:n,:]
            ## grad sdf ##
            x_grad = grad_sdf[:n, :]
            ax.quiver(x[:,0], x[:,1], x[:,2], x_grad[:,0], x_grad[:,1], x_grad[:,2], length=0.3)

            ## sdf visualization ##
            x_sdf = sdf[:n]
            x_sdf = 0.9*x_sdf/np.max(x_sdf)
            c = np.zeros((n, 3))
            c[:, 1] = x_sdf
            ax.scatter(x[:,0], x[:,1], x[:,2], c=c)

            plt.show(block=True)

        del sdf_dict
        
        
        if self.exp_feature:
            num_of_grasps = 200
            dist_threshold = 0.6 
            gripper_pos = np.expand_dims(H_grasps[:num_of_grasps,:3,3],axis=0) # [num_of_grasps,3]
            gripper_pos = np.expand_dims(gripper_pos,axis=1) # [num_of_grasps,3] --> [num_of_grasps,1,3] 
            
            pts = np.expand_dims(p_clouds,axis=0) # [num_of_pts,3] --> [1,num_of_pts,3]
            diff_btw_g2pt = gripper_pos - pts # [num_of_grasps,num_of_pts,3]
            
            dist_map = np.sqrt(np.expand_dims((diff_btw_g2pt**2).sum(axis=2),axis=-1)) # [num_of_grasp,num_of_pts,1] euclidean metric
            
            normalized_dist_map = (dist_map-dist_map.min(axis=0))/(dist_map.max(axis=0)-dist_map.min(axis=0)) # [num_of_grasp,num_of_pts,1] value in range [0,1].
            normalized_dist_map = np.abs(normalized_dist_map - 1.) # np.abs(nomalized_dist_map -1.) this is now a closeness map in range [0,1] i.e. close pts wrt to grasp will have value close to 1.
            
            closeness_mask = np.zeros_like(normalized_dist_map)
            closeness_mask[np.where(normalized_dist_map>dist_threshold)] = 1.
            
            res = {'point_cloud': torch.from_numpy(p_clouds).float(),
               'x_sdf': torch.from_numpy(coords).float(),
               'x_ene_pos': torch.from_numpy(H_grasps).float(),
               'x_neg_ene': torch.from_numpy(H_bad_grasps).float(),
               'scale': torch.Tensor([self.scale]).float(),
               'visual_context':  torch.Tensor([index_obj]),
               'closeness_score': torch.from_numpy(normalized_dist_map).float(),
               'closeness_mask': torch.from_numpy(closeness_mask).float()}
        else:
            res = {'point_cloud': torch.from_numpy(p_clouds).float(),
                'x_sdf': torch.from_numpy(coords).float(),
                'x_ene_pos': torch.from_numpy(H_grasps).float(),
                'x_neg_ene': torch.from_numpy(H_bad_grasps).float(),
                'scale': torch.Tensor([self.scale]).float(),
                'visual_context':  torch.Tensor([index_obj])}

        return res, {'sdf': torch.from_numpy(sdf).float(), 'grad_sdf': torch.from_numpy(grad_sdf).float()}

    def __getitem__(self, index):
        'Generates one sample of data'
        return self._get_item(index)


class PointcloudDA2AndSDFDataset(Dataset):
    'DataLoader for training DeepSDF with a Rotation Invariant Encoder model'
    def __init__(self, class_type=['Cup', 'Mug', 'Fork', 'Hat', 'Bottle', 'Bowl', 'Car', 'Donut', 'Laptop', 'MousePad', 'Pencil',
                                   'Plate', 'ScrewDriver', 'WineBottle','Backpack', 'Bag', 'Banana', 'Battery', 'BeanBag', 'Bear',
                                   'Book', 'Books', 'Camera','CerealBox', 'Cookie','Hammer', 'Hanger', 'Knife', 'MilkCarton', 'Painting',
                                   'PillBottle', 'Plant','PowerSocket', 'PowerStrip', 'PS3', 'PSP', 'Ring', 'Scissors', 'Shampoo', 'Shoes',
                                   'Sheep', 'Shower', 'Sink', 'SoapBottle', 'SodaCan','Spoon', 'Statue', 'Teacup', 'Teapot', 'ToiletPaper',
                                   'ToyFigure', 'Wallet','WineGlass',
                                   'Cow', 'Sheep', 'Cat', 'Dog', 'Pizza', 'Elephant', 'Donkey', 'RubiksCube', 'Tank', 'Truck', 'USBStick'],
                 se3=False, phase='train', one_object=False,
                 n_pointcloud = 1000, n_density = 200, n_coords = 1000,
                 augmented_rotation=True, visualize=False, single_arm=True, split = True, exp_feature=False):

        #class_type = ['Mug']
        self.class_type = class_type
        self.single_arm = single_arm
        self.exp_feature = exp_feature
        self.data_dir = get_data_src()

        self.grasps_dir = os.path.join(self.data_dir, 'grasps')
        # self.grasps_dir = '/scratch/dualarm/DA2_15mar/grasps/'

        self.grasp_files = []
        # for class_type_i in class_type:
        cls_grasps_files = sorted(glob.glob(self.grasps_dir+'/*.h5'))
        print(self.grasps_dir)

        for grasp_file in tqdm(cls_grasps_files):
            g_obj = DA2Grasps(grasp_file, single_arm=single_arm)

            ## Grasp File ##
            if g_obj.good_grasps.shape[0] > 0:
                self.grasp_files.append(grasp_file)

        ## Split Train/Validation
        n = len(self.grasp_files)
        train_size = int(n*0.9)
        test_size  =  n - train_size

        self.train_grasp_files, self.test_grasp_files = torch.utils.data.random_split(self.grasp_files, [train_size, test_size])
        
        # print(self.train_grasp_files.indices)
        # print(self.test_grasp_files.indices)
        # exit()

        self.type = 'train'
        self.len = len(self.train_grasp_files)

        self.n_pointcloud = n_pointcloud
        self.n_density  = n_density
        self.n_occ = n_coords

        ## Variables on Data
        self.one_object = one_object
        self.augmented_rotation = augmented_rotation
        self.se3 = se3

        ## Visualization
        self.visualize = visualize
        self.scale = 8.

    def __len__(self):
        return self.len

    def set_test_data(self):
        self.len = len(self.test_grasp_files)
        self.type = 'test'

    def _get_grasps(self, grasp_obj):
        try:
            rix = np.random.randint(low=0, high=grasp_obj.good_grasps.shape[0], size=self.n_density)
        except:
            print('lets see')
        H_grasps = grasp_obj.good_grasps[rix, ...]
        # H_grasps = grasp_obj.good_grasps[:200, ...]
        return H_grasps

    def _get_sdf(self, grasp_obj, grasp_file):

        mesh_fname = grasp_obj.mesh_fname
        mesh_scale = grasp_obj.mesh_scale

        mesh_name = mesh_fname.split('/')[-1]
        filename  = mesh_name.split('.obj')[0]
        sdf_file = os.path.join(self.data_dir, 'sdf', filename+'.json')

        with open(sdf_file, 'rb') as handle:
            sdf_dict = pickle.load(handle)

        loc = sdf_dict['loc']
        scale = sdf_dict['scale']
        xyz = (sdf_dict['xyz'] + loc)*scale*mesh_scale
        rix = np.random.permutation(xyz.shape[0])
        xyz = xyz[rix[:self.n_occ], :]
        sdf = sdf_dict['sdf'][rix[:self.n_occ]]*scale*mesh_scale
        return xyz, sdf

    def _get_mesh_pcl(self, grasp_obj):
        mesh = grasp_obj.load_mesh()
        return mesh.sample(self.n_pointcloud)

    def _get_grasps_qualities(self, grasp_obj):
        try:
            rix = np.random.randint(low=0, high=grasp_obj.grasps.shape[0], size=self.n_density * 3 // 4)
        except:
            print('lets see')
        H_grasps = grasp_obj.grasps[rix, ...]
        score = grasp_obj.bin_indices[rix, ...]
        # hard_negative_grasps = grasp_obj.ng_index
        # H_negative_grasps = grasp_obj.grasps.reshape(-1,4,4)[hard_negative_grasps, ...]
        # try:
        #     rix = np.random.randint(low=0, high=H_negative_grasps.shape[0], size=self.n_density // 4)
        # except:
        #     print('lets see')
        # H_negative_grasps = H_negative_grasps[rix, ...]
        # hard_ng_scores = np.zeros((H_negative_grasps.shape[0],1)).astype(np.int64)
        # H_grasps = np.concatenate([H_grasps,H_negative_grasps],axis=0)
        # score = np.concatenate([score,hard_ng_scores],axis=0)
        
        # print(H_grasps.shape, score.shape)
        # exit()
        
        return H_grasps, score

    def _get_item(self, index):
        if self.one_object:
            index = 0

        ## Load Files ##
        if self.type == 'train':
            grasps_obj = DA2Grasps(self.train_grasp_files[index], single_arm=self.single_arm)
        else:
            grasps_obj = DA2Grasps(self.test_grasp_files[index], single_arm=self.single_arm)

        ## SDF
        xyz, sdf = self._get_sdf(grasps_obj, self.train_grasp_files[index])

        ## PointCloud
        pcl = self._get_mesh_pcl(grasps_obj)

        ## Grasps good/bad
        H_grasps = self._get_grasps(grasps_obj)

        ## Quality metrics
        # qual_g, score = self._get_grasps_qualities(grasps_obj)
        ## rescale, rotate and translate ##
        xyz = xyz*self.scale
        sdf = sdf*self.scale
        pcl = pcl*self.scale
        H_grasps[..., :3, -1] = H_grasps[..., :3, -1]*self.scale
        # qual_g[..., :3, -1] = qual_g[..., :3, -1]*self.scale
        ## Random rotation ##
        R = special_ortho_group.rvs(3)
        H = np.eye(4)
        H[:3, :3] = R
        mean = np.mean(pcl, 0)
        ## translate ##
        xyz = xyz - mean
        pcl = pcl - mean
        H_grasps[..., :3, -1] = H_grasps[..., :3, -1] #- mean
        ## rotate ##
        pcl = np.einsum('mn,bn->bm',R, pcl)
        xyz = np.einsum('mn,bn->bm',R, xyz)
        # ic(H_grasps.shape, H.shape)
        if self.single_arm:
            H_grasps = np.einsum('mn,bnk->bmk', H, H_grasps)
            # qual_g = np.einsum('mn,bnk->bmk', H, qual_g)
        else:
            H_grasps = np.einsum('mn,bonk->bomk', H, H_grasps)
            # qual_g = np.einsum('mn,bonk->bomk', H, qual_g)
        #######################

    
        res = {'visual_context': torch.from_numpy(pcl).float(),
            'x_sdf': torch.from_numpy(xyz).float(),
            'x_ene_pos': torch.from_numpy(H_grasps).float(),
            # 'qual_g': torch.from_numpy(qual_g).float(),
            'scale': torch.Tensor([self.scale]).float()}
        gt = {'sdf': torch.from_numpy(sdf).float(),
            #   'score': torch.from_numpy(score).long()
                }

        return res, gt

    def __getitem__(self, index):
        'Generates one sample of data'
        # index = 1212
        # print(index)
        return self._get_item(index)


class PartialPointcloudDA2AndSDFDataset(Dataset):
    'DataLoader for training DeepSDF with a Rotation Invariant Encoder model'
    def __init__(self, class_type=['Cup', 'Mug', 'Fork', 'Hat', 'Bottle'],
                 se3=False, phase='train', one_object=False,
                 n_pointcloud = 1000, n_density = 200, n_coords = 1000, augmented_rotation=True, visualize=False, single_arm=True):

        #class_type = ['Mug']
        self.class_type = class_type
        self.single_arm = single_arm
        self.data_dir = get_data_src()

        self.grasps_dir = os.path.join(self.data_dir, 'grasps')

        self.grasp_files = []
        # for class_type_i in class_type:
        cls_grasps_files = sorted(glob.glob(self.grasps_dir+'/*.h5'))

        for grasp_file in cls_grasps_files:
            g_obj = DA2Grasps(grasp_file, single_arm=single_arm)

            ## Grasp File ##
            if g_obj.good_grasps.shape[0] > 0:
                self.grasp_files.append(grasp_file)

        ## Split Train/Validation
        n = len(self.grasp_files)
        train_size = int(n*0.9)
        test_size  =  n - train_size

        self.train_grasp_files, self.test_grasp_files = torch.utils.data.random_split(self.grasp_files, [train_size, test_size])
        
        print(len(self.train_grasp_files))
        print(self.test_grasp_files.indices)
        # exit()

        self.type = 'train'
        self.len = len(self.train_grasp_files)

        self.n_pointcloud = n_pointcloud
        self.n_density  = n_density
        self.n_occ = n_coords

        ## Variables on Data
        self.one_object = one_object
        self.augmented_rotation = augmented_rotation
        self.se3 = se3

        ## Visualization
        self.visualize = visualize
        self.scale = 8.

        ## Sampler
        self.scan_pointcloud = None

    def __len__(self):
        return self.len

    def set_test_data(self):
        self.len = len(self.test_grasp_files)
        self.type = 'test'

    def _get_grasps(self, grasp_obj):
        try:
            rix = np.random.randint(low=0, high=grasp_obj.good_grasps.shape[0], size=self.n_density)
        except:
            print('lets see')
        H_grasps = grasp_obj.good_grasps[rix, ...]
        return H_grasps

    def _get_sdf(self, grasp_obj, grasp_file):

        mesh_fname = grasp_obj.mesh_fname
        mesh_scale = grasp_obj.mesh_scale

        mesh_name = mesh_fname.split('/')[-1]
        filename  = mesh_name.split('.obj')[0]
        sdf_file = os.path.join(self.data_dir, 'sdf', filename+'.json')

        with open(sdf_file, 'rb') as handle:
            sdf_dict = pickle.load(handle)

        loc = sdf_dict['loc']
        scale = sdf_dict['scale']
        xyz = (sdf_dict['xyz'] + loc)*scale*mesh_scale
        rix = np.random.permutation(xyz.shape[0])
        xyz = xyz[rix[:self.n_occ], :]
        sdf = sdf_dict['sdf'][rix[:self.n_occ]]*scale*mesh_scale
        return xyz, sdf


    def _get_mesh_pcl(self, grasp_obj):
        mesh = grasp_obj.load_mesh()
        ## 1. Mesh Centroid ##
        centroid = mesh.centroid
        H = np.eye(4)
        H[:3, -1] = -centroid
        mesh.apply_transform(H)
        ######################
        #time0 = time.time()
        P = self.scan_pointcloud.get_hq_scan_view(mesh)
        #print('Sample takes {} s'.format(time.time() - time0))
        P +=centroid
        try:
            rix = np.random.randint(low=0, high=P.shape[0], size=self.n_pointcloud)
        except:
            print('here')
        return P[rix, :]

    def _get_item(self, index):
        if self.one_object:
            index = 0

        ## Load Files ##
        if self.type == 'train':
            grasps_obj = DA2Grasps(self.train_grasp_files[index], single_arm=self.single_arm)
        else:
            grasps_obj = DA2Grasps(self.test_grasp_files[index], single_arm=self.single_arm)

        ## SDF
        xyz, sdf = self._get_sdf(grasps_obj, self.train_grasp_files[index])

        ## PointCloud
        pcl = self._get_mesh_pcl(grasps_obj)

        ## Grasps good/bad
        H_grasps = self._get_grasps(grasps_obj)

        ## rescale, rotate and translate ##
        xyz = xyz*self.scale
        sdf = sdf*self.scale
        pcl = pcl*self.scale
        H_grasps[..., :3, -1] = H_grasps[..., :3, -1]*self.scale
        ## Random rotation ##
        R = special_ortho_group.rvs(3)
        H = np.eye(4)
        H[:3, :3] = R
        mean = np.mean(pcl, 0)
        ## translate ##
        xyz = xyz - mean
        pcl = pcl - mean
        H_grasps[..., :3, -1] = H_grasps[..., :3, -1]
        ## rotate ##
        pcl = np.einsum('mn,bn->bm',R, pcl)
        xyz = np.einsum('mn,bn->bm',R, xyz)
        if self.single_arm:
            H_grasps = np.einsum('mn,bnk->bmk', H, H_grasps)
            # qual_g = np.einsum('mn,bnk->bmk', H, qual_g)
        else:
            H_grasps = np.einsum('mn,bonk->bomk', H, H_grasps)
            # qual_g = np.einsum('mn,bonk->bomk', H, qual_g)
        #######################

        # Visualize
        if self.visualize:

            ## 3D matplotlib ##
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(pcl[:,0], pcl[:,1], pcl[:,2], c='r')

            x_grasps = H_grasps[..., :3, -1]
            ax.scatter(x_grasps[:,0], x_grasps[:,1], x_grasps[:,2], c='b')

            ## sdf visualization ##
            n = 100
            x = xyz[:n,:]

            x_sdf = sdf[:n]
            x_sdf = 0.9*x_sdf/np.max(x_sdf)
            c = np.zeros((n, 3))
            c[:, 1] = x_sdf
            ax.scatter(x[:,0], x[:,1], x[:,2], c=c)

            plt.show()
            #plt.show(block=True)

        res = {'visual_context': torch.from_numpy(pcl).float(),
               'x_sdf': torch.from_numpy(xyz).float(),
               'x_ene_pos': torch.from_numpy(H_grasps).float(),
               'scale': torch.Tensor([self.scale]).float()}
        # for key in res.keys():
        #     print(key, res[key].shape)
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(to_numpy(res['visual_context']))
        # o3d.io.write_point_cloud('partial_test.pcd', pcd)
        # exit()

        return res, {'sdf': torch.from_numpy(sdf).float()}

    def __getitem__(self, index):
        'Generates one sample of data'
        return self._get_item(index)


if __name__ == '__main__':
    from se3dif.utils.torch_utils import seed_everything
    seed_everything()
    ## Index conditioned dataset
    # dataset = DA2AndSDFDataset(visualize=True, augmented_rotation=True, one_object=False)

    ## Pointcloud conditioned dataset
    # dataset = PointcloudDA2AndSDFDataset(visualize=True, augmented_rotation=True, one_object=False, single_arm=False, exp_feature=True, n_pointcloud=4096)

    dataset = DA2GraspsDirectory(single_arm=False)
    print(dataset.avail_obj[0])
