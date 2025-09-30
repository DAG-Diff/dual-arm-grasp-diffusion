from se3dif.datasets.dg16m_dataset import DG16MPointcloudSDFDataset
from torch.utils.data import DataLoader
from se3dif.utils import load_experiment_specifications
from se3dif import summaries, trainer
from se3dif.losses.main import get_losses
import copy
from se3dif.models import loader
import torch
import torch.optim as optim
import os
import numpy as np
import random
import torch.nn as nn
from torchinfo import summary as model_summary
import shutil

def seed_all(seed=28):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    
    seed_all(56)
    
    spec_file = './configs/'
    args = load_experiment_specifications(spec_file, load_yaml='dual_arm_params')
    
    grasps_dir = '/scratch/dualarm/DG16M/dg16m/grasps'
    meshes_dir = '/scratch/dualarm/DA2_15mar/meshes'
    sdf_dir = '/scratch/dualarm/DA2_15mar/sdf'
    train_meshes_to_take = open('/scratch/dualarm/DG16M/train_final.txt').readlines()
    train_meshes_to_take = [mesh.strip() for mesh in train_meshes_to_take][:100]
    
    val_meshes_to_take = open('/scratch/dualarm/DG16M/test_final.txt').readlines()
    val_meshes_to_take = [mesh.strip() for mesh in val_meshes_to_take][:100]
    
    train_dataset = DG16MPointcloudSDFDataset(
        grasps_dir=grasps_dir,
        meshes_dir=meshes_dir,
        sdf_dir=sdf_dir,
        single_arm=args['single_arm'],
        meshes_to_take=train_meshes_to_take,
        n_points = args['num_input_points'],
        n_grasps=args['num_input_grasps']
    )
    
    val_dataset = DG16MPointcloudSDFDataset(
        grasps_dir=grasps_dir,
        meshes_dir=meshes_dir,
        sdf_dir=sdf_dir,
        single_arm=args['single_arm'],
        meshes_to_take=val_meshes_to_take,
        n_points = args['num_input_points'],
        n_grasps=args['num_input_grasps'],
    )
    
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=args['TrainSpecs']['batch_size'], 
                                  shuffle=True, 
                                  num_workers=12)
    val_loader = DataLoader(val_dataset,
                            batch_size=5, 
                            shuffle=False, 
                            num_workers=12)
    
    exp_dir = os.path.join('.', args['exp_log_dir'])
    args['saving_folder'] = exp_dir
    
    data = next(iter(train_dataloader))
    res, gt = next(iter(train_dataloader))
    
    print(gt['labels'].sum(-1))
    
    for k, v in res.items():
        print(k, v.shape)
    for k, v in gt.items():
        print(k, v.shape)
    
    device = 'cuda'
    args['device'] = device
    model = loader.load_model(args)
    
    loss = get_losses(args)
    loss_fn = val_loss_fn = loss.loss_fn
    summary = summaries.get_summary(args)
    lr = args['learning_rate']
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print('Number of parameters:', count_params(model))
    
    model_summary(model, col_names=['num_params', 'trainable'])
        
    trainer.train(model=model.float(), train_dataloader=train_dataloader, epochs=args['TrainSpecs']['num_epochs'], 
                  model_dir= exp_dir, summary_fn=summary, device=device, lr=lr, optimizers=[optimizer],
                steps_til_summary=args['TrainSpecs']['steps_til_summary'],
                epochs_til_checkpoint=args['TrainSpecs']['epochs_til_checkpoint'],
                loss_fn=loss_fn, iters_til_checkpoint=args['TrainSpecs']['iters_til_checkpoint'],
                clip_grad=False, val_loss_fn=val_loss_fn, overwrite=True,
                val_dataloader=val_loader, args=args)
    
if __name__ == '__main__':
    main()