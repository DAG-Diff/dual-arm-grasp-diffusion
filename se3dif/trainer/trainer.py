import os
import time
import datetime
import numpy as np
import torch

from collections import defaultdict

from se3dif.utils import makedirs, dict_to_device
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import wandb


def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn=None, iters_til_checkpoint=None, val_dataloader=None, clip_grad=False, val_loss_fn=None,
          overwrite=True, optimizers=None, batches_per_validation=10,  rank=0, max_steps=None, device='cpu',
          args=None):
    
    val_epoch_interval = 5
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizers[0], step_size=40, gamma=0.8)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizers[0], milestones=[40, 100, 180, 300], gamma=0.75)

    if optimizers is None:
        optimizers = [torch.optim.Adam(lr=lr, params=model.parameters())]

    if val_dataloader is not None:
        assert val_loss_fn is not None, "If validation set is passed, have to pass a validation loss_fn!"

    ## Build saving directories
    makedirs(model_dir)
    
    if args['use_wandb']:
        logger = wandb.init(
            project="dual-cgdf",
            config=args
        )
    else:
        logger = None
        
    if rank == 0:
        summaries_dir = os.path.join(model_dir, 'summaries')
        makedirs(summaries_dir)

        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        makedirs(checkpoints_dir)

        exp_name = datetime.datetime.now().strftime("%m.%d.%Y %H:%M:%S")
        writer = SummaryWriter(summaries_dir+ '/' + exp_name)

    total_steps = -1
    val_steps = -1 
    
    model.train()
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        epoch_losses = []
        for epoch in range(epochs):
            if not (epoch + 1) % epochs_til_checkpoint and epoch and rank == 0:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                           np.array(train_losses))
            
            print(f'Running epoch {epoch}')
            classifier_accuracies, collision_classifier_accuracies = [], []
            sdf_losses, denoising_losses, classifier_losses, collision_classifier_losses = [], [], [], []
            for step, (model_input, gt) in enumerate(train_dataloader):         
                for optim in optimizers:
                    optim.zero_grad()
                    
                total_steps += 1
                model_input = dict_to_device(model_input, device)
                gt = dict_to_device(gt, device)

                start_time = time.time()

                losses, iter_info = loss_fn(model, model_input, gt)

                train_loss = 0.
                
                for loss_name, loss in losses.items():
                    if loss_name == 'Classifier Accuracy':
                        classifier_accuracies.append(loss)
                        continue
                    if loss_name == 'Collision Classifier Accuracy':
                        collision_classifier_accuracies.append(loss)
                        continue
                    
                    single_loss = loss.mean()

                    if rank == 0:
                        writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss
                    if loss_name == 'sdf':
                        sdf_losses.append(single_loss.item())
                    elif loss_name == 'Dual Score loss':
                        denoising_losses.append(single_loss.item())
                    elif loss_name == 'Classifier Loss':
                        classifier_losses.append(single_loss.item())
                    elif loss_name == 'Collision Classifier Loss':
                        collision_classifier_losses.append(single_loss.item())

                # train_losses.append(train_loss.item())
                epoch_losses.append(train_loss.item())
                if rank == 0:
                    writer.add_scalar("total_train_loss", train_loss, total_steps)
                if not (total_steps + 1)% steps_til_summary and rank == 0:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    if summary_fn is not None:
                        summary_fn(model, model_input, gt, iter_info, writer, total_steps)
                    print("Epoch: %d, Step: %d, sdf_loss: %0.6f, denoise_loss: %0.6f, classifier_loss: %0.6f, classifier_acc: %0.6f, coll_loss: %0.6f, coll_acc: %0.6f" % (epoch, 
                                                   total_steps, 
                                                    np.mean(sdf_losses),
                                                    np.mean(denoising_losses),
                                                    np.mean(classifier_losses),
                                                    np.mean(classifier_accuracies),
                                                    np.mean(collision_classifier_losses),
                                                    np.mean(collision_classifier_accuracies)))
                        
                    if logger: 
                        logger.log(
                            {
                                # "epoch": epoch,
                                "sdf_loss": np.mean(sdf_losses),
                                "denoise_loss": np.mean(denoising_losses),
                                "classifier_loss": np.mean(classifier_losses),
                                "classifier_acc": np.mean(classifier_accuracies),
                            })
                    epoch_losses = []
                    sdf_losses, denoising_losses, classifier_losses, collision_classifier_losses =  [], [], [], []
                    classifier_accuracies, collision_classifier_accuracies = [], []
                    
                train_loss.backward()

                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                for optim in optimizers:
                    optim.step()

                if rank == 0:
                    pbar.update(1)
                
            if (epoch + 1) % val_epoch_interval == 0:
                if val_dataloader is None:
                    print('No validation set passed, skipping validation')
                    continue
                
                print("Running validation set...")
                model.eval()
                val_sdf_losses, val_denoising_losses, val_classifier_losses, val_collision_losses = [], [], [], []
                val_classifier_accuracies, val_collision_accuracies = [], []
                
                for val_step, (model_input, gt) in enumerate(val_dataloader):
                    val_steps += 1  
                    model_input = dict_to_device(model_input, device)
                    gt = dict_to_device(gt, device)
                    
                    with torch.no_grad():
                        losses, iter_info = loss_fn(model, model_input, gt)

                    val_loss = 0.
                    
                    for loss_name, loss in losses.items():
                        if loss_name == 'Classifier Accuracy':
                            val_classifier_accuracies.append(loss)
                            continue
                        if loss_name == 'Collision Classifier Accuracy':
                            val_collision_accuracies.append(loss)
                            continue
                        
                        single_loss = loss.mean()

                        if rank == 0:
                            writer.add_scalar(loss_name, single_loss, total_steps)
                        val_loss += single_loss
                        if loss_name == 'sdf':
                            val_sdf_losses.append(single_loss.item())
                        elif loss_name == 'Dual Score loss':
                            val_denoising_losses.append(single_loss.item())
                        elif loss_name == 'Classifier Loss':
                            val_classifier_losses.append(single_loss.item())
                        elif loss_name == 'Collsiion Classifier Loss':
                            val_collision_losses.append(single_loss.item())
                            
                    if (val_step + 1) % 20 == 0:
                        print("Epoch: %d, Step: %d, sdf_loss: %0.6f, denoise_loss: %0.6f, classifier_loss: %0.6f, classifier_acc: %0.6f, col_loss: %0.6f, col_acc: %0.6f" % (epoch, 
                                                    val_step, 
                                                    np.mean(val_sdf_losses),
                                                    np.mean(val_denoising_losses),
                                                    np.mean(val_classifier_losses),
                                                    np.mean(val_classifier_accuracies),
                                                    np.mean(val_collision_losses),
                                                    np.mean(val_collision_accuracies)))
                        if logger:
                            logger.log({
                                "val_sdf_loss": np.mean(val_sdf_losses),
                                "val_denoise_loss": np.mean(val_denoising_losses),
                                "val_classifier_loss": np.mean(val_classifier_losses),
                                "val_classifier_acc": np.mean(val_classifier_accuracies),
                            })
                        val_sdf_losses, val_denoising_losses, val_classifier_losses, val_collision_losses =  [], [], [], []
                        val_classifier_accuracies, val_collision_accuracies = [], []

                print('Validation set finished')
                model.train()
                
            lr_scheduler.step()
            print('Epoch %d finished' % epoch)

        return model, optimizers