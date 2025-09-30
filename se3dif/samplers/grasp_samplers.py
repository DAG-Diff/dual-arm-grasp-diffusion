import numpy as np
import torch
import os, os.path as osp

import theseus as th
from theseus import SO3
from se3dif.utils import SO3_R3
from tqdm import tqdm
from icecream import ic
import random

class ApproximatedGrasp_AnnealedLD():
    def __init__(self, model, device='cpu', batch=10, dim =3,
                 T=200, T_fit=5, deterministic=False):

        self.model = model
        self.device = device
        self.dim = dim
        self.shape = [4,4]
        self.batch = batch

        ## Langevin Dynamics evolution ##
        self.T = T
        self.T_fit = T_fit
        self.deterministic = deterministic

    def _marginal_prob_std(self, t, sigma=0.5):
        return np.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

    def _step(self, H0, t, noise_off=True):

        ## Phase
        eps = 1e-3
        phase = ((self.T - t)/self.T) + eps
        sigma_T = self._marginal_prob_std(eps)

        ## Move points to axis-angle ##
        xw0 = SO3_R3(R=H0[..., :3, :3] , t=H0[..., :3, -1]).log_map()

        ## Annealed Langevin Dynamics ##
        alpha = 1e-3
        sigma_i = self._marginal_prob_std(phase)
        ratio = sigma_i/sigma_T
        c_lr = alpha*ratio**2

        ## 1. Add Noise
        if noise_off:
            noise = torch.zeros_like(xw0)
        else:
            noise = torch.randn_like(xw0)

        noise = np.sqrt(c_lr)*noise

        xw01 = xw0 + np.sqrt(alpha)*ratio*noise


        ## 2. Compute gradient ##
        t_in = phase*torch.ones_like(xw01[...,0])
        xw01 = xw01.detach().requires_grad_(True)
        H_in = SO3_R3().exp_map(xw01).to_matrix()
        energy = self.model(H_in, t_in)
        grad_energy = torch.autograd.grad(energy.sum(), xw01, only_inputs=True)[0]

        ## 3. Evolve gradient ##
        delta_x = -.5*c_lr*grad_energy
        #! This is for single arm grasps 
        xw1 = xw01 + delta_x # shift the point two the gradient direction

        ## Build H ##
        H1 = SO3_R3().exp_map(xw1)
        return H1.to_matrix()

    def sample(self, save_path=False, batch=None):

        ## 1.Sample initial SE(3) ##
        if batch is None:
            batch = self.batch
        H0 = SO3_R3().sample(batch).to(self.device, torch.float32)

        ## 2.Langevin Dynamics (We evolve the data as [R3, SO(3)] pose)##
        Ht = H0
        if save_path:
            trj_H = Ht[None,...]
        for t in range(self.T):
            Ht = self._step(Ht, t, noise_off=self.deterministic)
            if save_path:
                trj_H = torch.cat((trj_H, Ht[None,:]), 0)
        for t in range(self.T_fit):
            Ht = self._step(Ht, self.T, noise_off=True)
            if save_path:
                trj_H = torch.cat((trj_H, Ht[None,:]), 0)

        if save_path:
            return Ht, trj_H
        else:
            return Ht


class Grasp_AnnealedLD():
    def __init__(self, model, device='cpu', batch=10, dim =3, k_steps=1,
                 T=200, T_fit=5, deterministic=False):

        self.model = model
        self.device = device
        self.dim = dim
        self.shape = [4,4]
        self.batch = batch

        ## Langevin Dynamics evolution ##
        self.T = T
        self.T_fit = T_fit
        self.k_steps = k_steps
        self.deterministic = deterministic

    def _marginal_prob_std(self, t, sigma=0.5):
        return np.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

    def _step(self, H0, t, noise_off=True):

        ## Phase
        noise_std = .5
        eps = 1e-3
        phase = ((self.T - t) / (self.T)) + eps
        sigma_T = self._marginal_prob_std(eps)

        ## Annealed Langevin Dynamics ##
        alpha = 1e-3
        sigma_i = self._marginal_prob_std(phase)
        ratio = sigma_i ** 2 / sigma_T ** 2
        c_lr = alpha * ratio
        if noise_off:
            c_lr = 0.003

        H1 = H0
        for k in range(self.k_steps):

            ## 1.Set input variable to Theseus ##
            H0_in = SO3_R3(R=H1[:,:3,:3], t=H1[:,:3, -1])
            phi0 = H0_in.log_map()

            ## 2. Compute energy gradient ##
            phi0_in = phi0.detach().requires_grad_(True)
            H_in = SO3_R3().exp_map(phi0_in).to_matrix()
            t_in = phase*torch.ones_like(H_in[:,0,0])
            e = self.model(H_in, t_in, dual=True)
            d_phi = torch.autograd.grad(e.sum(), phi0_in)[0]

            ## 3. Compute noise vector ##
            if noise_off:
                noise = torch.zeros_like(phi0_in)
            else:
                noise = torch.randn_like(phi0_in)*noise_std

            ## 4. Compute translation ##
            delta = -c_lr/2*d_phi + np.sqrt(c_lr)*noise
            w_Delta = SO3().exp_map(delta[:, 3:])
            t_delta = delta[:, :3]

            ## 5. Move the points ##
            R1_out = th.compose(w_Delta, H0_in.R)
            t1_out = H0_in.t + t_delta
            H1 = SO3_R3(R=R1_out, t=t1_out).to_matrix()

        return H1, t_in

    def sample(self, save_path=False, batch=None):

        ## 1.Sample initial SE(3) ##
        if batch is None:
            batch = self.batch
        H0 = SO3_R3().sample(batch).to(self.device, torch.float32)

        ## 2.Langevin Dynamics (We evolve the data as [R3, SO(3)] pose)##
        Ht = H0
        if save_path:
            trj_H = Ht[None,...]
        for t in tqdm(range(self.T)):
            Ht, tt = self._step(Ht, t, noise_off=self.deterministic)
            if save_path:
                trj_H = torch.cat((trj_H, Ht[None,:]), 0)
        for t in tqdm(range(self.T_fit)):
            Ht, tt = self._step(Ht, self.T, noise_off=True)
            if save_path:
                trj_H = torch.cat((trj_H, Ht[None,:]), 0)

        if save_path:
            return Ht, trj_H, tt
        else:
            return Ht, tt

class PartGuidedGrasp_AnnealedLD():
    def __init__(self, model, model_full, device='cpu', batch=10, dim =3, k_steps=1,
                 T=200, T_fit=5, deterministic=False):

        self.model = model
        self.model_full = model_full
        self.device = device
        self.dim = dim
        self.shape = [4,4]
        self.batch = batch
        self.global_t_delta = []

        ## Langevin Dynamics evolution ##
        self.T = T
        self.T_fit = T_fit
        self.k_steps = k_steps
        self.deterministic = deterministic

    def _marginal_prob_std(self, t, sigma=0.5):
        return np.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

    def _step(self, H0, t, noise_off=True):

        ## Phase
        noise_std = .5
        eps = 1e-3
        phase = ((self.T - t) / (self.T)) + eps
        sigma_T = self._marginal_prob_std(eps)

        ## Annealed Langevin Dynamics ##
        alpha = 1e-3
        sigma_i = self._marginal_prob_std(phase)
        ratio = sigma_i ** 2 / sigma_T ** 2
        c_lr = alpha * ratio
        if noise_off:
            c_lr = 0.003

        H1 = H0
        for k in range(self.k_steps):

            ## 1.Set input variable to Theseus ##
            H0_in = SO3_R3(R=H1[:,:3,:3], t=H1[:,:3, -1])
            phi0 = H0_in.log_map()

            ## 2. Compute energy gradient ##
            phi0_in = phi0.detach().requires_grad_(True)
            H_in = SO3_R3().exp_map(phi0_in).to_matrix()
            t_in = phase*torch.ones_like(H_in[:,0,0])
            # e = self.model(H_in, t_in)
            # with torch.no_grad():
            p = self.model.geometry_encoder(H_in, self.model.points)
            e1 = self.model.part_forward(p, t_in)
            e2 = self.model_full.part_forward(p, t_in)
            e = torch.cat([e1,e2],dim=-1)
            # print(e.shape)
            e = e.max(keepdim=True,dim=-1).values
            # print(e.shape)
            # exit()
            d_phi = torch.autograd.grad(e.sum(), phi0_in)[0]

            ## 3. Compute noise vector ##
            if noise_off:
                noise = torch.zeros_like(phi0_in)
            else:
                noise = torch.randn_like(phi0_in)*noise_std

            ## 4. Compute translation ##
            delta = -c_lr/2*d_phi + np.sqrt(c_lr)*noise
            w_Delta = SO3().exp_map(delta[:, 3:])
            t_delta = delta[:, :3]

            ## 5. Move the points ##
            R1_out = th.compose(w_Delta, H0_in.R)
            # self.global_t_delta.append(t_delta[0].detach().cpu().numpy())
            t1_out = H0_in.t + t_delta
            H1 = SO3_R3(R=R1_out, t=t1_out).to_matrix()

        return H1, t_in

    def sample(self, save_path=False, batch=None):

        ## 1.Sample initial SE(3) ##
        if batch is None:
            batch = self.batch
        H0 = SO3_R3().sample(batch).to(self.device, torch.float32)

        ## 2.Langevin Dynamics (We evolve the data as [R3, SO(3)] pose)##
        Ht = H0
        if save_path:
            trj_H = Ht[None,...]
        for t in tqdm(range(self.T)):
            Ht, tt = self._step(Ht, t, noise_off=self.deterministic)
            if save_path:
                trj_H = torch.cat((trj_H, Ht[None,:]), 0)
        for t in tqdm(range(self.T_fit)):
            Ht, tt = self._step(Ht, self.T, noise_off=True)
            if save_path:
                trj_H = torch.cat((trj_H, Ht[None,:]), 0)

        if save_path:
            return Ht, trj_H, tt
        else:
            return Ht, tt
        
class DualGrasp_AnnealedLD():
    def __init__(self, model, device='cpu', batch=10, dim =3, k_steps=1,
                 T=200, T_fit=5, deterministic=False, seed=28):
        
        self.model = model
        self.device = device
        self.dim = dim
        self.shape = [4,4]
        self.batch = batch

        ## Langevin Dynamics evolution ##
        self.T = T
        self.T_fit = T_fit
        self.k_steps = k_steps
        self.deterministic = deterministic

    def _marginal_prob_std(self, t, sigma=0.5):
        return np.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

    def _step(self, H0, t, noise_off=True, grasp_condition=None, dual=False, refine=False, refine_step=None):
        ## Phase
        noise_std = 0.5
        eps = 1e-3
        phase = ((self.T - t) / (self.T)) + eps
        sigma_T = self._marginal_prob_std(eps)

        ## Annealed Langevin Dynamics ##
        alpha = 1e-3
        sigma_i = self._marginal_prob_std(phase)
        ratio = (sigma_i ** 2) / (sigma_T ** 2)
        c_lr = alpha * ratio
        if noise_off:
            c_lr = 0.003

        H1 = H0
        if not refine:
            for k in range(self.k_steps):
                H0_in1 = SO3_R3(R=H1.reshape(-1,2,4,4)[:,0,:3,:3], t=H1.reshape(-1,2,4,4)[:,0,:3,-1])
                H0_in2 = SO3_R3(R=H1.reshape(-1,2,4,4)[:,1,:3,:3], t=H1.reshape(-1,2,4,4)[:,1,:3,-1])

                ## 1.Set input variable to Theseus ##
                H0_in = SO3_R3(R=H1[:,:3,:3], t=H1[:,:3, -1])
                phi0 = H0_in.log_map()
                phi0 = phi0.reshape(-1, 2 * phi0.shape[-1])

                ## 2. Compute energy gradient ##
                phi0_in = phi0.clone().detach().requires_grad_(True)
                H_in = SO3_R3().exp_map(phi0_in.reshape(2*phi0_in.shape[0], -1)).to_matrix()
                t_in = phase*torch.ones_like(H_in[:self.batch,0,0])
                # e = self.model(H_in, t_in)
                if dual:
                    # ic(H_in.shape, t_in.shape)
                    t_in = phase * torch.ones_like(H_in[:, 0, 0])
                    e = self.model(H_in, t_in, batch=1, dual=True)
                else:
                    temp_H_in = H_in.reshape(-1, 2, 4, 4)
                    H1_in, H2_in = temp_H_in[:, 0, :, :], temp_H_in[:, 1, :, :]
                    e = self.model(H1=H1_in,
                                H2=H2_in,
                                k1=t_in, k2=t_in, batch=1)
                d_phi = torch.autograd.grad(e.sum(), phi0_in, retain_graph=True)[0]
                if self.model.classifier:
                    # logits corresponding to the good grasp
                    logits = torch.log(self.model.pred_label/(1 - self.model.pred_label + 1e-8))
                    classifier_grad = torch.autograd.grad(logits.sum(), phi0_in)[0]
                    # print(logits[0], classifier_grad[0])
                d_phi = d_phi - (1 * phase) * classifier_grad
                # print(d_phi[0])

                ## 3. Compute noise vector ##
                if noise_off:
                    noise = torch.zeros_like(phi0_in)
                else:
                    noise = torch.randn_like(phi0_in)*noise_std
                # ic(d_phi.shape)
                d_phi_left, d_phi_right = d_phi[:, :6], d_phi[:, 6:]
                delta_left = -0.5*c_lr*d_phi_left + np.sqrt(c_lr) * noise[:, :6]
                delta_right = -0.5*c_lr*d_phi_right + np.sqrt(c_lr) * noise[:, 6:]
                left_shifted = phi0_in[:, :6] + delta_left
                right_shifted = phi0_in[:, 6:] + delta_right
                # print(phi0_in[0])
                # print(d_phi[0])
                # print(-0.5 * c_lr)
                H1 = SO3_R3().exp_map(left_shifted).to_matrix()
                H2 = SO3_R3().exp_map(right_shifted).to_matrix()
                H1 = torch.stack([H1, H2], dim=1).reshape(-1, 4, 4)
                # exit()
        
        if refine:
            H1_temp = H1.clone()
            t_in = torch.ones_like(H1[:, 0, 0]) * 1e-3
            for kk in range(1):
                H0_in = SO3_R3(R=H1_temp[:, :3, :3], t=H1_temp[:, :3, -1])
                phi0 = H0_in.log_map()
                phi0 = phi0.reshape(-1, 2 * phi0.shape[-1])
                phi0_in = phi0.clone().detach().requires_grad_(True)
                H_in = SO3_R3().exp_map(phi0_in.reshape(2*phi0_in.shape[0], -1)).to_matrix()
            
                e = self.model(H_in, t_in, batch=1, dual=True)
                delta = torch.autograd.grad(e.sum(), phi0_in, retain_graph=False)[0]
                delta = delta.reshape(-1, 6).clone()
                
                phi0_in = phi0_in.reshape(-1, phi0_in.shape[-1]//2).clone().detach().requires_grad_(True)
                H_in = SO3_R3().exp_map(phi0_in).to_matrix()
                
                collision_pred = self.model(H_in, t_in, 
                                            dual=True, collision_forward=True)
                
                # print(collision_pred.reshape(-1))
                collision_logits = torch.log(collision_pred/(1 - collision_pred + 1e-8))
                collision_grad = torch.autograd.grad(collision_logits.sum(), phi0_in)[0]
                mask = (collision_pred.reshape(-1) < 1.0).float()
                
                damping_energy = 5e-3 * ((refine_step/100) ** 2)
                damping_collision = 0.005 * (1 - ((refine_step/100) ** 2))
                # damping_collision = 0.01 * (1 - ((refine_step/100) ** 2))
                # 0.004
                delta = mask[:, None] * (damping_energy * delta - collision_grad * damping_collision)
                # print(torch.where(mask.reshape(-1, 2) == 1)[0])
                # shifted = phi0_in.clone() - 0.5 * c_lr * delta
                shifted = phi0_in.clone() - delta
                H1_temp = SO3_R3().exp_map(shifted).to_matrix()
                
            H1 = H1_temp.reshape(-1, 4, 4)
        # return H1, t_in
        if self.model.classifier and refine:
            return H1, t_in, collision_logits
        if self.model.classifier and not refine:
            return H1, t_in, e, logits

    # def sample(self, save_path=False, batch=None, grasp_condition=None, dual=False):
    #     ## 1.Sample initial SE(3) ##
    #     if batch is None:
    #         batch = self.batch
    #     # Reproduce
    #     H0 = SO3_R3().sample(batch*2).to(self.device, torch.float32)
    #     ## 2.Langevin Dynamics (We evolve the data as [R3, SO(3)] pose)##
    #     Ht = H0
    #     if save_path:
    #         trj_H = Ht[None,...]
    #         energies = torch.zeros((self.T, 300,1), device=Ht.device)
    #         force_closures = torch.zeros((self.T, 300,1), device=Ht.device)
    #         collisions = torch.zeros((100, 600,1), device=Ht.device)
    #     for t in tqdm(range(self.T), desc='Langevin Dynamics Steps'):
    #         Ht, tt, e, fc = self._step(Ht, t, noise_off=self.deterministic, dual=dual)
    #         if save_path:
    #             trj_H = torch.cat((trj_H, Ht[None,:]), 0)
    #             energies[t] = e
    #             force_closures[t] = fc
    #             # print(Ht.shape)
                
    #     for t in tqdm(range(self.T_fit), desc='Fitting Steps'):
    #         Ht, tt, e, fc = self._step(Ht, self.T, noise_off=True, dual=dual, refine=False)
    #         if save_path:
    #             trj_H = torch.cat((trj_H, Ht[None,:]), 0)
                                                
    #     for t in tqdm(range(100), desc='Refining Steps'):
    #         Ht, tt, coll = self._step(Ht, self.T, noise_off=True, dual=dual, refine=True, refine_step=t)
    #         if save_path:
    #             trj_H = torch.cat((trj_H, Ht[None,:]), 0)
    #             collisions[t] = coll

    #     if save_path:
    #         return Ht, trj_H, tt
    #     else:
    #         return Ht, tt
    
    def sample(self, save_path=False, batch=None, grasp_condition=None, dual=False):
        ## 1. Sample initial SE(3) ##
        if batch is None:
            batch = self.batch
        H0 = SO3_R3().sample(batch * 2).to(self.device, torch.float32)

        ## 2. Langevin Dynamics ##
        Ht = H0
        if save_path:
            trj_H = [Ht.detach().clone()]
            energies = []
            force_closures = []
            collisions = []

        for t in tqdm(range(self.T), desc='Langevin Dynamics Steps'):
            Ht, tt, e, fc = self._step(Ht, t, noise_off=self.deterministic, dual=dual)
            if save_path:
                trj_H.append(Ht.detach().clone())
                energies.append(e.detach().clone())
                force_closures.append(fc.detach().clone())

        for t in tqdm(range(self.T_fit), desc='Fitting Steps'):
            Ht, tt, e, fc = self._step(Ht, self.T, noise_off=True, dual=dual, refine=False)
            if save_path:
                trj_H.append(Ht.detach().clone())
                energies.append(e.detach().clone())
                force_closures.append(fc.detach().clone())

        for t in tqdm(range(100), desc='Refining Steps'):
            Ht, tt, coll = self._step(Ht, self.T, noise_off=True, dual=dual, refine=True, refine_step=t)
            if save_path:
                trj_H.append(Ht.detach().clone())
                collisions.append(coll.detach().clone())

        if save_path:
            return (
                Ht,
                torch.stack(trj_H),
                tt,
                torch.stack(energies),
                torch.stack(force_closures),
                torch.stack(collisions),
            )
        else:
            return Ht, tt

            
            
    def sample_debug(self, save_path=False, batch=None, grasp_condition=None, dual=False):
        ## 1.Sample initial SE(3) ##
        if batch is None:
            batch = self.batch
        # Reproduce
        H0 = SO3_R3().sample(batch*2).to(self.device, torch.float32)
        ## 2.Langevin Dynamics (We evolve the data as [R3, SO(3)] pose)##
        Ht = H0
        if save_path:
            trj_H = Ht[None,...]
        for t in tqdm(range(self.T), desc='Langevin Dynamics Steps'):
            Ht, tt = self._step(Ht, t, noise_off=self.deterministic, dual=dual)
            if save_path:
                trj_H = torch.cat((trj_H, Ht[None,:]), 0)
                # print(Ht.shape)
        for t in tqdm(range(self.T_fit), desc='Fitting Steps'):
            Ht, tt = self._step(Ht, self.T, noise_off=True, dual=dual, refine=False)
            if save_path:
                trj_H = torch.cat((trj_H, Ht[None,:]), 0)
        
        H_ = Ht.reshape(-1, 4, 4).detach()
        with torch.no_grad():
            e_wo_refinement = self.model(H_, tt, batch=1, dual=dual).flatten()
            fc_wo_refinement = self.model.pred_label
            
            
        Ht = Ht.detach()            
        for t in tqdm(range(100), desc='Refining Steps'):
            Ht, tt = self._step(Ht, self.T, noise_off=True, dual=dual, refine=True, refine_step=t)
            if save_path:
                trj_H = torch.cat((trj_H, Ht[None,:]), 0)

        H_ = Ht.reshape(-1, 4, 4).detach()
        with torch.no_grad():
            e_w_refinement = self.model(H_, tt, batch=1, dual=dual).flatten()
            fc_w_refinement = self.model.pred_label

        torch.save(e_wo_refinement.detach().cpu().numpy(), './temp/e_wo_refinement.pt')
        torch.save(fc_wo_refinement.detach().cpu().numpy(), './temp/fc_wo_refinement.pt')
        
        torch.save(e_w_refinement.detach().cpu().numpy(), './temp/e_w_refinement.pt')
        torch.save(fc_w_refinement.detach().cpu().numpy(), './temp/fc_w_refinement.pt')
                
        if save_path:
            return Ht, trj_H, tt
        else:
            return Ht, tt
        

if __name__ == '__main__':
    import torch.nn as nn

    class model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, H, k):
            H_th = SO3_R3(R=H[:, :3, :3], t=H[:, :3, -1])
            x = H_th.log_map()
            return x.pow(2).sum(-1)

    ## 1. Approximated Grasp_AnnealedLD
    generator = ApproximatedGrasp_AnnealedLD(model(), T=100, T_fit=500)
    H = generator.sample()
    print(H)

    ## 2. Grasp_AnnealedLD
    generator = Grasp_AnnealedLD(model(), T=100, T_fit=500, k_steps=1)
    H = generator.sample()
    print(H)




