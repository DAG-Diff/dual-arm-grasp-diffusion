import torch
import numpy as np
import torch.nn as nn
from icecream import ic

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = torch.einsum('...,b->...b',x, self.W)* 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class NaiveSE3DiffusionModel(nn.Module):
    def __init__(self, energy=False):
        super().__init__()

        input_size = 12
        enc_dim = 128
        if energy:
            output_size = 1
        else:
            output_size = 6

        self.network = nn.Sequential(
            nn.Linear(2*enc_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

        ## Time Embedings Encoder ##
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=enc_dim),
            nn.Linear(enc_dim, enc_dim),
            nn.SiLU(),
        )
        self.x_embed = nn.Sequential(
            nn.Linear(input_size, enc_dim),
            nn.SiLU(),
        )

    def marginal_prob_std(self, t, sigma=0.5):
        return torch.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

    def forward(self, x, R, k):
        std = self.marginal_prob_std(k)
        x_R_input = torch.cat((x, R.reshape(R.shape[0], -1)), dim=-1)
        z = self.x_embed(x_R_input)
        z_time = self.time_embed(k)
        z_in = torch.cat((z, z_time),dim=-1)
        v = self.network(z_in)
        return v/(std[:,None].pow(2))


class GraspDiffusionFields(nn.Module):
    ''' Grasp DiffusionFields. SE(3) diffusion model to learn 6D grasp distributions. See
        SE(3)-DiffusionFields: Learning cost functions for joint grasp and motion optimization through diffusion
    '''
    def __init__(self, vision_encoder, geometry_encoder, points, feature_encoder, decoder):
        super().__init__()
        ## Register points to map H to points ##
        self.register_buffer('points', points)
        ## Vision Encoder. Map observation to visual latent code ##
        self.vision_encoder = vision_encoder
        ## vision latent code
        self.z = None
        ## Geometry Encoder. Map H to points ##
        self.geometry_encoder = geometry_encoder
        ## Feature Encoder. Get SDF and latent features ##
        self.feature_encoder = feature_encoder
        ## Decoder ##
        self.decoder = decoder

    def set_latent(self, O, batch = 1):
        self.z = self.vision_encoder(O.squeeze(1))
        self.z = self.z.unsqueeze(1).repeat(1, batch, 1).reshape(-1, self.z.shape[-1])

    def forward(self, H, k):
        ## 1. Represent H with points
        p = self.geometry_encoder(H, self.points)
        k_ext = k.unsqueeze(1).repeat(1, p.shape[1])
        z_ext = self.z.unsqueeze(1).repeat(1, p.shape[1], 1)
        ## 2. Get Features
        psi = self.feature_encoder(p, k_ext, z_ext)
        ## 3. Flat and get energy
        psi_flatten = psi.reshape(psi.shape[0], -1)
        e = self.decoder(psi_flatten)
        return e

    def compute_sdf(self, x):
        k = torch.rand_like(x[..., 0])
        psi = self.feature_encoder(x, k, self.z)
        return psi[..., 0]

class ConvGraspVAE(GraspDiffusionFields):
    def __init__(self, vision_encoder, geometry_encoder, points, vision_decoder, feature_encoder, gaussian_mlp, feature_decoder):
        super().__init__(vision_encoder, geometry_encoder, points, feature_encoder, None)
        self.vision_encoder = vision_encoder
        self.vision_decoder = vision_decoder
        self.geometry_encoder = geometry_encoder    
        self.gaussian_mlp = gaussian_mlp
        self.feature_decoder = feature_decoder
        self.z = None
        print('Inside ConvGraspVAE')
    
    def set_latent(self, O, batch = 1):
        self.z = self.vision_encoder(O.squeeze(1))

    def get_latent(self, local_context=None):
        out = self.vision_decoder(local_context, self.z)
        return out
    
    def reparamterization_trick(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, H, k, batch=1):
        p = self.geometry_encoder(H, self.points) # H are the grasp poses and self.points are 30 cube points
        # k_ext = k.unsqueeze(1).repeat(1, p.shape[1])
        local_context = p.reshape(batch, -1, p.shape[-1])
        z_local = self.get_latent(local_context=local_context)
        
        z_local = z_local.reshape(-1,p.shape[1], z_local.shape[-1])
        ## 2. Get Features
        p = p.reshape(-1, p.shape[1] * 2, p.shape[2])
        z_local = z_local.reshape(-1, z_local.shape[1] * 2, z_local.shape[2])
        psi = self.feature_encoder(p, torch.tensor([0.0], device=p.device), z_local)
        psi_flatten = psi.reshape(psi.shape[0], -1)
        temp = self.gaussian_mlp(psi_flatten)
        mu, logvar = temp[..., :256], temp[..., 256:]
        
        latent = self.reparamterization_trick(mu, logvar)
        out = self.feature_decoder(latent)
        return out, mu, logvar
        

class ConvGraspDiffusionFields(GraspDiffusionFields):
    
    def __init__(self, vision_encoder, vision_decoder, geometry_encoder, 
                 points, feature_encoder, decoder, 
                 use_attention=False, classifier=None, collision_predictor=None):
        super().__init__(vision_encoder, geometry_encoder, points, feature_encoder, decoder)
        self.vision_decoder = vision_decoder
        self.use_attention = use_attention
        
        if self.use_attention:
            embed_dim = 32
            num_heads = 4
            self.projection_layer_up = nn.Linear(7, embed_dim)
            self.projection_layer_down = nn.Linear(embed_dim, 7)
            self.attention_layer = nn.MultiheadAttention(embed_dim=embed_dim, 
                                                         num_heads=num_heads, 
                                                         batch_first=True)
        self.classifier = classifier
        self.collision_predictor = collision_predictor
    
    def set_latent(self, O, batch = 1):
        self.z = self.vision_encoder(O.squeeze(1))

    def get_latent(self, local_context=None):
        out = self.vision_decoder(local_context, self.z)
        return out
    
    def compute_sdf(self, x):
        latent_vec = self.get_latent(local_context=x)
        latent_vec = latent_vec.reshape(-1, latent_vec.shape[-1])
        x = x.reshape(-1,3)
        k = torch.rand_like(x[..., 0])
        psi = self.feature_encoder(x, k, latent_vec)
        return psi[..., 0] * 8.
    
    def collision_forward(self, p, k, batch=1):
        k_ext = k.unsqueeze(1).repeat(1, p.shape[1])
        local_context = p.reshape(batch, -1, p.shape[-1])
        z_ext = self.get_latent(local_context=local_context)
        
        z_ext = z_ext.reshape(-1,p.shape[1], z_ext.shape[-1])
        psi = self.feature_encoder(p, k_ext, z_ext)
        psi_flatten = psi.reshape(psi.shape[0], -1)
        self.collision_pred = self.collision_predictor(psi_flatten)
        return self.collision_pred
        
    def part_forward(self,p,k,batch=1, dual=False):
        k_ext = k.unsqueeze(1).repeat(1, p.shape[1])
        local_context = p.reshape(batch, -1, p.shape[-1])
        z_ext = self.get_latent(local_context=local_context)
        # ic(z_ext.shape)
        z_ext = z_ext.reshape(-1,p.shape[1], z_ext.shape[-1])
        ## 2. Get Features
        p = p.reshape(-1, p.shape[1] * 2, p.shape[2])
        k_ext = k_ext.reshape(-1, k_ext.shape[1] * 2)
        z_ext = z_ext.reshape(-1, z_ext.shape[1] * 2, z_ext.shape[2])
        
        psi = self.feature_encoder(p, k_ext, z_ext)
        if self.use_attention:
            psi_dual = psi.reshape(-1, 2, psi.shape[1], psi.shape[2])
            psi1, psi2 = psi_dual[:, 0, ...], psi_dual[:, 1, ...]
            psi1 = self.projection_layer_up(psi1)
            psi2 = self.projection_layer_up(psi2)
            psi_dual = torch.cat([psi1, psi2], dim=1)
            psi, _ = self.attention_layer(psi_dual, psi_dual, psi_dual)
            psi = self.projection_layer_down(psi)
            
        ## 3. Flat and get energy
        # if dual:
        #     if self.use_attention:
        #         psi_flatten = psi.reshape(psi.shape[0], -1)
        #     else:
        #         psi_flatten = psi.reshape(psi.shape[0]//2, -1)
        # else:
        #     psi_flatten = psi.reshape(psi.shape[0], -1)
        
        psi_flatten_energy = psi.reshape(psi.shape[0], -1)
        e = self.decoder(psi_flatten_energy) # decoder is energy mlp
        if self.classifier is not None:
            self.pred_label = self.classifier(psi_flatten_energy)
        return e
    
    def forward(self, H, k, batch=1, dual=False, collision_forward=False):
        ## 1. Represent H with points
        p = self.geometry_encoder(H, self.points) # H are the grasp poses and self.points are 30 cube points
        if collision_forward:
            return self.collision_forward(p, k, batch=batch)
        else:
            return self.part_forward(p, k, batch=batch, dual=dual)
    
class DualGraspDiffusionFields(nn.Module):
    ''' Grasp DiffusionFields. SE(3) diffusion model to learn 6D grasp distributions. See
        SE(3)-DiffusionFields: Learning cost functions for joint grasp and motion optimization through diffusion
    '''
    def __init__(self, vision_encoder, vision_decoder, 
                 geometry_encoder, points, feature_encoder, 
                 decoder1, decoder2, classifier=None):
        super().__init__()
        ## Register points to map H to points ##
        self.register_buffer('points', points)
        ## Vision Encoder. Map observation to visual latent code ##
        self.vision_encoder = vision_encoder
        ## vision latent code
        self.z = None
        ## Geometry Encoder. Map H to points ##
        self.geometry_encoder = geometry_encoder
        ## Feature Encoder. Get SDF and latent features ##
        self.feature_encoder = feature_encoder
        ## Decoders ##
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        self.vision_decoder = vision_decoder
        self.classifier = classifier

    def set_latent(self, O, batch=1):
        self.z = self.vision_encoder(O.squeeze(1))
        # self.z = self.z.unsqueeze(1).repeat(1, batch, 1)  # Shape becomes (4, batch, 32)
        
    def get_latent(self, local_context=None):
        ic(local_context)
        out = self.vision_decoder(local_context, self.z)
        return out

    def single_forward(self, H, k, batch, grasp_condition=None):
        ## 1. Represent H with points
        p = self.geometry_encoder(H, self.points)
        k_ext = k.unsqueeze(1).repeat(1, p.shape[1])
        z_ext = self.get_latent(local_context=p.reshape(batch, -1, p.shape[-1])) #! bs*100, 30, 132
        # z_ext = z_ext.unsqueeze(1).repeat(1, p.shape[1], 1)
        z_ext = z_ext.reshape(-1, p.shape[1], z_ext.shape[-1])

        ## 2. Get Features
        psi = self.feature_encoder(p, k_ext, z_ext, grasp_condition=grasp_condition)

        ## 3. Flat and get energy
        psi_flatten = psi.reshape(psi.shape[0], -1)
        if grasp_condition is not None:
            concat_psi_flatten = torch.cat([psi_flatten, grasp_condition], dim=-1)
            e = self.decoder2(concat_psi_flatten)
        else:
            e = self.decoder1(psi_flatten)

        return e, psi_flatten
    
    #! returns 2 energies
    def forward(self, H1, H2, k1, k2, batch, dual=False):
        e1, grasp_condition = self.single_forward(H1, k1, batch=batch)
        e2, _ = self.single_forward(H2, k2, grasp_condition=grasp_condition, batch=batch)
        # ic(e1, e2)
        return torch.max(e1, e2)
    
    def compute_sdf(self, x):
        latent_vec = self.get_latent(local_context=x) #! bs, 1000, 132
        latent_vec = latent_vec.reshape(-1, latent_vec.shape[-1]) #! 4000, 132
        x = x.reshape(-1,3) #! 4000, 3
        k = torch.rand_like(x[..., 0]) #! 4000
        psi = self.feature_encoder(x, k, latent_vec) #! 4000, 7
        return psi[..., 0] * 8. #! why multiply with 8? 
    

class ConvContactPoinDiffusionFields(GraspDiffusionFields):
    
    def __init__(self, vision_encoder, vision_decoder, geometry_encoder, 
                 points, feature_encoder, decoder, 
                 use_attention=False, classifier=None):
        super().__init__(vision_encoder, geometry_encoder, points, feature_encoder, decoder)
        self.vision_decoder = vision_decoder
        self.use_attention = use_attention
        
        if self.use_attention:
            embed_dim = 32
            num_heads = 4
            self.projection_layer_up = nn.Linear(7, embed_dim)
            self.projection_layer_down = nn.Linear(embed_dim, 7)
            self.attention_layer = nn.MultiheadAttention(embed_dim=embed_dim, 
                                                         num_heads=num_heads, 
                                                         batch_first=True)
        self.classifier = classifier
    
    def set_latent(self, O, batch = 1):
        self.z = self.vision_encoder(O.squeeze(1))

    def get_latent(self, local_context=None):
        out = self.vision_decoder(local_context, self.z)
        return out
    
    def compute_sdf(self, x):
        latent_vec = self.get_latent(local_context=x)
        latent_vec = latent_vec.reshape(-1, latent_vec.shape[-1])
        x = x.reshape(-1,3)
        k = torch.rand_like(x[..., 0])
        psi = self.feature_encoder(x, k, latent_vec)
        return psi[..., 0] * 8.
    
    def part_forward(self,p,k,batch=1, dual=False):
        k_ext = k.unsqueeze(1).repeat(1, p.shape[1])
        local_context = p.reshape(batch, -1, p.shape[-1])
        z_ext = self.get_latent(local_context=local_context)
        # ic(z_ext.shape)
        z_ext = z_ext.reshape(-1,p.shape[1], z_ext.shape[-1])
        ## 2. Get Features
        p = p.reshape(-1, p.shape[1] * 2, p.shape[2])
        k_ext = k_ext.reshape(-1, k_ext.shape[1] * 2)
        z_ext = z_ext.reshape(-1, z_ext.shape[1] * 2, z_ext.shape[2])
        
        psi = self.feature_encoder(p, k_ext, z_ext)[..., 1:] # Exclude the first channel (SDF)
        
        if self.use_attention:
            psi_dual = psi.reshape(-1, 2, psi.shape[1], psi.shape[2])
            psi1, psi2 = psi_dual[:, 0, ...], psi_dual[:, 1, ...]
            psi1 = self.projection_layer_up(psi1)
            psi2 = self.projection_layer_up(psi2)
            psi_dual = torch.cat([psi1, psi2], dim=1)
            psi, _ = self.attention_layer(psi_dual, psi_dual, psi_dual)
            psi = self.projection_layer_down(psi)
            
        ## 3. Flat and get energy
        # if dual:
        #     if self.use_attention:
        #         psi_flatten = psi.reshape(psi.shape[0], -1)
        #     else:
        #         psi_flatten = psi.reshape(psi.shape[0]//2, -1)
        # else:
        #     psi_flatten = psi.reshape(psi.shape[0], -1)
        
        psi_energy = psi + k_ext[:, :, None]
        psi_flatten_energy = psi_energy.reshape(psi.shape[0], -1)
        
        psi_flatten = psi.reshape(psi.shape[0], -1)
        
        e = self.decoder(psi_flatten_energy)
        if self.classifier is not None:
            self.pred_label = self.classifier(psi_flatten)
        
        return e
    
    def forward(self, H, k, batch=1, dual=False):
        ## 1. Represent H with points
        p = self.geometry_encoder(H, self.points)
        ic(p.shape)
        exit()
        return self.part_forward(p, k, batch=batch, dual=dual)