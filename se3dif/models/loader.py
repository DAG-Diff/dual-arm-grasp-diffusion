import os
import torch
import torch.nn as nn
import numpy as np

from se3dif import models


from se3dif.utils import get_pretrained_models_src, load_experiment_specifications
pretrained_models_dir = get_pretrained_models_src()
import random

def load_model(args):
    if 'pretrained_model' in args:
        model_args = load_experiment_specifications('/home/dualarm/dummyhome/md/March2025/dual-arm-grasp-diffusion/configs/',
                                                    load_yaml=args['pretrained_model'])
        args['classifier_path'] = model_args['classifier_path']      
        args["NetworkArch"] = model_args["NetworkArch"]
        args["NetworkSpecs"] = model_args["NetworkSpecs"]
        args['use_attention'] = model_args['use_attention']

    if args['NetworkArch'] == 'GraspDiffusion':
        print('Loading GraspDiffusion')
        model = load_grasp_diffusion(args)
    elif args['NetworkArch'] == 'PointcloudGraspDiffusion':
        print('Loading PointcloudGraspDiffusion')
        model = load_pointcloud_grasp_diffusion(args)
    elif args['NetworkArch'] == 'PointcloudGraspDiffusionConv':
        print('Loading PointcloudGraspDiffusionConv')
        model = load_pointcloud_grasp_diffusion_conv_encoder(args)
    elif args['NetworkArch'] == 'DualGraspDiffusionConv':
        # model = load_dual_arm_pointcloud_grasp_diffusion(args, inference='pretrained_model' in args) 
        model = load_dual_arm_pointcloud_grasp_diffusion_occupancy_encoder(args, inference='pretrained_model' in args)
        print('Loaded DualGraspDiffusionConv')
    elif args['NetworkArch'] == 'DualGraspVAE':
        model = load_dual_arm_pointcloud_grasp_vae(args, inference='pretrained_model' in args)
        print('Loaded DualGraspVAE')
        
    if 'pretrained_model' in args:
        model_path = './experiments_jul/collision_predictor_30jul_UniformPts/checkpoints/model_epoch_0359_iter_211091.pth'
        print('Loading Pretrained model from', model_path)

        ret = model.load_state_dict(torch.load(model_path), strict=False)
        print(ret)

        if args['device'] != 'cpu':
            model = model.to(args['device'], dtype=torch.float32)

    return model


def load_grasp_diffusion(args):
    device = args['device']
    params = args['NetworkSpecs']
    feat_enc_params = params['feature_encoder']
    lat_params = params['latent_codes']
    points_params = params['points']
    # vision encoder
    vision_encoder = models.vision_encoder.LatentCodes(num_scenes=lat_params['num_scenes'], latent_size=lat_params['latent_size'])
    # Geometry encoder
    geometry_encoder = models.geometry_encoder.map_projected_points
    # Feature Encoder
    feature_encoder = models.nets.TimeLatentFeatureEncoder(
            enc_dim=feat_enc_params['enc_dim'],
            latent_size= lat_params['latent_size'],
            dims = feat_enc_params['dims'],
            out_dim=feat_enc_params['out_dim'],
            dropout=feat_enc_params['dropout'],
            dropout_prob=feat_enc_params['dropout_prob'],
            norm_layers = feat_enc_params['norm_layers'],
            latent_in = feat_enc_params["latent_in"],
            xyz_in_all = feat_enc_params["xyz_in_all"],
            use_tanh = feat_enc_params["use_tanh"],
            latent_dropout = feat_enc_params["latent_dropout"],
            weight_norm= feat_enc_params["weight_norm"]
        )
    # 3D Points
    if 'loc' in points_params:
        points = models.points.get_3d_pts(n_points = points_params['n_points'],
                            loc=np.array(points_params['loc']),
                            scale=np.array(points_params['scale']))
    else:
        points = models.points.get_3d_pts(n_points=points_params['n_points'])
    # Energy Based Model
    in_dim = points_params['n_points']*feat_enc_params['out_dim']
    hidden_dim = 512
    energy_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
    )

    model = models.GraspDiffusionFields(vision_encoder=vision_encoder, feature_encoder=feature_encoder, geometry_encoder=geometry_encoder,
                                       decoder=energy_net, points=points).to(device)
    return model


def load_pointcloud_grasp_diffusion(args):
    device = args['device']
    params = args['NetworkSpecs']
    feat_enc_params = params['feature_encoder']
    v_enc_params = params['encoder']
    points_params = params['points']
    # vision encoder
    vision_encoder = models.vision_encoder.VNNPointnet2(out_features=v_enc_params['latent_size'], device=device)
    # Geometry encoder
    geometry_encoder = models.geometry_encoder.map_projected_points
    # Feature Encoder
    feature_encoder = models.nets.TimeLatentFeatureEncoder(
            enc_dim=feat_enc_params['enc_dim'],
            latent_size= v_enc_params['latent_size'],
            dims = feat_enc_params['dims'],
            out_dim=feat_enc_params['out_dim'],
            dropout=feat_enc_params['dropout'],
            dropout_prob=feat_enc_params['dropout_prob'],
            norm_layers = feat_enc_params['norm_layers'],
            latent_in = feat_enc_params["latent_in"],
            xyz_in_all = feat_enc_params["xyz_in_all"],
            use_tanh = feat_enc_params["use_tanh"],
            latent_dropout = feat_enc_params["latent_dropout"],
            weight_norm= feat_enc_params["weight_norm"]
        )
    # 3D Points
    if 'loc' in points_params:
        points = models.points.get_3d_pts(n_points = points_params['n_points'],
                            loc=np.array(points_params['loc']),
                            scale=np.array(points_params['scale']))
    else:
        points = models.points.get_3d_pts(n_points=points_params['n_points'])
    # Energy Based Model
    in_dim = points_params['n_points']*feat_enc_params['out_dim']
    hidden_dim = 512
    energy_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
    )

    model = models.GraspDiffusionFields(vision_encoder=vision_encoder, feature_encoder=feature_encoder, geometry_encoder=geometry_encoder,
                                       decoder=energy_net, points=points).to(device)
    return model


def load_pointcloud_grasp_diffusion_conv_encoder(args):
    device = args['device']
    params = args['NetworkSpecs']
    feat_enc_params = params['feature_encoder']
    v_enc_params = params['encoder']
    points_params = params['points']
    # vision encoder
    plane_type   = ['xz', 'xy', 'yz']
    # plane_type   = ['grid']
    grid_resolution = 32
    plane_resolution = 32
    vision_encoder = models.vision_encoder.VNNLocalPoolPointnet(c_dim = int(v_enc_params['latent_size'] / 3),
                                                                    grid_resolution=grid_resolution,
                                                                    plane_type=plane_type,
                                                                    unet=True,
                                                                    plane_resolution=plane_resolution, device=device).to(device)
    
    # Simple PointNet Version
    # vision_encoder = models.vision_encoder.LocalPoolPointnet(c_dim = v_enc_params['latent_size'],
    #                                                                 grid_resolution=grid_resolution,
    #                                                                 plane_type=plane_type,
    #                                                                 unet=True,
    #                                                                 plane_resolution=plane_resolution).to(device)
    
    vision_decoder = models.vision_encoder.LocalDecoder(c_dim = v_enc_params['latent_size'])
    # Geometry encoder
    geometry_encoder = models.geometry_encoder.map_projected_points
    # Feature Encoder
    feature_encoder = models.nets.TimeLatentFeatureEncoder(
            enc_dim=feat_enc_params['enc_dim'],
            latent_size= v_enc_params['latent_size'],
            dims = feat_enc_params['dims'],
            out_dim=feat_enc_params['out_dim'],
            dropout=feat_enc_params['dropout'],
            dropout_prob=feat_enc_params['dropout_prob'],
            norm_layers = feat_enc_params['norm_layers'],
            latent_in = feat_enc_params["latent_in"],
            xyz_in_all = feat_enc_params["xyz_in_all"],
            use_tanh = feat_enc_params["use_tanh"],
            latent_dropout = feat_enc_params["latent_dropout"],
            weight_norm= feat_enc_params["weight_norm"]
        )
    # 3D Points
    if 'loc' in points_params:
        print('Using fixed points using loc and scale')
        points = models.points.get_3d_pts(n_points = points_params['n_points'],
                            loc=np.array(points_params['loc']),
                            scale=np.array(points_params['scale']))
    else:
        points = models.points.get_3d_pts(n_points=points_params['n_points'])
    # Energy Based Model
    in_dim = points_params['n_points']*feat_enc_params['out_dim']
    hidden_dim = 512
    energy_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
    )
    
    model = models.ConvGraspDiffusionFields(vision_encoder=vision_encoder, vision_decoder=vision_decoder, feature_encoder=feature_encoder, geometry_encoder=geometry_encoder,
                                       decoder=energy_net, points=points).to(device)
    
    weights = torch.load('./demo/data/models/cgdf_v1/model_pretrained.pth')
    # print(weights.keys())
    ret = model.load_state_dict(weights, strict=True)
    print(f'Pretrained weights loaded! | {ret}')
    return model

def load_dual_arm_pointcloud_grasp_diffusion(args, inference=False):
    device = args['device']
    params = args['NetworkSpecs']
    feat_enc_params = params['feature_encoder']
    v_enc_params = params['encoder']
    points_params = params['points']
    # vision encoder
    # vision_encoder = models.vision_encoder.VNNPointnet2(out_features=v_enc_params['latent_size'], device=device)
    # vision encoder
    plane_type   = ['xz', 'xy', 'yz']
    # plane_type   = ['grid']
    grid_resolution = 32
    plane_resolution = 32
    vision_encoder = models.vision_encoder.VNNLocalPoolPointnet(c_dim = int(v_enc_params['latent_size'] / 3),
                                                                grid_resolution=grid_resolution,
                                                                plane_type=plane_type,
                                                                unet=True,
                                                                plane_resolution=plane_resolution, device=device).to(device)
    # Geometry encoder
    geometry_encoder = models.geometry_encoder.map_projected_points
    # Feature Encoder
    feature_encoder = models.nets.TimeLatentFeatureEncoderGraspConditioned(
            enc_dim=feat_enc_params['enc_dim'],
            latent_size= v_enc_params['latent_size'],
            dims = feat_enc_params['dims'],
            out_dim=feat_enc_params['out_dim'],
            dropout=feat_enc_params['dropout'],
            dropout_prob=feat_enc_params['dropout_prob'],
            norm_layers = feat_enc_params['norm_layers'],
            latent_in = feat_enc_params["latent_in"],
            xyz_in_all = feat_enc_params["xyz_in_all"],
            use_tanh = feat_enc_params["use_tanh"],
            latent_dropout = feat_enc_params["latent_dropout"],
            weight_norm= feat_enc_params["weight_norm"],
            # recurrent=True
        )
    # 3D Points
    if 'loc' in points_params:
        points = models.points.get_3d_pts(n_points = points_params['n_points'],
                            loc=np.array(points_params['loc']),
                            scale=np.array(points_params['scale']))
    else:
        points = models.points.get_3d_pts(n_points=points_params['n_points'])
    # Energy Based Model
    in_dim = points_params['n_points']*feat_enc_params['out_dim']
    hidden_dim = 512
    energy_net1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim//2, bias=False),
            nn.LayerNorm(hidden_dim//2),
            nn.SiLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4, bias=False),
            nn.LayerNorm(hidden_dim//4),
            nn.SiLU(),
            nn.Linear(hidden_dim//4, 1),
    )
    energy_net2 = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim//2, bias=False),
            nn.LayerNorm(hidden_dim//2),
            nn.SiLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4, bias=False),
            nn.LayerNorm(hidden_dim//4),
            nn.SiLU(),
            nn.Linear(hidden_dim//4, 1),
    )
    
    vision_decoder = models.vision_encoder.LocalDecoder(c_dim = v_enc_params['latent_size'])
    
    if args['classifier_path'] is not None:
        classifier = models.DualGPDClassifier()
        classifier_weights = torch.load(args['classifier_path'])
        classifier.load_state_dict(classifier_weights, strict=True)
        classifier.eval()
        print('Loaded classifier pretrained weights [strict = True]')
    else:
      classifier = None
      
    model = models.DualGraspDiffusionFields(vision_encoder=vision_encoder, vision_decoder=vision_decoder, feature_encoder=feature_encoder, geometry_encoder=geometry_encoder,
                                       decoder1=energy_net1, decoder2=energy_net2, points=points, classifier=classifier).to(device)
    
    # if not inference:
    #     model_weights = torch.load('./demo/data/models/cgdf_v1/model_pretrained.pth')
    #     weight_name = list(model_weights.keys())
        
    #     ret = model.load_state_dict(model_weights, strict=False)
    #     print(ret)
    #     print('Loaded Pretrained Weights for Vision Encoder and Feature Encoder [strict = False]')
        
    return model

def load_dual_arm_pointcloud_grasp_diffusion_occupancy_encoder(args, inference=False):
    device = args['device']
    params = args['NetworkSpecs']
    feat_enc_params = params['feature_encoder']
    v_enc_params = params['encoder']
    points_params = params['points']
    # vision encoder
    plane_type   = ['xz', 'xy', 'yz']
    grid_resolution = 32 
    plane_resolution = 32
    vision_encoder = models.vision_encoder.VNNLocalPoolPointnet(c_dim = int(v_enc_params['latent_size'] / 3),
                                                                    grid_resolution=grid_resolution,
                                                                    plane_type=plane_type,
                                                                    unet=True,
                                                                    plane_resolution=plane_resolution, 
                                                                    device=device,
                                                                    unet_depth=5).to(device)
    # vision_encoder = models.vision_encoder.LocalPoolPointnet(c_dim = v_enc_params['latent_size'],
    #                                                                 grid_resolution=grid_resolution,
    #                                                                 plane_type=plane_type,
    #                                                                 unet=True,
    #                                                                 plane_resolution=plane_resolution).to(device)
    vision_decoder = models.vision_encoder.LocalDecoder(c_dim = v_enc_params['latent_size'])
    # Geometry encoder
    geometry_encoder = models.geometry_encoder.map_projected_points
    # Feature Encoder
    feature_encoder = models.nets.TimeLatentFeatureEncoder(
            enc_dim=feat_enc_params['enc_dim'],
            latent_size= v_enc_params['latent_size'],
            dims = feat_enc_params['dims'],
            out_dim=feat_enc_params['out_dim'],
            dropout=feat_enc_params['dropout'],
            dropout_prob=feat_enc_params['dropout_prob'],
            norm_layers = feat_enc_params['norm_layers'],
            latent_in = feat_enc_params["latent_in"],
            xyz_in_all = feat_enc_params["xyz_in_all"],
            use_tanh = feat_enc_params["use_tanh"],
            latent_dropout = feat_enc_params["latent_dropout"],
            weight_norm= feat_enc_params["weight_norm"]
        )
    # 3D Points
    # if 'loc' in points_params:
    # points = models.points.get_3d_pts(n_points = points_params['n_points'],
                        # loc=np.array(points_params['loc']),
                        # scale=np.array(points_params['scale']))
    # else:
    points = models.points.get_3d_pts(n_points=points_params['n_points'],
                                    loc=np.array(points_params['loc']),
                                    scale=np.array(points_params['scale']))
    
    # Energy Based Model
    in_dim = points_params['n_points']*feat_enc_params['out_dim']
    hidden_dim = 512
    energy_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
    )
    
    # if args['use_attention']:
    #     in_dim = 30 * 16
    
    dual_energy_net = nn.Sequential(
        nn.Linear(2 * in_dim, hidden_dim, bias=False),
        nn.LayerNorm(hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
        nn.LayerNorm(hidden_dim // 2),
        nn.ELU(),
        nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=False),
        nn.LayerNorm(hidden_dim // 4),
        nn.ELU(),
        nn.Linear(hidden_dim // 4, hidden_dim//8, bias=False),
        nn.LayerNorm(hidden_dim//8),
        nn.ELU(),
        nn.Linear(hidden_dim//8, 1),
    )
    
    classifier = nn.Sequential(
        nn.Linear(2 * in_dim, hidden_dim, bias=False),
        nn.LayerNorm(hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
        nn.LayerNorm(hidden_dim // 2),
        nn.ELU(),
        nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=False),
        nn.LayerNorm(hidden_dim // 4),
        nn.ELU(),
        nn.Linear(hidden_dim // 4, hidden_dim//8, bias=False),
        nn.LayerNorm(hidden_dim//8),
        nn.ELU(),
        nn.Linear(hidden_dim//8, 1),
        nn.Sigmoid()
    )
    
    collision_predictor = nn.Sequential(
        nn.Linear(in_dim, hidden_dim, bias=False),
        nn.LayerNorm(hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
        nn.LayerNorm(hidden_dim // 2),
        nn.ELU(),
        nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=False),
        nn.LayerNorm(hidden_dim // 4),
        nn.ELU(),
        nn.Linear(hidden_dim // 4, hidden_dim//8, bias=False),
        nn.LayerNorm(hidden_dim//8),
        nn.ELU(),
        nn.Linear(hidden_dim//8, 1),
        nn.Sigmoid()
    )

    model = models.ConvGraspDiffusionFields(vision_encoder=vision_encoder, 
                                            vision_decoder=vision_decoder, 
                                            feature_encoder=feature_encoder, 
                                            geometry_encoder=geometry_encoder,
                                            decoder=dual_energy_net, points=points, 
                                            use_attention=args['use_attention'],
                                            classifier=classifier,
                                            collision_predictor=collision_predictor).to(device)
    
    # weights_path = './demo/data/models/cgdf_v1/model_pretrained.pth'
    weights_path = './experiments_jul/ablations/wo_fc_from_scratch/checkpoints/model_current.pth'
    # weights_path = './experiments_jul/dual_grasp_gen:linear-std/checkpoints/model_current.pth'
    # weights_path = './experiments_may/dual_grasp_diffusion_conv_classifier_joint_june23/checkpoints/model_epoch_0229_iter_114957.pth'
    # weights_path = './experiments_may/dual_grasp_diffusion_conv_classifier_may22/checkpoints/model_epoch_0509_iter_255517.pth'
    # weights_path = './experiments_jul/collision_prediction_joint_10jul_2/checkpoints/model_current.pth'

    # weights_path = './experiments_jul/collision_prediction_joint_10jul/checkpoints/model_current.pth'
    # weights_path = './experiments_may/dual_grasp_diffusion_conv_classifier_joint_june23/checkpoints/model_epoch_0389_iter_195277.pth'
    # weights_path = './experiments_jul/dual_arm_grasp_gen_w_gripper_pcd_13jul/checkpoints/model_current.pth'
    # weights_path = './experiments_jul/collision_predictor_3aug_collision/checkpoints/model_current.pth'

    
    if os.path.exists(weights_path):
        model_weights = torch.load(weights_path)
        if not inference:
            if 'model_pretrained' in weights_path:
                for name, param in model.named_parameters():
                    if 'vision_encoder' in name or 'feature_encoder' in name:
                        param.data = model_weights[name]          
                print(f'Loaded Pretrained Weights for Vision Encoder and Feature Encoder from {weights_path}')
            else:
                ret = model.load_state_dict(model_weights, strict=False)
                print(ret)
                print(f'Loaded Pretrainedel_weights, Weights from {weights_path}')
        
    # print(model)
    return model


def load_dual_arm_pointcloud_grasp_vae(args, inference=False):
    device = args['device']
    params = args['NetworkSpecs']
    feat_enc_params = params['feature_encoder']
    v_enc_params = params['encoder']
    points_params = params['points']
    # vision encoder
    plane_type   = ['xz', 'xy', 'yz']
    grid_resolution = 32 
    plane_resolution = 32
    # k = 20 * (args['num_input_points']//1000) # scale the knn (k) by the number of input points
    vision_encoder = models.vision_encoder.VNNLocalPoolPointnet(c_dim = int(v_enc_params['latent_size'] / 3),
                                                                    grid_resolution=grid_resolution,
                                                                    plane_type=plane_type,
                                                                    unet=args['unet'],
                                                                    plane_resolution=plane_resolution, 
                                                                    device=device,
                                                                    unet_depth=5).to(device)
    vision_decoder = models.vision_encoder.LocalDecoder(c_dim = v_enc_params['latent_size'])
    # Geometry encoders
    geometry_encoder = models.geometry_encoder.map_projected_points
    # 3D Points
    if 'loc' in points_params:
        points = models.points.get_3d_pts(n_points = points_params['n_points'],
                            loc=np.array(points_params['loc']),
                            scale=np.array(points_params['scale']))
    else:
        points = models.points.get_3d_pts(n_points=points_params['n_points'])
    # Energy Based Model
    in_dim = points_params['n_points']*feat_enc_params['out_dim']
    hidden_dim = 512
    feature_encoder = models.nets.TimeLatentFeatureEncoder(
            enc_dim=feat_enc_params['enc_dim'],
            latent_size= v_enc_params['latent_size'],
            dims = feat_enc_params['dims'],
            out_dim=feat_enc_params['out_dim'],
            dropout=feat_enc_params['dropout'],
            dropout_prob=feat_enc_params['dropout_prob'],
            norm_layers = feat_enc_params['norm_layers'],
            latent_in = feat_enc_params["latent_in"],
            xyz_in_all = feat_enc_params["xyz_in_all"],
            use_tanh = feat_enc_params["use_tanh"],
            latent_dropout = feat_enc_params["latent_dropout"],
            weight_norm= feat_enc_params["weight_norm"]
        )
    
    gaussian_mlp = nn.Sequential(
        nn.Linear(2 * in_dim, hidden_dim, bias=False),
        nn.LayerNorm(hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim, bias=False),
        nn.LayerNorm(hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim, bias=False),
        nn.LayerNorm(hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim, bias=False),
        nn.LayerNorm(hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim),
    )
    
    feature_decoder = nn.Sequential(
        nn.Linear(256, hidden_dim, bias=False),
        nn.LayerNorm(hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim//2, bias=False),
        nn.LayerNorm(hidden_dim//2),
        nn.ELU(),
        nn.Linear(hidden_dim//2, hidden_dim//4, bias=False),
        nn.LayerNorm(hidden_dim//4),
        nn.ELU(),
        nn.Linear(hidden_dim//4, hidden_dim//4, bias=False),
        nn.LayerNorm(hidden_dim//4),
        nn.ELU(),
        nn.Linear(hidden_dim//4, 12),
    )
    
    model = models.ConvGraspVAE(
        vision_encoder=vision_encoder, 
        geometry_encoder=geometry_encoder,
        points=points,
        vision_decoder=vision_decoder,
        feature_encoder=feature_encoder,
        gaussian_mlp=gaussian_mlp,
        feature_decoder=feature_decoder
    ).to(device=device)
    
    
    weights_path = './experiments_jul/dual_grasp_vae/checkpoints/model_current.pth'
    
    model_weights = torch.load(weights_path)
    if not inference:
        if 'model_pretrained' in weights_path:
            for name, param in model.named_parameters():
                if 'vision_encoder' in name or 'feature_encoder' in name:
                    param.data = model_weights[name]          
            print(f'Loaded Pretrained Weights for Vision Encoder and Feature Encoder from {weights_path}')
        else:
            ret = model.load_state_dict(model_weights, strict=False)
            print(ret)
            print(f'Loaded Pretrainedel_weights, Weights from {weights_path}')
        
    # print(model)
    return model