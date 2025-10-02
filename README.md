<!-- # DAGDiff: <ins>D</ins>ual-<ins>A</ins>rm <ins>G</ins>rasp <ins>Diff</ins>usion -->
# DAGDiff: Guiding Dual-Arm Grasp Diffusion to Stable and Collision-Free Grasps

This is the official repository for DAGDiff: Guiding Dual-Arm Grasp Diffusion to Stable and Collision-Free Grasps. The codebase and the documentation is still in progress. <br>

Check the <a href="https://dag-diff.github.io/dagdiff/">[Project Website]</a> for more results and updates.

## TODO
- [ ] : Add visualization notebook
- [ ] : Update documentation
- [ ] : Refactor training and eval code
- [x] : Push inference code and model checkpoint
- [x] : Conda env working fine 
- [x] : Initial release


## 1. Installation

### Creating the Conda Env
Run the following commands

```sh
conda create --name dagdiff -y python=3.8
conda activate dagdiff
```

Install Packages
```sh
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118  torchaudio==2.0.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+118.html # will take some time to install 
conda install conda-forge::suitesparse
conda install -c conda-forge scikit-sparse
pip install theseus-ai==0.1.3
```

Install remaining packages
```sh
pip install -r requirements.txt
pip install -e . # installing se3dif module
```

## 2. Download Dataset

Coming soon

## 3. Inference

For inference, first download the model checkpoint from <a href="https://iiithydresearch-my.sharepoint.com/:u:/g/personal/md_faizal_research_iiit_ac_in/EegOVM7li5xAsG7fFH9B4OIB07OSM7INiTIQDmiWpeRoFw?e=qU2po1">link</a> and place it in `./checkpoint` directory. The same path needs to be provided in `./configs/dual_arm_params.yaml` as <b>inference_checkpoint</b>. Two example object meshes are stored in `./try_meshes` directory which can be used to run the model. 

Once done, run the following command to generate dual-arm grasps. 

```sh
CUDA_VISIBLE_DEVICES=0 python3 scripts/sample/generate_dual_6d_grasp_poses.py \
--n_grasps 300 \
--model dual_arm_params \
--input ./try_meshes/monitor.obj
```

Use <a href="https://github.com/DAG-Diff/dual-arm-grasp-diffusion/blob/main/notebooks/viz_grasps.ipynb">viz_grasp.ipynb</a> to visualize the generated grasps and the denoising trajectory.

## 4. Training 

Coming soon

<!-- ## 6. Research Progression  

Our research is part of a continuing line of projects.
To see how it has developed over time, take a look at our earlier works:



```
[CGDF] ────┐------┐
         |        |  
         |        v
         ├─────> DG16M ────> DAGDiff
         |
         |
DAVIL ───┘

``` -->


## Acknowledgment
Our codebase is built upon the existing works of <a href="https://sites.google.com/view/se3dif">SE(3)-diff</a> and <a href="https://constrained-grasp-diffusion.github.io/">CGDF</a>. We thank the authors for releasing the code.

## Cite

