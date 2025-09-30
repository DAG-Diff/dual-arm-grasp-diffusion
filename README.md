# DAGDiff: <u>D</u>ual-<u>A</u>rm <u>G</u>rasp <u>Diff</u>usion

This is the official repository for DAGDiff: Guiding Dual-Arm Grasp Diffusion to Stable and Collision-Free Grasps. The codebase and the documentation is still in progress. <br>

Check the Project <a href="https://dag-diff.github.io/dagdiff/">[Website]</a> for more results and updates. 

## Installation

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

## Download Dataset

## Inference

## Training 

## Acknowledgment
Our codebase is built upon the existing works of <a href="https://sites.google.com/view/se3dif">SE(3)-diff</a> and <a href="https://constrained-grasp-diffusion.github.io/">CGDF</a>. We thank the authors for releasing the code.

## Cite


## TODO

- [ ] : Add visualization notebook
- [ ] : Update documentation
- [ ] : Refactor training and eval code
- [x] : Conda env working fine 
- [x] : Initial release
