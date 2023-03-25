## NerfAcc visualization example

The majority of code is copy-pasted from the [nerfacc](https://github.com/KAIR-BAIR/nerfacc) repo

```
# Setup.
cd <PATH/TO/8_nerfacc>
pip install -r requirements.txt
git submodule update --init

# Train models with train_ngp_nerf*.py...
# python train_ngp_nerf.py --data_root <PATH/TO/BLENDER> --scene lego
# python train_ngp_nerf_prop.py --data_root <PATH/TO/360> --scene garden
# ... Or use provided checkpoints.
bash ../assets/download_nerfacc_checkpoints.sh

# Visualize using viser!
python visualize_ngp_nerf.py --data_root <PATH/TO/BLENDER> --scene lego
python visualize_ngp_nerf_prop.py --data_root <PATH/TO/360> --scene garden
```
