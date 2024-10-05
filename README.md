# Improved Off-policy Reinforcement Learning in  Biological Sequence Design

This repository provided implemented codes for the paper -- Improved Off-policy Reinforcement Learning in  Biological Sequence Design. 
> 

Our codes are implemented based on
- Proximal Exploration for Model-guided Protein Sequence Design ([paper](https://proceedings.mlr.press/v162/ren22a.html), [code](https://github.com/HeliXonProtein/proximal-exploration))
- Biological Sequence Design with GFlowNets ([paper](https://proceedings.mlr.press/v162/jain22a/jain22a.pdf), [code](https://github.com/MJ10/BioSeq-GFN-AL))


### Usage

#### TF-Bind-8
```
cd BioSeq-GFN-AL
python run_tfbind_delta.py --gen_do_explicit_Z 1 --acq_fn ucb --radius_option proxy_var --min_radius 0.5 --max_radius 0.5 --sigma_coeff 5
```

#### AMP
```
cd BioSeq-GFN-AL
python run_amp_delta.py --gen_do_explicit_Z 1 --acq_fn ucb --radius_option proxy_var --min_radius 0.5 --max_radius 0.5 --sigma_coeff 1 --use_rank_based_proxy_training
``` 

#### RNA and protein designs
```
cd proximal-exploration
python run_flexs.py --alg=gfn-al --net=cnn --task=rna1 --radius_option proxy_var --min_radius 0.5 --max_radius 0.5 --sigma_coeff 5 --use_rank_based_proxy_training
python run_flexs.py --alg=gfn-al --net=cnn --task=gfp --radius_option proxy_var --min_radius 0.05 --max_radius 0.05 --sigma_coeff 1 --use_rank_based_proxy_training
python run_flexs.py --alg=gfn-al --net=cnn --task=aav --radius_option proxy_var --min_radius 0.05 --max_radius 0.05 --sigma_coeff 0.1 --use_rank_based_proxy_training
```

`rna1`, `rna2`, and `rna3` correspond to RNA-A, RNA-B, and RNA-C, respectively.

Note: we use baselines implementation in [FLEXS](https://github.com/samsinai/FLEXS) 

