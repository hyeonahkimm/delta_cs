import torch
import numpy as np

import argparse
import wandb

from lib.dataset import get_dataset
from lib.oracle_wrapper import get_oracle


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Task
    parser.add_argument('method', default='gfn_al')
    parser.add_argument("--task", default="tfbind", type=str, choices=["tfbind", "gfp", "aav", "rna1", "rna2", "rna3"])  # add gfp rna14 rna100 aav rosetta (and tasks in PEX?)
    
    # General
    parser.add_argument("--save_path", default='results/test_mlp.pkl.gz')
    parser.add_argument("--tb_log_dir", default='results/test_mlp')
    parser.add_argument("--load_scores_path", default='.')
    parser.add_argument("--name", default='test_mlp')
    parser.add_argument("--use_wandb", action="store_true")
    
    parser.add_argument("--save_scores_path", default=".")
    # parser.add_argument("--save_scores", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    # parser.add_argument("--run", default=-1, type=int)
    # parser.add_argument("--noise_params", action="store_true")
    # parser.add_argument("--enable_tensorboard", action="store_true")
    # parser.add_argument("--save_proxy_weights", action="store_true")
    # parser.add_argument("--use_uncertainty", action="store_true")
    
    args, _ = parser.parse_known_args()
    
    ############# temp!!
    parser.add_argument("--rank_based_proxy_training", action="store_true")
    parser.add_argument("--reward_prioritized", action="store_true")
        
    if args.task == 'amp':
        parser.add_argument("--num_rounds", default=10, type=int)
        parser.add_argument("--num_queries_per_round", default=1024, type=int) # 10k
        parser.add_argument("--num_folds", default=5)
        parser.add_argument("--vocab_size", default=21)
        parser.add_argument("--max_len", default=65)
        parser.add_argument("--gen_max_len", default=50+1)
        parser.add_argument("--proxy_num_iterations", default=30000, type=int)
        parser.add_argument("--gen_num_iterations", default=10000, type=int)
    elif args.task == 'tfbind':
        parser.add_argument("--num_rounds", default=10, type=int)
        parser.add_argument("--num_queries_per_round", default=128, type=int)  # the current setting is 100 (mismatch)
        parser.add_argument("--vocab_size", default=4)
        parser.add_argument("--max_len", default=8)
        parser.add_argument("--gen_max_len", default=8)
        parser.add_argument("--proxy_num_iterations", default=3000, type=int)
        parser.add_argument("--gen_num_iterations", default=5000, type=int)
    elif args.task.startswith('rna'):
        parser.add_argument("--num_rounds", default=10, type=int)
        parser.add_argument("--num_queries_per_round", default=128, type=int)
        parser.add_argument("--vocab_size", default=4)
        parser.add_argument("--max_len", default=14)
        parser.add_argument("--gen_max_len", default=14)
        parser.add_argument("--proxy_num_iterations", default=3000, type=int)
        parser.add_argument("--gen_num_iterations", default=5000, type=int)
    elif args.task == 'gfp':
        parser.add_argument("--num_rounds", default=10, type=int)
        parser.add_argument("--num_queries_per_round", default=128, type=int)
        parser.add_argument("--vocab_size", default=20)
        parser.add_argument("--max_len", default=238)
        parser.add_argument("--gen_max_len", default=238)
        parser.add_argument("--proxy_num_iterations", default=3000, type=int)
        parser.add_argument("--gen_num_iterations", default=10000, type=int)
    elif args.task == 'aav':
        parser.add_argument("--num_rounds", default=10, type=int)
        parser.add_argument("--num_queries_per_round", default=128, type=int)
        parser.add_argument("--vocab_size", default=20)
        parser.add_argument("--max_len", default=90)
        parser.add_argument("--gen_max_len", default=90)
        parser.add_argument("--proxy_num_iterations", default=3000, type=int)
        parser.add_argument("--gen_num_iterations", default=10000, type=int)
        
    # if args.method.startswith('gfn'):
    if args.method == 'gfn_seq_editor':
        parser.add_argument("--edit_identifier_sigma", default=0.001, type=float)
        parser.add_argument("--lstm_num_layers", default=2, type=int)
        parser.add_argument("--lstm_hidden_dim", default=512, type=int)
        parser.add_argument("--beta", default=1, type=int)
        parser.add_argument("--use_rank_based_off", action='store_true')
    elif args.method == 'gfn_lstm':
        parser.add_argument("--lstm_num_layers", default=2, type=int)
        parser.add_argument("--lstm_hidden_dim", default=512, type=int)
        parser.add_argument("--sample_width", default=100, type=int)
        parser.add_argument("--beta", default=1, type=int)
        parser.add_argument("--warmup_iter", default=0, type=int)
        parser.add_argument("--use_rank_based_off", action='store_true')
    parser.add_argument("--rank_coeff", default=0.01, type=float)
        
    parser.add_argument("--radius_option", default='none', type=str)
    parser.add_argument("--min_radius", default=0.5, type=float)
    parser.add_argument("--max_radius", default=1.0, type=float)
    parser.add_argument("--sigma_coeff", default=10, type=float)
    
    # Proxy (shared for GFNs)
    parser.add_argument("--acq_fn", default="none", type=str)
    parser.add_argument("--proxy_uncertainty", default="dropout")
    
    #### move to gfn.utils get_gfn_args
    parser.add_argument("--filter", action="store_true")
    parser.add_argument("--kappa", default=0.1, type=float)
    parser.add_argument("--max_percentile", default=80, type=int)
    parser.add_argument("--filter_threshold", default=0.1, type=float)
    parser.add_argument("--filter_distance_type", default="edit", type=str)
    parser.add_argument("--oracle_split", default="D2_target", type=str)
    parser.add_argument("--oracle_type", default="MLP", type=str)
    parser.add_argument("--oracle_features", default="AlBert", type=str)
    parser.add_argument("--medoid_oracle_dist", default="edit", type=str)
    parser.add_argument("--medoid_oracle_norm", default=1, type=int)
    parser.add_argument("--medoid_oracle_exp_constant", default=6, type=int)

    parser.add_argument("--load_proxy_weights", type=str)
    parser.add_argument("--proxy_data_split", default="D1", type=str)
    parser.add_argument("--proxy_learning_rate", default=1e-4)
    parser.add_argument("--proxy_type", default="regression")
    parser.add_argument("--proxy_arch", default="mlp")
    parser.add_argument("--proxy_num_layers", default=2)
    parser.add_argument("--proxy_dropout", default=0.1)

    parser.add_argument("--proxy_num_hid", default=2048, type=int)
    parser.add_argument("--proxy_L2", default=1e-4, type=float)
    parser.add_argument("--proxy_num_per_minibatch", default=256, type=int)
    parser.add_argument("--proxy_early_stop_tol", default=5, type=int)
    parser.add_argument("--proxy_early_stop_to_best_params", default=0, type=int)
    # parser.add_argument("--proxy_num_iterations", default=3000, type=int)
    parser.add_argument("--proxy_num_dropout_samples", default=25, type=int)
    parser.add_argument("--proxy_pos_ratio", default=0.9, type=float)

    # Generator
    parser.add_argument("--gen_learning_rate", default=1e-5, type=float)
    parser.add_argument("--gen_Z_learning_rate", default=1e-3, type=float)
    parser.add_argument("--gen_clip", default=10, type=float)
    parser.add_argument("--gen_episodes_per_step", default=16, type=int)
    parser.add_argument("--gen_num_hidden", default=2048, type=int)
    parser.add_argument("--gen_reward_norm", default=1, type=float)
    parser.add_argument("--gen_reward_exp", default=2, type=float)
    parser.add_argument("--gen_reward_min", default=0, type=float)
    parser.add_argument("--gen_L2", default=0, type=float)
    parser.add_argument("--gen_partition_init", default=50, type=float)

    # Soft-QLearning/GFlownet gen
    parser.add_argument("--gen_reward_exp_ramping", default=3, type=float)
    parser.add_argument("--gen_balanced_loss", default=1, type=float)
    parser.add_argument("--gen_output_coef", default=10, type=float)
    parser.add_argument("--gen_loss_eps", default=1e-5, type=float)
    parser.add_argument("--gen_random_action_prob", default=0.001, type=float)
    parser.add_argument("--gen_sampling_temperature", default=2., type=float)
    parser.add_argument("--gen_leaf_coef", default=25, type=float)
    parser.add_argument("--gen_data_sample_per_step", default=16, type=int)
    # PG gen
    parser.add_argument("--gen_do_pg", default=0, type=int)
    parser.add_argument("--gen_pg_entropy_coef", default=1e-2, type=float)
    # learning partition Z explicitly
    parser.add_argument("--gen_do_explicit_Z", default=1, type=int)
    parser.add_argument("--gen_model_type", default="mlp")

    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = get_args()
    
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    oracle = get_oracle(args)
    dataset = get_dataset(args, oracle)
    
    if args.use_wandb:
        run = wandb.init(project='gfn_al_0919', group=args.task, config=args, reinit=True)
        wandb.run.name = f"{args.name}_{str(args.seed)}_{wandb.run.id}"
        
    if args.method == 'gfn_seq_editor':
        from lib.algorithms.GFNSeqEditor.runner import GFNSeqEditorRunner, GFNSeqEditorLSTMRunner
        args.reward_exp_min = 1e-32
        runner = GFNSeqEditorLSTMRunner(args, oracle, dataset)
    elif args.method == 'gfn_lstm':
        args.reward_exp_min = 1e-32
        from lib.algorithms.gfn_lstm.runner import GFNLSTMRunner
        runner = GFNLSTMRunner(args, oracle, dataset)
    # train(args, oracle, dataset)
    runner.run()
    
    if args.use_wandb:
        wandb.finish()
