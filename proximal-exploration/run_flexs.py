import random
import torch
import wandb

import flexs.utils.sequence_utils as s_utils

import numpy as np
from landscape import get_landscape, task_collection, landscape_collection
from algorithm import get_algorithm, algorithm_collection
from model import get_model, model_collection
from model.ensemble import ensemble_rules
from utils.os_utils import get_arg_parser
from utils.eval_utils import Runner


def get_args():
    parser = get_arg_parser()
    
    parser.add_argument('--device', help='device', type=str, default='cuda')
    
    # landscape arguments
    parser.add_argument('--task', help='fitness landscape', type=str, default='tfbind', choices=task_collection.keys())
    parser.add_argument('--oracle_model', help='oracle model of fitness landscape', type=str, default='flexs', choices=landscape_collection.keys())

    # algorithm arguments
    parser.add_argument('--alg', help='exploration algorithm', type=str, default='pex', choices=algorithm_collection.keys())
    parser.add_argument('--num_rounds', help='number of query rounds', type=np.int32, default=10)
    parser.add_argument('--num_queries_per_round', help='number of black-box queries per round', type=np.int32, default=128)
    parser.add_argument('--num_model_queries_per_round', help='number of model predictions per round', type=np.int32, default=2000)
    parser.add_argument('--num_model_max_epochs', help='number of model predictions per round', type=np.int32, default=3000)
    
    # model arguments
    parser.add_argument('--net', help='surrogate model architecture', type=str, default='cnn', choices=model_collection.keys())   # PEX default: 'mufacnet'
    parser.add_argument('--lr', help='learning rate', type=np.float32, default=1e-3)
    parser.add_argument('--batch_size', help='batch size', type=int, default=256)
    parser.add_argument('--patience', help='number of epochs without improvement to wait before terminating training', type=np.int32, default=10)
    parser.add_argument('--ensemble_size', help='number of model instances in ensemble', type=np.int32, default=3)
    parser.add_argument('--ensemble_rule', help='rule to aggregate the ensemble predictions', type=str, default='ucb', choices=ensemble_rules.keys())  # PEX default: 'mean'
    
    parser.add_argument('--seed',  type=np.int32, default=0)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument('--name', type=str, default='default')

    args, _ = parser.parse_known_args()
    
    # PEX arguments
    if args.alg == 'pex':
        parser.add_argument('--num_random_mutations', help='number of amino acids to mutate per sequence', type=np.int32, default=2)
        parser.add_argument('--frontier_neighbor_size', help='size of the frontier neighbor', type=np.int32, default=5)
    elif args.alg == 'gfn-al':
        parser.add_argument('--radius_option', default='none')
        parser.add_argument("--lstm_num_layers", default=2, type=int)
        parser.add_argument("--lstm_hidden_dim", default=512, type=int)
        parser.add_argument("--gen_train_batch_size", default=256, type=int)
        parser.add_argument('--gen_learning_rate', help='learning rate', type=float, default=5e-4)
        parser.add_argument('--gen_Z_learning_rate', help='Z learning rate', type=float, default=1e-3)
        parser.add_argument('--max_radius', type=float, default=0.05)
        parser.add_argument('--min_radius', type=float, default=0.0)
        parser.add_argument('--sigma_coeff', type=float, default=1.0)
        parser.add_argument('--rank_coeff', type=float, default=0.01)
        parser.add_argument('--gen_sampling_temperature', type=float, default=2.0)  # from BioSeq GFN-AL
        parser.add_argument('--gen_random_action_prob', type=float, default=0.001)  # from BioSeq GFN-AL
        parser.add_argument('--frontier_neighbor_size', help='size of the frontier neighbor', type=np.int32, default=0)
        parser.add_argument('--num_random_mutations', help='number of amino acids to mutate per sequence', type=np.int32, default=2)  # for PEX
        parser.add_argument('--warmup_iter',type=np.int32, default=0)
        parser.add_argument('--start_from_data', action='store_true')
        
    parser.add_argument('--generator_train_epochs', help='number of model predictions per round', type=np.int32, default=1000)
    parser.add_argument('--use_rank_based_proxy_training', action='store_true')
    
    # MuFacNet arguments
    if args.net == 'mufacnet':
        parser.add_argument('--latent_dim', help='dimension of latent mutation embedding', type=np.int32, default=32)
        parser.add_argument('--context_radius', help='the radius of context window', type=np.int32, default=10)

    args = parser.parse_args()
    return args


def get_initial_dataset(task_name):
    stoi = dict(enumerate(task_collection[task_name.lower()]))
    if task_name.lower() == 'tfbind':
        encoded = np.load("../dataset/tfbind/tfbind-x-init.npy")
        x = np.array([''.join([stoi[c] for c in seq]) for seq in encoded])
        y = np.load("../dataset/tfbind/tfbind-y-init.npy").reshape(-1)
    elif task_name.lower().startswith("rna"):
        encoded = np.load(f"../dataset/rna/{task_name.upper()}_x.npy")
        x = np.array([''.join([stoi[c] for c in seq]) for seq in encoded])
        y = np.load(f"../dataset/rna/{task_name.upper()}_y.npy").reshape(-1)
    elif task_name.lower() == 'gfp':
        x = np.load("../dataset/gfp/gfp-x-init.npy")
        y = np.load("../dataset/gfp/gfp-y-init.npy").reshape(-1)
    elif task_name.lower() == 'aav':
        x = np.load("../dataset/aav/aav-x-init.npy")
        y = np.load("../dataset/aav/aav-y-init.npy").reshape(-1)
    else:
        raise ValueError(f"Unknown task: {task_name}")
    # x, y = x[:1000], y[:1000]
    return x, y, x[y.argmax()]


if __name__=='__main__':
    args = get_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed_all(args.seed)
        
    if args.use_wandb:
        run = wandb.init(project='bioseq_0928', group=args.task, config=args, reinit=True)
        wandb.run.name = f"{args.alg}_{args.name}_{str(args.seed)}_{wandb.run.id}"
    
    landscape, alphabet, starting_sequence = get_landscape(args)
    starting_sequences, starting_scores, ref = get_initial_dataset(args.task)
    starting_sequence = ref if starting_sequence is None else starting_sequence
    # import pdb; pdb.set_trace()
    
    # print(starting_sequence)
    model = get_model(args, alphabet=alphabet, starting_sequence=starting_sequence)
    explorer = get_algorithm(args, model=model, alphabet=alphabet, starting_sequence=starting_sequence)

    runner = Runner(args)
    runner.run(landscape, starting_sequence, model, explorer, starting_dataset=(starting_sequences, starting_scores))
    
    if args.use_wandb:
        wandb.finish()

