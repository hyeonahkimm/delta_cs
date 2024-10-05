import torch
import random
import numpy as np

from tqdm import tqdm

from . import register_algorithm
from utils.seq_utils import hamming_distance, random_mutation
from lib.generator.lstm import GFNLSTMGenerator
from lib.utils.env import get_tokenizer


@register_algorithm("gfn-al")
class GFNGeneratorExploration:
    """
        GFlowNet-AL
    """
    def __init__(self, args, model, alphabet, starting_sequence):
        self.args = args
        self.model = model
        self.alphabet = alphabet
        self.wt_sequence = starting_sequence
        self.num_queries_per_round = args.num_queries_per_round
        self.num_model_queries_per_round = args.num_model_queries_per_round
        self.batch_size = args.batch_size
        self.num_random_mutations = args.num_random_mutations
        self.frontier_neighbor_size = args.frontier_neighbor_size
        self.tokenizer = get_tokenizer(args, alphabet)
        self.dataset = None
        self.round = 0
        
        # hyperparameters from GFN-AL
        args.vocab_size = len(alphabet)
        args.gen_reward_exp_ramping = 3.
        args.gen_reward_exp = 2.
        args.gen_reward_norm = 1.
        args.reward_exp_min = 1e-32
        args.gen_clip = 10.
        
        if args.gen_reward_exp_ramping > 0:
            self.l2r = lambda x, t=0: (x) ** (1 + (args.gen_reward_exp - 1) * (1 - 1/(1 + t / args.gen_reward_exp_ramping)))
        else:
            self.l2r = lambda x, t=0: (x) ** args.gen_reward_exp
    
    def propose_sequences(self, measured_sequences):
        # Input:  - measured_sequences: pandas.DataFrame
        #           - 'sequence':       [sequence_length]
        #           - 'true_score':     float
        # Output: - query_batch:        [num_queries, sequence_length]
        #         - model_scores:       [num_queries]
        
        query_batch, model_scores = self._propose_sequences(measured_sequences)
        if model_scores is None:
            model_scores = np.concatenate([
                self.model.get_fitness(query_batch[i:i+self.batch_size])
                for i in range(0, len(query_batch), self.batch_size)
            ])
        return query_batch, model_scores

    def _propose_sequences(self, measured_sequences):
        measured_sequence_set = set(measured_sequences['sequence'])
        
        # Generate random mutations in the first round.
        if len(measured_sequence_set)<=1:
            query_batch = []
            while len(query_batch) < self.num_queries_per_round:
                random_mutant = random_mutation(self.wt_sequence, self.alphabet, self.num_random_mutations)
                if random_mutant not in measured_sequence_set:
                    query_batch.append(random_mutant)
                    measured_sequence_set.add(random_mutant)
            return query_batch
        
        # Arrange measured sequences by the distance to the wild type.
        measured_sequence_dict = {}
        for _, data in measured_sequences.iterrows():
            distance_to_wt = hamming_distance(data['sequence'], self.wt_sequence)
            if distance_to_wt not in measured_sequence_dict.keys():
                measured_sequence_dict[distance_to_wt] = []
            measured_sequence_dict[distance_to_wt].append(data)
        
        # Highlight measured sequences near the proximal frontier.
        frontier_neighbors, frontier_height = [], -np.inf
        for distance_to_wt in sorted(measured_sequence_dict.keys()):
            data_list = measured_sequence_dict[distance_to_wt]
            data_list.sort(reverse=True, key=lambda x:x['true_score'])
            for data in data_list[:self.frontier_neighbor_size]:
                if data['true_score'] > frontier_height:
                    frontier_neighbors.append(data)
            frontier_height = max(frontier_height, data_list[0]['true_score'])

        # Construct the candiate pool by randomly mutating the sequences. (line 2 of Algorithm 2 in the paper)
        # An implementation heuristics: only mutating sequences near the proximal frontier.
        generator = GFNLSTMGenerator(self.args, max_len=len(self.wt_sequence), partition_init=self.args.partition_init)
        candidates = self._train_generator(generator, t=self.round)
        
        candidate_pool = []
        for candidate_sequence in candidates:
            if candidate_sequence not in measured_sequence_set:
                candidate_pool.append(candidate_sequence)
                measured_sequence_set.add(candidate_sequence)
        
        scores = np.concatenate([
            self.model.get_fitness(candidate_pool[i:i+self.batch_size])
            for i in range(0, len(candidate_pool), self.batch_size)
        ])
        idx_pick = np.argsort(scores)[::-1][:self.args.num_queries_per_round]
        
        if self.args.frontier_neighbor_size == 0:
            return np.array(candidate_pool)[idx_pick], scores[idx_pick]
        
        # Ours + PEX
        # Arrange the candidate pool by the distance to the wild type.
        candidate_pool_dict = {}
        for i in range(0, len(candidate_pool), self.batch_size):
            candidate_batch =  candidate_pool[i:i+self.batch_size]
            model_scores = self.model.get_fitness(candidate_batch)
            for candidate, model_score in zip(candidate_batch, model_scores):
                distance_to_wt = hamming_distance(candidate, self.wt_sequence)
                if distance_to_wt not in candidate_pool_dict.keys():
                    candidate_pool_dict[distance_to_wt] = []
                candidate_pool_dict[distance_to_wt].append(dict(sequence=candidate, model_score=model_score))
        for distance_to_wt in sorted(candidate_pool_dict.keys()):
            candidate_pool_dict[distance_to_wt].sort(reverse=True, key=lambda x:x['model_score'])
        
        # Construct the query batch by iteratively extracting the proximal frontier. 
        query_batch = []
        while len(query_batch) < self.num_queries_per_round:
            # Compute the proximal frontier by Andrew's monotone chain convex hull algorithm. (line 5 of Algorithm 2 in the paper)
            # https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain
            stack = []
            for distance_to_wt in sorted(candidate_pool_dict.keys()):
                if len(candidate_pool_dict[distance_to_wt])>0:
                    data = candidate_pool_dict[distance_to_wt][0]
                    new_point = np.array([distance_to_wt, data['model_score']])
                    def check_convex_hull(point_1, point_2, point_3):
                        return np.cross(point_2-point_1, point_3-point_1) <= 0
                    while len(stack)>1 and not check_convex_hull(stack[-2], stack[-1], new_point):
                        stack.pop(-1)
                    stack.append(new_point)
            while len(stack)>=2 and stack[-1][1] < stack[-2][1]:
                stack.pop(-1)
            
            # Update query batch and candidate pool. (line 6 of Algorithm 2 in the paper)
            for distance_to_wt, model_score in stack:
                if len(query_batch) < self.num_queries_per_round:
                    query_batch.append(candidate_pool_dict[distance_to_wt][0]['sequence'])
                    candidate_pool_dict[distance_to_wt].pop(0)
        
        return query_batch, None  # candidate_pool #
    
    def _train_generator(self, generator, t=0):
        losses = []
        candidates = []
        batch_size = int(self.args.gen_train_batch_size / 2)
        p_bar = tqdm(range(self.args.generator_train_epochs))
        
        for it in p_bar:
            p_bar_log = {}
            if self.args.radius_option == "none" and it > self.args.warmup_iter:  # default GFN-AL
                seqs = generator.decode(batch_size, random_action_prob=self.args.gen_random_action_prob, temp=self.args.gen_sampling_temperature)
                radius = 1.
            else:
                x, _ = self.dataset.weighted_sample(batch_size, self.args.rank_coeff)
                guide = torch.from_numpy(np.stack(x)).to(self.args.device)
                
                with torch.no_grad():
                    ys, std = self.model.get_fitness(self.tokenizer.decode(x), return_std=True)
                radius = get_current_radius(t, self.args, std=std)
                seqs = generator.decode(batch_size, guide_seqs=guide, explore_radius=radius, temp=self.args.gen_sampling_temperature)
                p_bar_log = {"std": std.mean(), "radius": radius if isinstance(radius, float) else radius.mean().item()}

            # offline data (both)
            off_x, _ = self.dataset.weighted_sample(batch_size, self.args.rank_coeff)
            if self.args.radius_option == "none" and it > self.args.warmup_iter:
                seqs = torch.tensor(np.array(off_x)).to(self.args.device)
            else:
                seqs = torch.cat([seqs, torch.tensor(np.array(off_x)).to(self.args.device)], dim=0)
            
            # seqs: tokenized tensor -> list of strings
            decoded = self.tokenizer.decode(seqs.cpu().numpy())
            rs = self.model.get_fitness(decoded)

            loss = generator.train_step(seqs, torch.from_numpy(self.l2r(rs)).to(seqs.device))

            p_bar_log['rs'] = rs.mean()
            p_bar_log['loss'] = loss.item()
            p_bar.set_postfix(p_bar_log)

            candidates.extend(decoded)
        return candidates


def get_current_radius(round, args, std=None):
    if args.radius_option == 'round_linear':
        return (args.max_radius-args.min_radius) * ((round+1)/args.num_rounds) + args.min_radius
    elif args.radius_option == 'fixed':
        return args.max_radius
    elif args.radius_option == 'proxy_var':
        r = (args.max_radius-args.min_radius) * ((round+1)/args.num_rounds) + args.min_radius
        return torch.from_numpy(r - args.sigma_coeff * std).to(args.device).clamp(0.001, 1.0)
    else:
        return 1.
