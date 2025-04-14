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
        self.tokenizer = get_tokenizer(args, alphabet)
        self.dataset = None
        self.round = 0
        
        # hyperparameters from GFN-AL (Jain et al., 2022; https://github.com/MJ10/BioSeq-GFN-AL)
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
        # if len(measured_sequence_set)<=self.args.num_starting_sequences:
        #     query_batch = []
        #     while len(query_batch) < self.num_queries_per_round:
        #         random_mutant = random_mutation(self.wt_sequence, self.alphabet, 2)
        #         if random_mutant not in measured_sequence_set:
        #             query_batch.append(random_mutant)
        #             measured_sequence_set.add(random_mutant)
        #     return query_batch
        
        # Arrange measured sequences by the distance to the wild type.
        measured_sequence_dict = {}
        for _, data in measured_sequences.iterrows():
            distance_to_wt = hamming_distance(data['sequence'], self.wt_sequence)
            if distance_to_wt not in measured_sequence_dict.keys():
                measured_sequence_dict[distance_to_wt] = []
            measured_sequence_dict[distance_to_wt].append(data)
        
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
        
        return np.array(candidate_pool)[idx_pick], scores[idx_pick]
    
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

                radius = get_current_radius(it, t, self.args, std=std)
                seqs = generator.decode(batch_size, guide_seqs=guide, explore_radius=radius, temp=self.args.gen_sampling_temperature, back_and_forth=self.args.back_and_forth)
                
                p_bar_log = {"std": std.mean(), "radius": radius if isinstance(radius, float) else radius.mean().item()}


            # offline data (both)
            off_x, _ = self.dataset.weighted_sample(batch_size, self.args.rank_coeff)  # rank-based
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


def get_current_radius(iter, round, args, std=None):
    if args.radius_option == 'linear':
        return (args.max_radius-args.min_radius) * ((round+1)/args.num_rounds) + args.min_radius
    elif args.radius_option == 'constant':
        return args.max_radius
    elif args.radius_option == 'adaptive':
        return torch.from_numpy(args.max_radius - args.sigma_coeff * std).to(args.device).clamp(0.001, 1.0)
    elif args.radius_option == 'adaptive_linear':
        return torch.from_numpy((args.max_radius-args.min_radius) * ((round)/args.num_rounds) + args.min_radius - std).to(args.device)
    else:
        return 1.
