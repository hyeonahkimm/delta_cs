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
        # self.gen_train_batch_size = 64  # args.batch_size
        self.num_random_mutations = args.num_random_mutations
        self.frontier_neighbor_size = args.frontier_neighbor_size
        self.tokenizer = get_tokenizer(args, alphabet)
        self.dataset = None
        self.round = 0
        # self.dataset = BioSeqDataset(args, self.tokenizer)
        # self.vocab_size = args.vocab_size + 1
        # self.start = args.vocab_size
        
        # hyperparameters from GFN-AL
        args.vocab_size = len(alphabet)
        args.gen_reward_exp_ramping = 3.
        args.gen_reward_exp = 2.
        args.gen_reward_norm = 1.
        args.reward_exp_min = 1e-32
        args.gen_clip = 10.
        # args.gen_train_batch_size = 32
        # args.gen_sampling_temperature = 2.0
        # args.K = 50
        
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
        if len(measured_sequence_set)<=self.args.num_starting_sequences:
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
        candidates = self._train_generator(generator, frontier_neighbors, t=self.round)
        # rs = self.model.get_fitness(candidates)
        
        # import pdb; pdb.set_trace()
        
        
        # print(len(candidate_pool))
        # if len(candidate_pool) < self.num_model_queries_per_round: #self.args.K > 0:
        if self.args.K > 0:
            batch_size = self.args.num_queries_per_round
            candidates = []
            guide = None
            # for k in range(self.args.K):
            while len(candidates) < self.args.num_queries_per_round * self.args.K: #self.args.num_model_queries_per_round:
                if self.args.radius_option == "none":  # default GFN-AL
                    seqs = generator.decode(batch_size, random_action_prob=0.001, temp=self.args.gen_sampling_temperature)
                else:
                    # radius = 0.0 #0.5 + 0.5 * (t+1) / self.args.num_rounds
                    if guide is None:
                        if self.args.frontier_neighbor_size > 0 and not self.args.start_from_data:
                            ref = [random.choice(frontier_neighbors)['sequence'] for _ in range(batch_size)]
                            x = self.tokenizer.encode(ref)
                        else:
                            x, _ = self.dataset.weighted_sample(batch_size, self.args.rank_coeff)
                            ref = self.tokenizer.decode(x)
                            x = np.array(x)
                        guide = torch.tensor(x).to(self.args.device)
                    else: 
                        ref = self.tokenizer.decode(guide)
                    with torch.no_grad():
                        ys, std = self.model.get_fitness(ref, return_std=True)
                    radius = get_current_radius(iter=1000, round=self.round, args=self.args, std=std)
                    # import pdb; pdb.set_trace()
                    # guide = torch.tensor(np.array(x)).to(self.args.device)
                    seqs = generator.decode(batch_size, guide_seqs=guide, explore_radius=radius, temp=self.args.gen_sampling_temperature, back_and_forth=self.args.back_and_forth)
                decoded_seqs = self.tokenizer.decode(seqs.cpu().numpy())
                
                if self.args.use_mh and self.args.radius_option != "none":
                    with torch.no_grad():
                        rs = self.model.get_fitness(decoded_seqs)  # np.array (100,) do we have to consider proxy scores too?
                        log_p = generator.get_log_prob(seqs)
                        ref_log_p = generator.get_log_prob(guide)
                        accept_mask = (log_p - ref_log_p) > 0
                        accept_mask += (torch.rand(accept_mask.shape) < 0.1).to(accept_mask.device)
                        guide[accept_mask] = seqs[accept_mask]
                        # import pdb; pdb.set_trace()
                    # if (k+1) % 5 == 0:
                    #     # candidates.extend(decoded_seqs)  # updated guided sequences?
                    #     candidates.extend(self.tokenizer.decode(guide))  # updated guided sequences?self.tokenizer.decode(guide)
                    #     guide = None
                else:
                    candidates.extend(decoded_seqs)
                    guide = None
                    for candidate_sequence in decoded_seqs:
                        if candidate_sequence not in measured_sequence_set and candidate_sequence not in candidates:
                            candidates.append(candidate_sequence)
                            measured_sequence_set.add(candidate_sequence)
            candidate_pool = candidates
        else:
            # candidate_pool = candidates
            candidate_pool = []
            for candidate_sequence in candidates:
                if candidate_sequence not in measured_sequence_set:
                    candidate_pool.append(candidate_sequence)
                    measured_sequence_set.add(candidate_sequence)
        
        # scores = self.model.get_fitness(candidate_pool)
        scores = np.concatenate([
            self.model.get_fitness(candidate_pool[i:i+self.batch_size])
            for i in range(0, len(candidate_pool), self.batch_size)
        ])
        idx_pick = np.argsort(scores)[::-1][:self.args.num_queries_per_round]
        
        if self.args.frontier_neighbor_size == 0:
            return np.array(candidate_pool)[idx_pick], scores[idx_pick]
        
        # candidate_pool = []
        # while len(candidate_pool) < self.num_model_queries_per_round:
        #     candidate_sequence = random_mutation(random.choice(frontier_neighbors)['sequence'], self.alphabet, self.num_random_mutations)
        #     if candidate_sequence not in measured_sequence_set:
        #         candidate_pool.append(candidate_sequence)
        #         measured_sequence_set.add(candidate_sequence)
        # import pdb; pdb.set_trace()
        # return np.array(candidate_pool)[idx_pick]
        
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
        # print(np.mean(candidate_pool_dict.keys()))
        # import pdb; pdb.set_trace()
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
    
    def _train_generator(self, generator, frontier_neighbors, t=0):
        losses = []
        candidates = []
        batch_size = int(self.args.gen_train_batch_size / 2)
        p_bar = tqdm(range(self.args.generator_train_epochs)) # tqdm(range((self.args.num_model_queries_per_round - self.num_queries_per_round * self.args.K) // batch_size))
        
        for it in p_bar:
            p_bar_log = {}
            if self.args.radius_option == "none" and it > self.args.warmup_iter:  # default GFN-AL
                seqs = generator.decode(batch_size, random_action_prob=self.args.gen_random_action_prob, temp=self.args.gen_sampling_temperature)
                radius = 1.
                # with torch.no_grad():
                #     ys, std = self.model.get_fitness(self.tokenizer.decode(seqs), return_std=True)
            else:
                # radius = 0.0 #0.5 + 0.5 * (t+1) / self.args.num_rounds
                x, _ = self.dataset.weighted_sample(batch_size, self.args.rank_coeff)
                # ref = [random.choice(frontier_neighbors)['sequence'] for _ in range(batch_size)]
                # x = self.tokenizer.encode(ref)
                guide = torch.from_numpy(np.stack(x)).to(self.args.device)
                # import pdb; pdb.set_trace()
                with torch.no_grad():
                    ys, std = self.model.get_fitness(self.tokenizer.decode(x), return_std=True)
                # p_bar.set_postfix({"std": std.mean()})
                radius = get_current_radius(it, t, self.args, std=std)
                # import pdb; pdb.set_trace()
                # guide = torch.tensor(np.array(x)).to(self.args.device)
                seqs = generator.decode(batch_size, guide_seqs=guide, explore_radius=radius, temp=self.args.gen_sampling_temperature, back_and_forth=self.args.back_and_forth)
                p_bar_log = {"std": std.mean(), "radius": radius if isinstance(radius, float) else radius.mean().item()}

            # on_seqs = generator.decode(batch_size, guide_seqs=guide, explore_radius=torch.ones(radius.shape).to(radius.device), temp=self.args.gen_sampling_temperature)
            # seqs = torch.cat([seqs, on_seqs], dim=0)

            # offline data (both)
            # off_x, _ = self.dataset.sample(batch_size)  # rank-based? weighted_sample(batch_size)
            off_x, _ = self.dataset.weighted_sample(batch_size, self.args.rank_coeff)  # rank-based? weighted_sample(batch_size)
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
            # losses.append(loss.item())
            # wandb_log = {"generator_total_loss": loss.item()}
            # if self.args.use_wandb:
            #     wandb.log(wandb_log)
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
