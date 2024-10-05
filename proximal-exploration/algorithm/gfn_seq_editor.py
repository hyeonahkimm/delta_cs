import torch
import torch.nn.functional as F
from torch.distributions import Categorical

import random
import numpy as np

from tqdm import tqdm

from . import register_algorithm
from utils.seq_utils import hamming_distance, random_mutation
from lib.generator.lstm import GFNLSTMGenerator
from lib.utils.env import get_tokenizer


@register_algorithm("gfn_seq_editor")
class GFNGeneratorExploration:
    """
        GFlowNet Sequence Editor
    """
    def __init__(self, args, model, alphabet, starting_sequence):
        self.args = args
        self.model = model
        self.alphabet = alphabet
        self.wt_sequence = starting_sequence
        self.num_queries_per_round = args.num_queries_per_round
        # self.num_model_queries_per_round = args.num_model_queries_per_round
        self.batch_size = args.batch_size
        # self.gen_train_batch_size = 64  # args.batch_size
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
        
        query_batch = self._propose_sequences(measured_sequences)
        model_scores = np.concatenate([
            self.model.get_fitness(query_batch[i:i+self.batch_size])
            for i in range(0, len(query_batch), self.batch_size)
        ])
        print('after:', model_scores.mean())
        return query_batch, model_scores

    def _propose_sequences(self, measured_sequences, delta=0.1, sigma=0.001, lamb=0.1):
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

        # Construct the candiate pool by randomly mutating the sequences. (line 2 of Algorithm 2 in the paper)
        # An implementation heuristics: only mutating sequences near the proximal frontier.
        generator = GFNLSTMGenerator(self.args, max_len=len(self.wt_sequence))
        self._train_generator(generator)
        # rs = self.model.get_fitness(candidates)
        
        # off_x, off_y = self.dataset.weighted_sample(self.args.num_queries_per_round, self.args.rank_coeff)  # rank-based? weighted_sample(batch_size)
        # # import pdb; pdb.set_trace()
        # # off_x = np.array(self.tokenizer.encode(self.wt_sequence)).reshape(1, -1).repeat(self.args.num_queries_per_round, axis=0)
        # off_x = torch.tensor(np.array(off_x)).to(self.args.device)
        # print('before:', np.mean(off_y))
        # start_tokens = torch.ones(off_x.size(0), 1).long().to(seqs.device) * generator.model.start
        
        off_x = torch.tensor(self.tokenizer.encode(self.wt_sequence)).reshape(1, -1).repeat(self.args.num_queries_per_round, 1).to(self.args.device)
        
        sequences = [torch.full((self.args.num_queries_per_round, 1), generator.model.start, dtype=torch.long).to(self.args.device)]
        hidden = None
        
        for t in range(len(self.wt_sequence)):
            with torch.no_grad():
                out = generator.model.encoder(sequences[-1])
                out, hidden = generator.model.lstm(out, hidden)
                logit = generator.model.decoder(out)
                
                prob_pf = torch.softmax(logit, dim=2).squeeze(1)
                ref_actions = off_x[:, t].to(self.args.device).long()
                
                # ref = torch.gather(prob_pf, -1, ref_actions.view(-1, 1))
                # noise = torch.normal(0, sigma, size=ref.shape).to(ref.device)
                # sub_opt_identifiers = ref < delta * prob_pf.max(-1)[0].view(-1, 1) + noise
                
                ref = torch.gather(prob_pf, -1, ref_actions.view(-1, 1)).view(-1)
                noise = torch.normal(0, sigma, size=ref.shape).to(ref.device)
                sub_opt_identifiers = ref < delta * prob_pf.max(-1)[0].view(-1) + noise
                
                try:
                    cat = Categorical((1-lamb) * prob_pf + lamb * F.one_hot(ref_actions.long(), self.args.vocab_size).float().to(prob_pf.device))
                    # import pdb; pdb.set_trace()
                except:
                    import pdb; pdb.set_trace()
                    
                new_actions = cat.sample()
                
            actions = torch.where(sub_opt_identifiers.view(-1), new_actions.view(-1), ref_actions.view(-1))
            sequences.append(actions[:, None])
        sequences = torch.cat(sequences, dim=1)[:, 1:]
        
        ref_seqs = [''.join([str(i) for i in seq]) for seq in off_x.tolist()]
        new_seqs = [''.join([str(i) for i in seq]) for seq in sequences.tolist()]
        edit = np.mean([hamming_distance(s, ss) for s, ss in zip(new_seqs, ref_seqs)])
        print('edit dits:', edit)
        decoded_seqs = self.tokenizer.decode(sequences.cpu().numpy())
        # import pdb; pdb.set_trace()
        return decoded_seqs
    
    def _train_generator(self, generator):
        batch_size = self.args.gen_train_batch_size
        p_bar = tqdm(range(self.args.generator_train_epochs)) # tqdm(range((self.args.num_model_queries_per_round - self.num_queries_per_round * self.args.K) // batch_size))
        
        for it in p_bar:
            p_bar_log = {}
            # offline data (both)
            off_x, off_y = self.dataset.sample(batch_size)  # rank-based? weighted_sample(batch_size)
            # off_x, off_y = self.dataset.weighted_sample(batch_size, self.args.rank_coeff)  # rank-based? weighted_sample(batch_size)
            seqs = torch.tensor(np.array(off_x)).to(self.args.device)
            rs = torch.tensor(off_y).to(seqs.device)

            loss = generator.train_step(seqs, self.l2r(rs))

            p_bar_log['rs'] = rs.mean().item()
            p_bar_log['loss'] = loss.item()
            p_bar.set_postfix(p_bar_log)

            # losses.append(loss.item())
            # wandb_log = {"generator_total_loss": loss.item()}
            # if self.args.use_wandb:
            #     wandb.log(wandb_log)
        return None
