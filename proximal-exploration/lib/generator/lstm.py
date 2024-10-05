import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import math


class LSTMDecoder(nn.Module):
    def __init__(self, num_layers, hidden_dim, args, num_token=5):
        super(LSTMDecoder, self).__init__()
        self.encoder = nn.Embedding(num_token, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            batch_first=True,
            num_layers=num_layers,
        )
        self.decoder = nn.Linear(hidden_dim, num_token-1)
        self.start = num_token-1
        self.device = args.device

    def forward(self, batched_sequence_data):
        out = self.encoder(batched_sequence_data)
        out, _ = self.lstm(out, None)
        out = self.decoder(out)
        
        return out

    def decode(self, sample_size, max_len, argmax=False, random_action_prob=0.0, guide_seqs=None, explore_radius=1.0, temp=1):

        sequences = [torch.full((sample_size, 1), self.start, dtype=torch.long).to(self.device)]
        hidden = None
        
        with torch.no_grad():
            for i in range(max_len):
                out = self.encoder(sequences[-1])
                out, hidden = self.lstm(out, hidden)
                logit = self.decoder(out)
            
                prob = torch.softmax(logit/temp, dim=2)
                
                if guide_seqs is not None:
                    if type(explore_radius) == float:
                        explore_radius = torch.ones(prob.size(0)).to(prob.device) * explore_radius
                    mask = torch.rand(prob.size(0)).to(prob.device) >= explore_radius
                    
                    distribution = Categorical(probs=prob)
                    tth_sequences = distribution.sample()
                    
                    tth_sequences[mask] = guide_seqs[mask,i].long().unsqueeze(1)
                elif argmax:
                    tth_sequences = torch.argmax(logit, dim=2)
                else:
                    # uniform random exploration
                    if torch.rand(1).item() < random_action_prob:
                        rand_prob = torch.ones_like(prob) * 1/prob.shape[2]
                        distribution = Categorical(probs=rand_prob)
                    else:
                        distribution = Categorical(probs=prob)
                    tth_sequences = distribution.sample()
                sequences.append(tth_sequences)

            sequences = torch.cat(sequences, dim=1)

        return sequences[:, 1:]


class GFNLSTMGenerator(nn.Module):
    def __init__(self, args, max_len, partition_init = 50.0):
        super(GFNLSTMGenerator, self).__init__()
        
        self.max_len = max_len
        
        self.model = LSTMDecoder(args.lstm_num_layers, args.lstm_hidden_dim, args, num_token = args.vocab_size+1)
        self._Z = nn.Parameter(torch.ones(64).to(args.device) * partition_init / 64)
        
        self.model.to(args.device)
        # self._Z.to(args.device)
        
        self.reward_exp_min = args.reward_exp_min
        self.gen_clip = args.gen_clip

        self.opt = torch.optim.Adam(self.model.parameters(), args.gen_learning_rate, weight_decay=0.0,
                            betas=(0.9, 0.999))
        self.opt_Z = torch.optim.Adam([self._Z], args.gen_Z_learning_rate, weight_decay=0.0,
                            betas=(0.9, 0.999))
        self.device = args.device
        

    @property
    def Z(self):
        return self._Z.sum()
    
    def forward(self, batched_sequence_data):
        return self.model(batched_sequence_data)
    
    def decode(self, sample_size, argmax=False, random_action_prob=0.0, guide_seqs=None, explore_radius=1.0, temp=2.0):
        return self.model.decode(sample_size,
                                 max_len=self.max_len,
                                 argmax=argmax,
                                 random_action_prob=random_action_prob,  # Exploration for GFN-AL
                                 guide_seqs=guide_seqs,
                                 explore_radius=explore_radius,  # Trust region
                                 temp=temp)

    def train_step(self, batched_sequence_data, reward):
        start_tokens = torch.ones(batched_sequence_data.size(0), 1).long().to(self.device) * self.model.start
        batched_sequence_data = torch.cat([start_tokens, batched_sequence_data], dim=1)
        
        logits = self.model(batched_sequence_data)[:, 1:, :]  # exclude the start token
        log_pf = F.log_softmax(logits, dim=-1)
        
        log_pf= torch.gather(log_pf, 2, batched_sequence_data[:, 1:].long().unsqueeze(2)).squeeze(2)
        sum_log_pf = log_pf.sum(-1)
        
        loss = (self.Z + sum_log_pf - reward.clamp(min=self.reward_exp_min).log()).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gen_clip)
        self.opt.step()
        self.opt_Z.step()
        self.opt.zero_grad()
        self.opt_Z.zero_grad()
        
        return loss
    
    def get_log_prob(self, batched_sequence_data):
        
        start_tokens = torch.ones(batched_sequence_data.size(0), 1).long().to(self.device) * self.model.start
        batched_sequence_data = torch.cat([start_tokens, batched_sequence_data], dim=1)
        
        logits = self.model(batched_sequence_data)[:, 1:, :]  # exclude the start token
        log_pf = F.log_softmax(logits, dim=-1)
        
        log_pf= torch.gather(log_pf, 2, batched_sequence_data[:, 1:].long().unsqueeze(2)).squeeze(2)
        sum_log_pf = log_pf.sum(-1)
        
        return sum_log_pf