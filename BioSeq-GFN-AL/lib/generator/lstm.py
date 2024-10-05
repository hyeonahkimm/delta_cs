import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import math


class LSTMModel(nn.Module):
    def __init__(self, num_layers, hidden_dim, args, num_tokens=5):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(num_tokens, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            batch_first=True,
            num_layers=num_layers,
        )
        self.fc = nn.Linear(hidden_dim, num_tokens-1)  # exclude the start token
        self.device = args.device
        self.start_token = num_tokens - 1
    
    def forward(self, x, hidden_state=None):
        x = self.embedding(x)
        lstm_out, hidden_state = self.lstm(x, hidden_state)
        out = self.fc(lstm_out)  # (batch_size, seq_length, num_tokens)
        return out, hidden_state
    
    def decode(self, sample_size, max_length=90, random_action_prob=0.0, guide_seqs=None, explore_radius=1.0, temp=1.0):
        with torch.no_grad():
            # Initialize the input tokens (start_token for all sequences in the batch)
            input_tokens = torch.full((sample_size, 1), self.start_token, dtype=torch.long).to(self.device)  # (batch_size, 1)
            generated_sequences = torch.full((sample_size, max_length+1), self.start_token, dtype=torch.long).to(self.device)  # (batch_size, length)
            
            # Initialize hidden state for LSTM
            hidden_state = None
            
            for t in range(1, max_length+1):
                # Forward pass through the model
                logits, hidden_state = self.forward(input_tokens, hidden_state)
                
                prob = torch.softmax(logits/temp, dim=2)
                
                # Predict next token (take output for the last time step)
                # next_token = torch.argmax(logits[:, -1, :], dim=-1)  # (batch_size,)
                if guide_seqs is not None:
                    distribution = Categorical(prob)
                    next_token = distribution.sample()
                    mask = torch.rand(explore_radius.size(0)).to(self.device) > explore_radius.view(-1) # follow guide
                    # import pdb; pdb.set_trace()
                    next_token[mask] = guide_seqs[:,t-1].long()[mask].unsqueeze(1)
                else:
                    # uniform random exploration
                    if torch.rand(1).item() < random_action_prob:
                        rand_prob = torch.ones_like(prob) * 1/prob.shape[2]
                        distribution = Categorical(probs=rand_prob)
                    else:
                        distribution = Categorical(probs=prob)
                    next_token = distribution.sample()
                
                # Store the next token in the generated_sequences
                generated_sequences[:, t] = next_token.view(-1)
                
                # Update the input tokens for the next step
                input_tokens = next_token  #.unsqueeze(1) # Reshape to (batch_size, 1)
        
        return generated_sequences

class GFNLSTMGenerator(nn.Module):
    def __init__(self, args, partition_init = 50.0):
        super(GFNLSTMGenerator, self).__init__()
        self.model = LSTMModel(args.lstm_num_layers, args.lstm_hidden_dim, args, num_tokens = args.vocab_size+1)
        self._Z = torch.nn.Parameter(torch.tensor([5.]).cuda()) # nn.Parameter(torch.ones(64).to(args.device) * partition_init / 64)
        
        self.model.to(args.device)
        # self._Z.to(args.device)
        
        self.reward_exp_min = args.reward_exp_min
        self.gen_clip = args.gen_clip

        self.opt = torch.optim.Adam(self.model.parameters(), args.gen_learning_rate, weight_decay=args.gen_L2,
                            betas=(0.9, 0.999))
        self.opt_Z = torch.optim.Adam([self._Z], args.gen_Z_learning_rate, weight_decay=args.gen_L2,
                            betas=(0.9, 0.999))
        self.device = args.device
        self.beta = args.beta

    @property
    def Z(self):
        return self._Z.sum()
    
    def forward(self, batched_sequence_data):
        start_tokens = torch.ones(batched_sequence_data.size(0), 1).long().to(self.device) * self.model.start_token
        batched_sequence_data = torch.cat([start_tokens, batched_sequence_data], dim=1)
        logits, _ = self.model(batched_sequence_data)
        return logits[:, :-1] # logits
    
    def decode(self, sample_size, max_len, argmax=False, random_action_prob=0.0, guide_seqs=None, explore_radius=1.0, temp=1.0):
        with torch.no_grad():
            seqs = self.model.decode(sample_size, 
                                     max_length=max_len, 
                                     random_action_prob=random_action_prob, 
                                     guide_seqs=guide_seqs, # ours
                                     explore_radius=explore_radius,  # ours
                                     temp=temp)
        # import pdb; pdb.set_trace() 
        return seqs[:, 1:]

    def train_step(self, batched_sequence_data, reward):
        start_tokens = torch.ones(batched_sequence_data.size(0), 1).long().to(self.device) * self.model.start_token
        batched_sequence_data = torch.cat([start_tokens, batched_sequence_data], dim=1)
        
        # logits = self.model(batched_sequence_data)[:, 1:, :]  # exclude the start token
        # log_pf = F.log_softmax(logits, dim=-1)
        
        # log_pf= torch.gather(log_pf, 2, batched_sequence_data[:, 1:].long().unsqueeze(2)).squeeze(2)
        # sum_log_pf = log_pf.sum(-1)
        
        # import pdb; pdb.set_trace()
        # loss = (self.Z + sum_log_pf - reward.clamp(min=self.reward_exp_min).log()).pow(2).mean()
        
        # logits = self.model(batched_sequence_data)
        # log_pf = F.log_softmax(logits, dim=-1)
        # log_pf = torch.gather(log_pf, 2, batched_sequence_data.long()[:, :, None]).squeeze(2)
        # sum_log_pf = log_pf.sum(-1)
        
        # loss = (self.Z + sum_log_pf - reward.clamp(min=self.reward_exp_min)).pow(2).mean()
        
        # batched_cond_data = batched_cond_data.float()        
        logits, _ = self.model(batched_sequence_data)
        
        logits = logits[:, :-1]
        log_pf = F.log_softmax(logits, dim=-1)
        targets = batched_sequence_data[:, 1:, None]
        
        # import pdb; pdb.set_trace()
        
        log_pf = torch.gather(log_pf, 2, targets).squeeze(2)
        sum_log_pf = log_pf.sum(-1)
        
        loss = (self.Z + sum_log_pf - self.beta * reward.clamp(min=self.reward_exp_min)).pow(2).mean()
    
        # loss = compute_sequence_cross_entropy(logits, batched_sequence_data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gen_clip)
        self.opt.step()
        self.opt_Z.step()
        self.opt.zero_grad()
        self.opt_Z.zero_grad()
        
        # import pdb; pdb.set_trace()
        
        return loss.item()
    
    def pre_train_step(self, batched_sequence_data):
        start_tokens = torch.ones(batched_sequence_data.size(0), 1).long().to(self.device) * self.model.start_token
        batched_sequence_data = torch.cat([start_tokens, batched_sequence_data], dim=1)
        
        logits, _ = self.model(batched_sequence_data)
    
        loss = compute_sequence_cross_entropy(logits, batched_sequence_data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gen_clip)
        self.opt.step()
        self.opt_Z.step()
        self.opt.zero_grad()
        self.opt_Z.zero_grad()
        
        # import pdb; pdb.set_trace()
        
        return loss.item()
        
        
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


def compute_sequence_cross_entropy(logits, batched_sequence_data):
    logits = logits[:, :-1]
    targets = batched_sequence_data[:, 1:]
    
    # import pdb; pdb.set_trace()

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    return loss

# class GFNLSTMGenerator(nn.Module):
#     def __init__(self, args, partition_init = 50.0):
#         super(GFNLSTMGenerator, self).__init__()
#         self.model = LSTMDecoder(args.lstm_num_layers, args.lstm_hidden_dim, args, num_token = args.vocab_size+1)
#         self._Z = nn.Parameter(torch.ones(64).to(args.device) * partition_init / 64)
        
#         self.model.to(args.device)
#         # self._Z.to(args.device)
        
#         self.reward_exp_min = args.reward_exp_min
#         self.gen_clip = args.gen_clip

#         self.opt = torch.optim.Adam(self.model.parameters(), args.gen_learning_rate, weight_decay=args.gen_L2,
#                             betas=(0.9, 0.999))
#         self.opt_Z = torch.optim.Adam([self._Z], args.gen_Z_learning_rate, weight_decay=args.gen_L2,
#                             betas=(0.9, 0.999))
#         self.device = args.device

#     @property
#     def Z(self):
#         return self._Z.sum()
    
#     def forward(self, batched_sequence_data):
#         return self.model(batched_sequence_data)
    
#     def decode(self, sample_size, max_len, argmax=False, random_action_prob=0.0, guide_seqs=None, explore_radius=1.0, temp=2.0):
#         return self.model.decode(sample_size,
#                                  max_len,
#                                  argmax=argmax,
#                                  random_action_prob=random_action_prob,  # Exploration for GFN-AL
#                                  guide_seqs=guide_seqs,
#                                  explore_radius=explore_radius,  # Trust region
#                                  temp=temp)

#     def train_step(self, batched_sequence_data, reward):
#         start_tokens = torch.ones(batched_sequence_data.size(0), 1).long().to(self.device) * self.model.start
#         batched_sequence_data = torch.cat([start_tokens, batched_sequence_data], dim=1)
        
#         # logits = self.model(batched_sequence_data)[:, 1:, :]  # exclude the start token
#         # log_pf = F.log_softmax(logits, dim=-1)
        
#         # log_pf= torch.gather(log_pf, 2, batched_sequence_data[:, 1:].long().unsqueeze(2)).squeeze(2)
#         # sum_log_pf = log_pf.sum(-1)
        
#         # import pdb; pdb.set_trace()
#         # loss = (self.Z + sum_log_pf - reward.clamp(min=self.reward_exp_min).log()).pow(2).mean()
        
#         # logits = self.model(batched_sequence_data)
#         # log_pf = F.log_softmax(logits, dim=-1)
#         # log_pf = torch.gather(log_pf, 2, batched_sequence_data.long()[:, :, None]).squeeze(2)
#         # sum_log_pf = log_pf.sum(-1)
        
#         # loss = (self.Z + sum_log_pf - reward.clamp(min=self.reward_exp_min)).pow(2).mean()
        
#         # batched_cond_data = batched_cond_data.float()        
#         logits = self.model(batched_sequence_data)
    
#         loss = compute_sequence_cross_entropy(logits, batched_sequence_data)
        
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gen_clip)
#         self.opt.step()
#         self.opt_Z.step()
#         self.opt.zero_grad()
#         self.opt_Z.zero_grad()
        
#         # import pdb; pdb.set_trace()
        
#         return loss.item()
    
#     def get_log_prob(self, batched_sequence_data):
        
#         start_tokens = torch.ones(batched_sequence_data.size(0), 1).long().to(self.device) * self.model.start
#         batched_sequence_data = torch.cat([start_tokens, batched_sequence_data], dim=1)
        
#         logits = self.model(batched_sequence_data)[:, 1:, :]  # exclude the start token
#         log_pf = F.log_softmax(logits, dim=-1)
        
#         log_pf= torch.gather(log_pf, 2, batched_sequence_data[:, 1:].long().unsqueeze(2)).squeeze(2)
#         sum_log_pf = log_pf.sum(-1)
        
#         return sum_log_pf
    
#     def save(self, path):
#         torch.save(self.model.state_dict(), path + '_model')
#         torch.save(self._Z, path + '_Z')
        
#     def load(self, path):
#         self.model.load_state_dict(torch.load(path + '_model'))
#         self._Z = torch.load(path + '_Z')
#         self._Z = nn.Parameter(self._Z)
#         self._Z.to(self.device)
#         self.model.to(self.device)