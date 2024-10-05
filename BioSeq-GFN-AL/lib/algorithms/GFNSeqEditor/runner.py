import wandb
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from lib.utils.env import get_tokenizer
from lib.generator.gfn import TBGFlowNetGenerator
from lib.generator.lstm import GFNLSTMGenerator
# from lib.utils.distance import mean_pairwise_distances, mean_novelty, edit_dist
from lib.utils.log_utils import log_overall_metrics


class GFNSeqEditorLSTMRunner:
    def __init__(self, args, oracle, dataset):
        self.args = args
        self.oracle = oracle  # not proxy
        self.dataset = dataset
        self.tokenizer = get_tokenizer(args)
        self.vocab_size = args.vocab_size
        self.start = args.vocab_size

    def run(self):
        rst = None
        for round in range(self.args.num_rounds):  # 0 ~ 9
            # Step 1: Train the editor with the previous round data (use the same agent with ours)
            editor = GFNLSTMGenerator(self.args, self.tokenizer)
            losses = self._train_editor(editor, batch_size=self.args.gen_data_sample_per_step)
            # Step 2: Propose new sequences
            # Step 2-1: Sample seed sequences (use the same logit with ours)
            seed_samples, seed_vals = self.dataset.weighted_sample(self.args.num_queries_per_round, self.args.rank_coeff)  # high-score samples
            # # Step 2-2: Edit the seed sequences (hyperparameter: delta, )
            samples = self.propose_sequences(editor, seed_samples) #delta=(round+1)/self.args.num_rounds)
            vals = self.oracle(samples).reshape(-1)
            # print("PI:", np.mean(seed_vals), np.mean(vals), np.sum((vals-seed_vals)) / np.sum(vals != seed_vals))
            # Step 3: Update the dataset
            self.dataset.add((samples, vals))
            rst = log_overall_metrics(self.args, self.dataset, round+1, collected=True, new_seqs=["".join([str(i) for i in x]) for x in samples], seed_seqs=["".join([str(i) for i in x]) for x in seed_samples], rst=rst)
                
    def _train_editor(self, editor, batch_size=32):
        """
            implemented based on GFN-AL and Algorithm 1 in the paper
        """
        losses = []
        p_bar = tqdm(range(self.args.gen_num_iterations + 1))
        for it in p_bar:
            if self.args.use_rank_based_off:
                off_x, off_y = self.dataset.weighted_sample(batch_size, self.args.rank_coeff)  # rank-based? weighted_sample(batch_size) 
            else:
                off_x, off_y = self.dataset.sample(batch_size)
                
            seqs = torch.tensor(np.array(off_x)).to(self.args.device)
            rs = torch.tensor(off_y).to(self.args.device)
            loss = editor.train_step(seqs, rs)
            # loss = generator.train_step(seqs, self.l2r(rs))
            p_bar_log = {}
            p_bar_log['off_oracle'] = np.mean(off_y)
            p_bar_log['proxy'] = rs.mean().item()
            p_bar_log['loss'] = loss
            p_bar.set_postfix(p_bar_log)

            losses.append(loss)
            wandb_log = {"generator_total_loss": loss}
            if self.args.use_wandb:
                wandb.log(wandb_log)
        return losses
    
    def propose_sequences(self, editor, seed_seqs, delta=0.1, sigma=0.001, lamb=0.1):
        seed_seqs = torch.from_numpy(np.array(seed_seqs)).to(self.args.device)

        sequences = [torch.full((self.args.num_queries_per_round, 1), self.start, dtype=torch.long).to(self.args.device)]
        hidden = None
        
        for t in range(self.args.max_len):
            with torch.no_grad():
                logit, hidden_state = editor.model(sequences[-1], hidden)
                
                prob_pf = torch.softmax(logit, dim=2).squeeze(1)
                # import pdb; pdb.set_trace()
                ref_actions = seed_seqs[:, t].to(self.args.device).long()
                
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
        
        return sequences.tolist()
        

class GFNSeqEditorRunner:
    def __init__(self, args, oracle, dataset):
        self.args = args
        self.oracle = oracle  # not proxy
        self.dataset = dataset
        self.tokenizer = get_tokenizer(args)
        self.vocab_size = args.vocab_size

    def run(self):
        rst = None
        for round in range(self.args.num_rounds):  # 0 ~ 9
            # Step 1: Train the editor with the previous round data (use the same agent with ours)
            editor = TBGFlowNetGenerator(self.args, self.tokenizer)
            losses = self._train_editor(editor, batch_size=self.args.gen_data_sample_per_step)
            # Step 2: Propose new sequences
            # Step 2-1: Sample seed sequences (use the same logit with ours)
            seed_samples, seed_vals = self.dataset.weighted_sample(self.args.num_queries_per_round, self.args.rank_coeff)  # high-score samples
            # # Step 2-2: Edit the seed sequences (hyperparameter: delta, )
            samples = self.propose_sequences(editor, seed_samples) #delta=(round+1)/self.args.num_rounds)
            vals = self.oracle(samples).reshape(-1)
            # Step 3: Update the dataset
            self.dataset.add((samples, vals))
            rst = log_overall_metrics(self.args, self.dataset, round+1, collected=True, new_seqs=["".join([str(i) for i in x]) for x in samples], seed_seqs=["".join([str(i) for i in x]) for x in seed_samples], rst=rst)
                
    def _train_editor(self, editor, batch_size=32):
        """
            implemented based on GFN-AL and Algorithm 1 in the paper
        """
        rollout_worker = RolloutWorker(self.args, self.tokenizer)
        losses = []
        p_bar = tqdm(range(self.args.gen_num_iterations + 1))
        for it in p_bar:
            rollout_artifacts = rollout_worker.execute_train_episode_batch(n=batch_size, it=it, dataset=self.dataset)
            loss, loss_info = editor.train_step(rollout_artifacts["trajectories"])
            losses.append(loss.item())
            wandb_log = {"generator_total_loss": loss.item()}
            if self.args.use_wandb:
                wandb.log(wandb_log)
        return losses
    
    def propose_sequences(self, model, seed_seqs, delta=0.01, sigma=0.0001, lamb=0.1):
        episodes = len(seed_seqs)
        states = [[] for i in range(episodes)]
        
        for t in range(self.args.max_len):
            x = self.tokenizer.process(states).to(self.args.device)
            with torch.no_grad():
                logits = model(x, None, coef=self.args.gen_output_coef)
            if t == 0 and self.args.task == 'amp':
                logits[:, 0] = -1000 # Prevent model from stopping
                                     # without having output anything
            # prob_pf = F.softmax(logits, dim=-1)  # logits.exp()?
            pf = logits.exp()
            prob_pf = F.softmax(logits, dim=-1)  # pf / pf.sum(-1, keepdim=True)  # F.softmax(logits, dim=-1)
            ref_actions = torch.tensor(seed_seqs)[:, t].to(self.args.device).long()
            
            ref = torch.gather(prob_pf, -1, ref_actions.view(-1, 1))
            noise = torch.normal(0, sigma, size=ref.shape).to(ref.device)
            sub_opt_identifiers = ref < delta * prob_pf.max(-1)[0].view(-1, 1) + noise
            
            try:
                cat = Categorical((1-lamb) * prob_pf + lamb * F.one_hot(ref_actions.long(), self.args.vocab_size).float())
            except:
                import pdb; pdb.set_trace()
            new_actions = cat.sample()
            # import pdb; pdb.set_trace()
            # ref_actions[sub_opt_identifiers] = new_actions[sub_opt_identifiers]
            actions = torch.where(sub_opt_identifiers.view(-1), new_actions.view(-1), ref_actions.view(-1))
            
            for i, a in enumerate(actions):
                states[i] += [a.item()]
                
        return states


class RolloutWorker:
    def __init__(self, args, tokenizer):
        # self.oracle = oracle
        self.max_len = args.max_len
        self.max_len = args.gen_max_len
        self.reward_exp = args.gen_reward_exp
        
        self.exp_ramping_factor = args.gen_reward_exp_ramping
        
        self.tokenizer = tokenizer
        if self.exp_ramping_factor > 0:
            self.l2r = lambda x, t=0: (x) ** (1 + (self.reward_exp - 1) * (1 - 1/(1 + t / self.exp_ramping_factor)))
        else:
            self.l2r = lambda x, t=0: (x) ** self.reward_exp
        self.device = args.device
        self.args = args
        # self.workers = MbStack(oracle)

    def execute_train_episode_batch(self, n, it=0, dataset=None):
        # run an episode
        lists = lambda n: [list() for i in range(n)]
        # n = self.args.gen_data_sample_per_step
        x, y = dataset.sample(n)#sample(n, 0.5)
        # x, y = dataset.weighted_sample(n, 0.01)
        n = len(x)
        traj_states = lists(n)
        traj_actions = lists(n)
        traj_rewards = lists(n)
        traj_dones = lists(n)
        bulk_trajs = list(zip([i for i in x],
                                [self.l2r(torch.tensor(i), it) for i in y]))
        for i in range(len(x)):
            traj_states[i].append([])
            for c, a in zip(x[i], self.tokenizer.process([x[i]]).reshape(-1)):
                traj_states[i].append(traj_states[i][-1] + [c])
                traj_actions[i].append(a)
                traj_rewards[i].append(0 if len(traj_actions[i]) != self.max_len else self.l2r(torch.tensor(y[i]), it))
                traj_dones[i].append(float(len(traj_rewards[i]) == self.max_len))
        return {
            # "visited": visited,
            "trajectories": {
                "traj_states": traj_states,
                "traj_actions": traj_actions,
                "traj_rewards": traj_rewards,
                "traj_dones": traj_dones,
                # "states": states,
                "bulk_trajs": bulk_trajs
            }
        }
