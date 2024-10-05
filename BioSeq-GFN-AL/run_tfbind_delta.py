import argparse
import gzip
import pickle
import itertools
import time
import wandb

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm

from lib.acquisition_fn import get_acq_fn
from lib.dataset import get_dataset
from lib.generator import get_generator
from lib.logging import get_logger
from lib.oracle_wrapper import get_oracle
from lib.proxy import get_proxy_model
from lib.utils.distance import is_similar, edit_dist
from lib.utils.env import get_tokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--save_path", default='results/test_mlp.pkl.gz')
parser.add_argument("--tb_log_dir", default='results/test_mlp')
parser.add_argument("--name", default='test_mlp')
parser.add_argument("--load_scores_path", default='.')

# Multi-round
parser.add_argument("--num_rounds", default=10, type=int)
parser.add_argument("--task", default="tfbind", type=str)
parser.add_argument("--num_queries_per_round", default=128, type=int)
parser.add_argument("--vocab_size", default=4, type=int)
parser.add_argument("--max_len", default=8, type=int)
parser.add_argument("--gen_max_len", default=8, type=int)
parser.add_argument("--proxy_uncertainty", default="dropout")
parser.add_argument("--save_scores_path", default=".")
parser.add_argument("--save_scores", action="store_true")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--run", default=-1, type=int)
parser.add_argument("--noise_params", action="store_true")
parser.add_argument("--enable_tensorboard", action="store_true")
parser.add_argument("--use_wandb", action="store_true")
parser.add_argument("--save_proxy_weights", action="store_true")
parser.add_argument("--use_uncertainty", action="store_true")
parser.add_argument("--filter", action="store_true")
parser.add_argument("--kappa", default=0.1, type=float)
parser.add_argument("--acq_fn", default="none", type=str)
parser.add_argument("--load_proxy_weights", type=str)
parser.add_argument("--max_percentile", default=80, type=int)
parser.add_argument("--filter_threshold", default=0.1, type=float)
parser.add_argument("--filter_distance_type", default="edit", type=str)
parser.add_argument("--oracle_split", default="D2_target", type=str)
parser.add_argument("--proxy_data_split", default="D1", type=str)
parser.add_argument("--oracle_type", default="MLP", type=str)
parser.add_argument("--oracle_features", default="AlBert", type=str)
parser.add_argument("--medoid_oracle_dist", default="edit", type=str)
parser.add_argument("--medoid_oracle_norm", default=1, type=int)
parser.add_argument("--medoid_oracle_exp_constant", default=6, type=int)

# Generator
parser.add_argument("--gen_learning_rate", default=1e-5, type=float)
parser.add_argument("--gen_Z_learning_rate", default=1e-3, type=float)
parser.add_argument("--gen_clip", default=10, type=float)
parser.add_argument("--gen_num_iterations", default=5000, type=int)
parser.add_argument("--gen_episodes_per_step", default=16, type=int)
parser.add_argument("--gen_num_hidden", default=2048, type=int)
parser.add_argument("--gen_reward_norm", default=1, type=float)
parser.add_argument("--gen_reward_exp", default=3, type=float)
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

parser.add_argument("--radius_option", default='none', type=str)
parser.add_argument("--min_radius", default=0.5, type=float)
parser.add_argument("--max_radius", default=0.5, type=float)
parser.add_argument("--K", default=25, type=int)

parser.add_argument("--rank_based_proxy_training", action="store_true")
parser.add_argument("--sigma_coeff", default=10, type=float)
parser.add_argument("--rank_coeff", default=0.01, type=float)

# Proxy
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
parser.add_argument("--proxy_num_iterations", default=3000, type=int)
parser.add_argument("--proxy_num_dropout_samples", default=25, type=int)
parser.add_argument("--proxy_pos_ratio", default=0.9, type=float)


class MbStack:
    def __init__(self, f):
        self.stack = []
        self.f = f

    def push(self, x, i):
        self.stack.append((x, i))

    def pop_all(self):
        if not len(self.stack):
            return []
        with torch.no_grad():
            ys = self.f([i[0] for i in self.stack])
        idxs = [i[1] for i in self.stack]
        self.stack = []
        return zip(ys, idxs)


def filter_len(x, y, max_len):
    res = ([], [])
    for i in range(len(x)):
        if len(x[i]) < max_len:
            res[0].append(x[i])
            res[1].append(y[i])
    return res

def get_current_radius(iter, round, args, rs=None, y=None, sigma=None):
    if args.radius_option == 'round_linear':
        return (args.max_radius-args.min_radius) * ((round+1)/args.num_rounds) + args.min_radius
    elif args.radius_option == 'proxy_var':
        linear_r = (args.max_radius-args.min_radius) * ((round+1)/args.num_rounds) + args.min_radius * torch.ones(rs.size(0)).to(rs.device)  #(round+1)/args.num_rounds * torch.ones(err.size(0)).to(err.device)
        return (linear_r - args.sigma_coeff * sigma.view(-1)).clamp(0.1, 1)
    elif args.radius_option == 'fixed':
        return args.max_radius * torch.ones(rs.size(0)).to(rs.device)
    else:
        return 1.
    
class RolloutWorker:
    def __init__(self, args, oracle, tokenizer):
        self.oracle = oracle
        self.max_len = args.max_len
        self.max_len = args.gen_max_len
        self.episodes_per_step = args.gen_episodes_per_step
        self.random_action_prob = args.gen_random_action_prob
        self.reward_exp = args.gen_reward_exp
        self.sampling_temperature = args.gen_sampling_temperature
        self.out_coef = args.gen_output_coef

        self.balanced_loss = args.gen_balanced_loss == 1
        self.reward_norm = args.gen_reward_norm
        self.reward_min = torch.tensor(float(args.gen_reward_min))
        self.loss_eps = torch.tensor(float(args.gen_loss_eps)).to(args.device)
        self.leaf_coef = args.gen_leaf_coef
        self.exp_ramping_factor = args.gen_reward_exp_ramping
        
        self.tokenizer = tokenizer
        if self.exp_ramping_factor > 0:
            self.l2r = lambda x, t=0: (x) ** (1 + (self.reward_exp - 1) * (1 - 1/(1 + t / self.exp_ramping_factor)))
        else:
            self.l2r = lambda x, t=0: (x) ** self.reward_exp
        self.device = args.device
        self.args = args
        self.workers = MbStack(oracle)

    def rollout(self, model, episodes, use_rand_policy=True):
        visited = []
        lists = lambda n: [list() for i in range(n)]
        states = [[] for i in range(episodes)]
        traj_states = torch.tensor([[[]] for i in range(episodes)]).to(model.device)
        traj_actions = torch.tensor(lists(episodes)).to(model.device)
        traj_rewards = torch.tensor(lists(episodes)).to(model.device)
        traj_dones = torch.tensor(lists(episodes)).to(model.device)

        traj_logprob = np.zeros(episodes)

        for t in (range(self.max_len) if episodes > 0 else []):
            x = self.tokenizer.process(states).to(self.device)
            with torch.no_grad():
                logits = model(x, None, coef=self.out_coef)
            if t == 0:
                logits[:, 0] = -1000 # Prevent model from stopping
                                     # without having output anything
            try:
                cat = Categorical(logits=logits / self.sampling_temperature)
            except:
                print(states)
                print(x)
                print(logits)
                print(list(model.model.parameters()))
            actions = cat.sample()
            if use_rand_policy and self.random_action_prob > 0:
                for i in range(actions.shape[0]):
                    if np.random.uniform(0,1) < self.random_action_prob:
                        actions[i] = torch.tensor(np.random.randint(t == 0, logits.shape[1])).to(self.device)
            
            # Append predicted characters for active trajectories
            states_tensor = torch.tensor(states).to(actions.device)
            # Update states with actions
            next_states = torch.cat([states_tensor, actions.view(-1, 1)], dim=1)

            # Update trajectory lists
            if t == self.max_len - 1:
                for ns, i in zip (next_states, torch.arange(next_states.size(0))):
                    self.workers.push(ns.tolist(), i.item())

                rewards = torch.zeros(actions.size(0), 1).to(actions.device)
                dones = torch.ones(actions.size(0), 1).to(actions.device)
            else:
                rewards = torch.zeros(actions.size(0), 1).to(actions.device)
                dones = torch.zeros(actions.size(0), 1).to(actions.device)

            # Append to trajectory tensors (assuming traj_* are lists of tensors or 3D tensors)
            traj_states = torch.cat([traj_states, next_states.unsqueeze(1)], dim=-1)
            traj_actions = torch.cat([traj_actions, actions.unsqueeze(1)], dim=-1)
            traj_rewards = torch.cat([traj_rewards, rewards], dim=-1)
            traj_dones = torch.cat([traj_dones, dones], dim=-1)
            states = next_states.tolist()

        return visited, states, traj_states.tolist(), traj_actions.tolist(), traj_rewards.tolist(), traj_dones.tolist()

    def execute_train_episode_batch(self, model, it=0, dataset=None, use_rand_policy=True, return_all_visited=False):
        # run an episode
        lists = lambda n: [list() for i in range(n)]
        visited, states, traj_states, \
            traj_actions, traj_rewards, traj_dones = self.rollout(model, self.episodes_per_step, use_rand_policy=use_rand_policy) 
        lens = np.mean([len(i) for i in traj_rewards])
        bulk_trajs = []
        rq = []
        for (r, mbidx) in self.workers.pop_all():
            traj_rewards[mbidx][-1] = self.l2r(r, it)
            rq.append(r.item())
            s = states[mbidx]
            visited.append((s, traj_rewards[mbidx][-1].item(), r.item()))
            bulk_trajs.append((s, traj_rewards[mbidx][-1].item()))
        if args.gen_data_sample_per_step > 0 and dataset is not None:
            n = args.gen_data_sample_per_step
            m = len(traj_states)
            x, y = dataset.sample(n)
            # x, y = dataset.weighted_sample(n, 0.01)
            n = len(x)
            traj_states += lists(n)
            traj_actions += lists(n)
            traj_rewards += lists(n)
            traj_dones += lists(n)
            bulk_trajs += list(zip([i for i in x],
                                   [self.l2r(torch.tensor(i), it) for i in y]))
            if return_all_visited:
                with torch.no_grad():
                    rs = self.oracle(x).view(-1).tolist()
                visited += list(zip([i.tolist() for i in x],
                                    [self.l2r(i, it) for i in y],
                                    [self.l2r(i, it) for i in rs],
                                    y, rs
                                    ))
            for i in range(len(x)):
                traj_states[i+m].append([])
                for c, a in zip(x[i], self.tokenizer.process([x[i]]).reshape(-1)):
                    traj_states[i+m].append(traj_states[i+m][-1] + [c])
                    traj_actions[i+m].append(a)
                    traj_rewards[i+m].append(0 if len(traj_actions[i+m]) != self.max_len else self.l2r(torch.tensor(y[i]), it))
                    traj_dones[i+m].append(float(len(traj_rewards[i+m]) == self.max_len))
        return {
            "visited": visited,
            "trajectories": {
                "traj_states": traj_states,
                "traj_actions": traj_actions,
                "traj_rewards": traj_rewards,
                "traj_dones": traj_dones,
                "states": states,
                "bulk_trajs": bulk_trajs
            }
        }
        
    def guided_rollout(self, model, guide_seqs, sample_action_prob):
        visited = []
        lists = lambda n: [list() for i in range(n)]
        episodes = len(guide_seqs)
        states = [[] for i in range(episodes)]
        traj_states = [[[]] for i in range(episodes)]
        traj_actions = lists(episodes)
        traj_rewards = lists(episodes)
        traj_dones = lists(episodes)

        traj_logprob = np.zeros(episodes)
        masked_cnt = torch.zeros(episodes).to(self.device)
        for t in (range(self.max_len) if episodes > 0 else []):
            x = self.tokenizer.process(states).to(self.device)
            with torch.no_grad():
                logits = model(x, None, coef=self.out_coef)
            if t == 0 and args.task == 'amp':
                logits[:, 0] = -1000 # Prevent model from stopping
                                     # without having output anything
            try:
                cat = Categorical(logits=logits / self.sampling_temperature)
            except:
                print(states)
                print(x)
                print(logits)
                print(list(model.model.parameters()))
            # import pdb; pdb.set_trace()
            actions = cat.sample()
            
            guide_actions = torch.tensor(np.array(guide_seqs))[:, t].to(self.device)
            # unmasked = (1 - masks[:, t]).bool()
            # actions[unmasked] = guide_actions.long()[unmasked]
            mask = torch.rand(actions.size(0)).to(self.device) > sample_action_prob.view(-1)
            masked_cnt += mask.int()
            actions[mask] = guide_actions.long()[mask]
            
            # Append predicted characters for active trajectories
            for i, a in enumerate(actions):
                if t == self.max_len - 1:
                    self.workers.push(states[i] + [a.item()], i)
                    r = 0
                    d = 1
                else:
                    r = 0
                    d = 0
                traj_states[i].append(states[i] + [a.item()])
                traj_actions[i].append(a)
                traj_rewards[i].append(r)
                traj_dones[i].append(d)
                states[i] += [a.item()]
        
        # if K.min() == 0: 
        #     import pdb; pdb.set_trace()
        return visited, states, traj_states, traj_actions, traj_rewards, traj_dones, self.max_len - masked_cnt
        
    def execute_train_episode_batch_with_delta(self, model, it=0, dataset=None, return_all_visited=False, round=0, use_offline_data=True, guide_seqs=None):
        # run an episode
        lists = lambda n: [list() for i in range(n)]
        n = self.episodes_per_step  # args.gen_data_sample_per_step
        x, y = dataset.weighted_sample(n, args.rank_coeff)
        if args.acq_fn.lower() == "ucb":
            with torch.no_grad():
                rs, mu, sigma = self.oracle(x, return_all=True)
        else:
            with torch.no_grad():
                rs = self.oracle(x)
                sigma = torch.zeros(n).to(self.device)
        radius = get_current_radius(iter=it, round=round, args=args, rs=rs, y=y, sigma=sigma)

        if type(radius) == float:
            radius = torch.tensor([radius] * n).to(self.device)
        visited, states, traj_states, \
            traj_actions, traj_rewards, traj_dones, noised_cnt = self.guided_rollout(model, x, sample_action_prob=radius)
        n = len(x)
        bulk_trajs = []
        rq = []
        for (r, mbidx) in self.workers.pop_all():
            traj_rewards[mbidx][-1] = self.l2r(r, it)
            rq.append(r.item())
            s = states[mbidx]
            visited.append((s, traj_rewards[mbidx][-1].item(), r.item()))
            bulk_trajs.append((s, traj_rewards[mbidx][-1].item()))
            
        if args.gen_data_sample_per_step > 0 and use_offline_data: # and guide_seqs is None:
            n = args.gen_data_sample_per_step
            m = len(traj_states)
            x, y = dataset.sample(n)#sample(n, 0.5)
            # x, y = dataset.weighted_sample(n, args.rank_coeff)
            n = len(x)
            traj_states += lists(n)
            traj_actions += lists(n)
            traj_rewards += lists(n)
            traj_dones += lists(n)
            bulk_trajs += list(zip([i for i in x],
                                   [self.l2r(torch.tensor(i), it) for i in y]))
            if return_all_visited:
                with torch.no_grad():
                    rs = self.oracle(x).view(-1).tolist()
                visited += list(zip([i.tolist() for i in x],
                                    [self.l2r(i, it) for i in y],
                                    [self.l2r(i, it) for i in rs],
                                    y, rs
                                    ))
            for i in range(len(x)):
                traj_states[i+m].append([])
                for c, a in zip(x[i], self.tokenizer.process([x[i]]).reshape(-1)):
                    traj_states[i+m].append(traj_states[i+m][-1] + [c])
                    traj_actions[i+m].append(a)
                    traj_rewards[i+m].append(0 if len(traj_actions[i+m]) != self.max_len else self.l2r(torch.tensor(y[i]), it))
                    traj_dones[i+m].append(float(len(traj_rewards[i+m]) == self.max_len))
        return {
            "visited": visited,
            "trajectories": {
                "traj_states": traj_states,
                "traj_actions": traj_actions,
                "traj_rewards": traj_rewards,
                "traj_dones": traj_dones,
                "states": states,
                "bulk_trajs": bulk_trajs
            },
            "sigma": sigma.mean().item() if args.acq_fn.lower() == "ucb" else 0,
            "radius": radius.mean().item(), #if args.acq_fn.lower() == "ucb" else 0
            "radius_max": radius.max().item(),
            "radius_min": radius.min().item(),
            "noised_cnt": noised_cnt.mean().item(),
            "noised_cnt_max": noised_cnt.max().item(),
        }
        

def train_generator(args, generator, oracle, tokenizer, dataset, sampling_temp=0., round=0):
    # oracle = proxy
    print("Training generator")
    visited = []
    rollout_worker = RolloutWorker(args, oracle, tokenizer)
    if sampling_temp > 0:
        rollout_worker.sampling_temperature = sampling_temp
    losses = []
    p_bar = tqdm(range(args.gen_num_iterations + 1))
    for it in p_bar:
        if args.radius_option == 'none':
            rollout_artifacts = rollout_worker.execute_train_episode_batch(generator, it, dataset, return_all_visited=True)
        else:
            rollout_artifacts = rollout_worker.execute_train_episode_batch_with_delta(generator, it, dataset, return_all_visited=True, round=round)
            p_bar.set_postfix({"sigma": rollout_artifacts["sigma"], "radius": rollout_artifacts["radius"]})
        visited.extend(rollout_artifacts["visited"])
        # memory.extend(rollout_artifacts["visited"])
        
        # import pdb; pdb.set_trace()
        loss, loss_info = generator.train_step(rollout_artifacts["trajectories"])
        losses.append(loss.item())

        # loss, loss_info = generator.train_with_rb(visited, inner_loop= (it // 10) + 1)  # NoAF
        args.logger.add_scalar("generator_total_loss", loss.item())
        wandb_log = {"generator_total_loss": loss.item()}
        if args.radius_option != 'none':
            wandb_log["sigma"] = rollout_artifacts["sigma"]
            wandb_log["radius"] = rollout_artifacts["radius"]
            wandb_log["radius_max"] = rollout_artifacts["radius_max"]
            wandb_log["radius_min"] = rollout_artifacts["radius_min"]
            wandb_log["noised_cnt"] = rollout_artifacts["noised_cnt"]
            wandb_log["noised_cnt_max"] = rollout_artifacts["noised_cnt_max"]
        for key, val in loss_info.items():
            args.logger.add_scalar(f"generator_{key}", val.item())
            wandb_log[f"generator_{key}"] = val.item()
        if it % 100 == 0:
            rs = torch.tensor([i[-1] for i in rollout_artifacts["trajectories"]["traj_rewards"]]).mean()
            args.logger.add_scalar("gen_reward", rs.item())
            wandb_log["gen_reward"] = rs.item()
        if it % 5000 == 0:
            args.logger.save(args.save_path, args)
        if args.use_wandb:
            wandb.log(wandb_log)
    rollout_worker.sampling_temperature = args.gen_sampling_temperature
    return rollout_worker, losses


def filter_samples(args, samples, reference_set):
    filtered_samples = []
    for sample in samples:
        similar = False
        for example in reference_set:
            if is_similar(sample, example, args.filter_distance_type, args.filter_threshold):
                similar = True
                break
        if not similar:
            filtered_samples.append(sample)
    return filtered_samples


def sample_batch(args, rollout_worker, generator, oracle, round=0, dataset=None):
    print("Generating samples")
    samples = ([], [])
    scores = []
    i = 0
    num_iter = args.K  #100 if args.max_len < 10 else 5
    while len(samples[0]) < args.num_queries_per_round * num_iter:
        # rollout_artifacts = rollout_worker.execute_train_episode_batch(generators[i%args.num_generators], it=0, use_rand_policy=False)
        if args.radius_option == 'none':
            rollout_artifacts = rollout_worker.execute_train_episode_batch(generator, it=0, use_rand_policy=False)
        else:
            rollout_artifacts = rollout_worker.execute_train_episode_batch_with_delta(generator, it=args.gen_num_iterations + 1, dataset=dataset, round=round, use_offline_data=False)
        states = rollout_artifacts["trajectories"]["states"]
        vals = oracle(states).reshape(-1)
        samples[0].extend(states)
        samples[1].extend(vals)
        scores.extend(torch.tensor(rollout_artifacts["trajectories"]["traj_rewards"])[:, -1].numpy().tolist())
        i += 1
    idx_pick = np.argsort(scores)[::-1][:args.num_queries_per_round]
    return (np.array(samples[0])[idx_pick].tolist(), np.array(samples[1])[idx_pick]), np.array(scores)[idx_pick]


def construct_proxy(args, tokenizer, dataset=None):
    proxy = get_proxy_model(args, tokenizer)
    sigmoid = nn.Sigmoid()
    if args.proxy_type == "classification":
        l2r = lambda x: sigmoid(x.clamp(min=args.gen_reward_min)) / args.gen_reward_norm
    elif args.proxy_type == "regression":
        l2r = lambda x: x.clamp(min=args.gen_reward_min) / args.gen_reward_norm
    args.reward_exp_min = max(l2r(torch.tensor(args.gen_reward_min)), 1e-32)
    acq_fn = get_acq_fn(args)
    return acq_fn(args, proxy, l2r, dataset)


def mean_pairwise_distances(args, seqs):
    dists = []
    for pair in itertools.combinations(seqs, 2):
        dists.append(edit_dist(*pair))
    return np.mean(dists)


def mean_novelty(seqs, ref_seqs):
    novelty = [min([edit_dist(seq, ref) for ref in ref_seqs]) for seq in seqs]
    return np.mean(novelty)

def pearson_correlation(X, Y):
    n = len(X)
    mean_X = sum(X) / n
    mean_Y = sum(Y) / n
    covariance = sum((X_i - mean_X) * (Y_i - mean_Y) for X_i, Y_i in zip(X, Y))
    std_X = np.sqrt(sum((X_i - mean_X)**2 for X_i in X))
    std_Y = np.sqrt(sum((Y_i - mean_Y)**2 for Y_i in Y))
    return covariance / (std_X * std_Y)


def spearman_correlation(X, Y):
    rank_X = np.argsort(np.argsort(-1 * X))  # Involves sorting - O(n log n)
    rank_Y = np.argsort(np.argsort(-1 * Y))  # Involves sorting - O(n log n)
    return pearson_correlation(rank_X, rank_Y)


def compute_correlations(seqs, scores, proxy, generator):
    print("Computing correlations")
    proxy_scores, log_ps = [], []
    batch_size = 256
    
    if len(seqs) == 0:
        return 0, 0, 0, 0
    for i in range(len(seqs) // batch_size + 1):
        start, end = batch_size*i, min(batch_size*(i+1), len(seqs))
        if start == end:
            continue
        with torch.no_grad():
            try:
                y = proxy(seqs[start:end])
                logp = generator.get_logp(seqs[start:end]).view(-1)
            except:
                import pdb; pdb.set_trace()
        proxy_scores.extend(y.view(-1).tolist())
        log_ps.extend(logp.tolist())
    proxy_scores = np.array(proxy_scores)
    log_ps = np.array(log_ps)
    
    if type(scores) == list:
        scores = np.array(scores)
    
    oracle_proxy = spearman_correlation(scores, proxy_scores)
    logp_proxy = spearman_correlation(proxy_scores, log_ps)
    logp_oracle = spearman_correlation(scores, log_ps)
    oracle_proxy_mse = np.mean((scores - proxy_scores)**2)
    
    return oracle_proxy, logp_proxy, logp_oracle, oracle_proxy_mse

def get_all_support(args, dataset, proxy, t=0):
    print("Gathering all support")
    x_all, y_all = dataset.x_all, dataset.y_all  # np.array
    
    if args.radius_option == 'none':
        x_holdout, y_holdout = dataset.get_holdout_data(return_as_str=False)
        return (x_all, y_all), (x_holdout, y_holdout), None, None
    
    x_queried, y_queried = dataset.get_all_data(return_as_str=False)
    x_holdout, _ = dataset.get_holdout_data(return_as_str=True)
    
    rs = []
    ys = []
    batch_size = 256
    for i in range(len(x_queried) // batch_size + 1):
        start, end = batch_size*i, min(batch_size*(i+1), len(x_queried))
        if start == end:
            continue
        with torch.no_grad():
            y, _, sigma = proxy(x_queried[start:end], return_all=True)
        radius = get_current_radius(iter=5001, round=t, args=args, rs=y, y=y, sigma=sigma)
        rs.extend(radius.tolist())
        ys.extend(y.tolist())
    rs = np.array(rs)
    ys = np.array(ys)
    
    seqs = np.stack(x_all)
    ref_seqs = np.stack(x_queried)
    dists = np.stack([(ref_seqs != seq).sum(axis=1) for seq in seqs])  # (all, queried)

    neighbor_seqs, holdout_neighbor_seqs = [], []
    neighbor_scores, holdout_neighbor_scores = [], []
    # import pdb; pdb.set_trace()
    for i, seq in enumerate(x_all):
        if (dists[i] / seqs.shape[1] <= rs).any():
            neighbor_seqs.append(seq)
            neighbor_scores.append(y_all[i])
            if ''.join([str(i) for i in seq]) in x_holdout:
                holdout_neighbor_seqs.append(seq)
                holdout_neighbor_scores.append(y_all[i])

    x_holdout, y_holdout = dataset.get_holdout_data(return_as_str=False)
    
    return (x_all, y_all), (x_holdout, y_holdout), (neighbor_seqs, neighbor_scores), (holdout_neighbor_seqs, holdout_neighbor_scores)

def log_overall_metrics(args, dataset, round, new_batch, collected=False, rst=None):
    top100 = dataset.top_k(128)
    args.logger.add_scalar("top-128-scores", np.mean(top100[1]), use_context=False)
    dist100 = mean_pairwise_distances(args, top100[0])
    args.logger.add_scalar("top-128-dists", dist100, use_context=False)
    args.logger.add_object("top-128-seqs", top100[0])
    queried_dist = mean_pairwise_distances(args, new_batch[0])
    queried_corrlation = spearman_correlation(new_batch[1], new_batch[2])
    
    if args.task not in ['gfp', 'aav']:
        ref_seqs, _ = dataset.get_ref_data()
        novelty100 = mean_novelty(top100[0], ref_seqs)
    else:
        novelty100 = 0
    print("========== Round {} ==========".format(round))
    print("Scores, 128", np.mean(top100[1]))
    print("Dist, 128", dist100)
    print("Dist and correlation, queried", queried_dist, queried_corrlation)
    print("Novelty, 128", novelty100)
    log = {'top-128': np.mean(top100[1]),
           'dist-128': dist100,
           'nov-128': novelty100,
           'max': np.max(top100[1]),
           'median': np.percentile(top100[1], 50),
           'queried_score': np.mean(new_batch[1]),
           'queried_dist': queried_dist,
           'queried_correlation': queried_corrlation,
           'round': round}
    if rst is None:
        rst = pd.DataFrame({'round': round, 'sequence': top100[0], 'true_score': top100[1]})
    else:
        rst = rst.append(pd.DataFrame({'round': round, 'sequence': top100[0], 'true_score': top100[1]}))
    if collected:
        top100 = dataset.top_k_collected(128)
        args.logger.add_scalar("top-128-collected-scores", np.mean(top100[1]), use_context=False)
        args.logger.add_scalar("max-128-collected-scores", np.max(top100[1]), use_context=False)
        dist100 = mean_pairwise_distances(args, top100[0])
        if args.task not in ['gfp', 'aav']:
            novelty100 = mean_novelty(top100[0], ref_seqs)
        else:
            novelty100 = 0
        args.logger.add_scalar("top-128-collected-dists", dist100, use_context=False)
        args.logger.add_object("top-128-collected-seqs", top100[0])
        print("Collected Scores, 128, max, 50 pl", np.mean(top100[1]), np.max(top100[1]), np.percentile(top100[1], 50))
        print("Collected Dist, 128", dist100)
        log["collected_top-128"] = np.mean(top100[1])
        log["collected_max"] = np.max(top100[1])
        log["collected_50pl"] = np.percentile(top100[1], 50)
        log["collected_dist-128"] = dist100
        log["collected_novelty-128"] = novelty100
    
    if args.use_wandb:
        wandb.log(log)

    return rst

def train(args, oracle, dataset):  # runner.run()
    tokenizer = get_tokenizer(args)
    args.logger.set_context("iter_0")
    proxy = construct_proxy(args, tokenizer, dataset=dataset)
    # import pdb; pdb.set_trace()
    proxy.update(dataset)
    rst = None
    for round in range(args.num_rounds):
        args.logger.set_context(f"iter_{round+1}")
        generator = get_generator(args, tokenizer)
        rollout_worker, losses = train_generator(args, generator, proxy, tokenizer, dataset, round=round)
        batch, proxy_score = sample_batch(args, rollout_worker, generator, oracle, round=round, dataset=dataset)
        args.logger.add_object("collected_seqs", batch[0])
        args.logger.add_object("collected_seqs_scores", batch[1])
        dataset.add(batch)
        new_seqs = ["".join([str(i) for i in x]) for x in batch[0]]
        rst = log_overall_metrics(args, dataset, round+1, new_batch=(new_seqs, batch[1], proxy_score), collected=True, rst=rst)
        if round != args.num_rounds - 1:
            proxy.update(dataset)
        args.logger.save(args.save_path, args)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.logger = get_logger(args)
    args.device = torch.device('cuda')
    oracle = get_oracle(args)
    dataset = get_dataset(args, oracle)
    
    if args.use_wandb:
        proj = 'gfn_al'
        run = wandb.init(project=proj, group=args.task, config=args, reinit=True)
        wandb.run.name = args.name + "_" + str(args.seed) + "_" + wandb.run.id
    train(args, oracle, dataset)
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.radius_option in ['round_linear', 'proxy_var', 'fixed', 'none']
    main(args)
