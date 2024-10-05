import wandb
import numpy as np
from tqdm import tqdm

import torch

from lib.utils.env import get_tokenizer
from lib.generator.lstm import GFNLSTMGenerator
from lib.utils.log_utils import log_overall_metrics
from lib.utils.distance import is_similar
from lib.proxy import get_proxy_model
from lib.acquisition_fn import get_acq_fn


def get_current_radius(iter, round, args, rs=None, y=None, sigma=None):
    if args.radius_option == 'round_linear':
        # return (round+1)/args.num_rounds
        r = (args.max_radius-args.min_radius) * ((round+1)/args.num_rounds) + args.min_radius
        return r * torch.ones(sigma.size(0)).to(sigma.device)
    elif args.radius_option == 'iter_round_linear':
        r = (iter+1)/args.gen_num_iterations * torch.ones(sigma.size(0)).to(sigma.device)
        upper = (args.max_radius-args.min_radius) * ((round+1)/args.num_rounds) + args.min_radius
        return r.clamp(0.001, upper)
        # r = (iter+1)/args.gen_num_iterations
        # r = max(0.1, min(0.5+(round+1)/(2*args.num_rounds), r))
        # return r * torch.ones(sigma.size(0)).to(sigma.device)  # max(0.1, min(1.0, r))
    elif args.radius_option == 'proxy_var':
        linear_r = (args.max_radius-args.min_radius) * ((round+1)/args.num_rounds) + args.min_radius * torch.ones(rs.size(0)).to(rs.device)  #(round+1)/args.num_rounds * torch.ones(err.size(0)).to(err.device)
        return (linear_r - args.sigma_coeff * sigma.view(-1)).clamp(0.0, 1)
    elif args.radius_option == 'proxy_var_iter':
        r = (iter+1)/args.gen_num_iterations * torch.ones(sigma.size(0)).to(sigma.device)
        upper = (args.max_radius-args.min_radius) * ((round+1)/args.num_rounds) + args.min_radius
        r = r.clamp(0.01, upper) - args.sigma_coeff * sigma.view(-1)
        return r.clamp(0.001, 1)
        # r = max(0.1, min(0.5+(round+1)/(2*args.num_rounds), r))
        # linear_r = r * torch.ones(rs.size(0)).to(rs.device) 
        # return (linear_r - args.sigma_coeff * sigma.view(-1)).clamp(0.1, 1) #(r - 5 * err).clamp(0.1, 1)
    elif args.radius_option == 'fixed':
        return args.min_radius * torch.ones(sigma.size(0)).to(sigma.device)
    else:
        return torch.ones(sigma.size(0)).to(sigma.device)


class GFNLSTMRunner:
    def __init__(self, args, oracle, dataset):
        self.args = args
        self.oracle = oracle  # not proxy
        self.dataset = dataset
        self.tokenizer = get_tokenizer(args)
        self.vocab_size = args.vocab_size
        self.start = args.vocab_size
        
        if args.gen_reward_exp_ramping > 0:
            self.l2r = lambda x, t=0: (x) ** (1 + (args.gen_reward_exp - 1) * (1 - 1/(1 + t / args.gen_reward_exp_ramping)))
        else:
            self.l2r = lambda x, t=0: (x) ** args.gen_reward_exp

    def construct_proxy(self):
        proxy = get_proxy_model(self.args, self.tokenizer)
        sigmoid = torch.nn.Sigmoid()
        if self.args.proxy_type == "classification":
            l2r = lambda x: sigmoid(x.clamp(min=self.args.gen_reward_min)) / self.args.gen_reward_norm
        elif self.args.proxy_type == "regression":
            l2r = lambda x: x.clamp(min=self.args.gen_reward_min) / self.args.gen_reward_norm
        self.args.reward_exp_min = max(l2r(torch.tensor(self.args.gen_reward_min)), 1e-32)
        acq_fn = get_acq_fn(self.args)
        return acq_fn(self.args, proxy, l2r, self.dataset)
    
    def run(self):
        proxy = self.construct_proxy()
        # if self.args.use_pretrain:
        #     generator = GFNLSTMGenerator(self.args)
        #     if self.args.checkpoint is None:
        #         generator.pretrain(self.dataset, self.args.gen_pretrain_steps)
        #     else:
        #         generator.load_state_dict(torch.load(self.args.checkpoint))
        rst = None
        for t in range(self.args.num_rounds):  # 0 ~ 9
            proxy.update(self.dataset)
            # Step 1: Train the editor with the previous round data (use the same agent with ours)
            # editor = TBGFlowNetGenerator(self.args, self.tokenizer)
            generator = GFNLSTMGenerator(self.args)
            losses = self._train_generator(generator, proxy, t)
            # samples = editor.decode(self.args.num_queries_per_round, argmax=False, max_len=self.args.gen_max_len)
            # sample_size, max_len, argmax=argmax, guide_seqs=guide_seqs, temp=temp
            # Step 2: Propose new sequences
            samples = self.propose_sequences(generator, proxy, t)
            # print("PI:", np.mean(seed_vals), np.mean(vals), np.sum((vals-seed_vals)) / np.sum(vals != seed_vals))
            # Step 3: Update the dataset
            vals = self.oracle(samples).reshape(-1)
            self.dataset.add((samples, vals))
            wt = self.oracle.wt if self.args.task in ['gfp', 'aav'] else None
            # import pdb; pdb.set_trace()
            rst = log_overall_metrics(self.args, self.dataset, t+1, collected=True, query=(samples, vals), wt=wt, rst=rst) #, new_seqs=["".join([str(i) for i in x]) for x in samples], ref_seqs=["".join([str(i) for i in x]) for x in seed_samples])
            
    def _train_generator(self, generator, proxy, t=0):
        losses = []
        batch_size = self.args.gen_episodes_per_step
        p_bar = tqdm(range(self.args.gen_num_iterations + 1))
        loss = 0
        for it in p_bar:
            # rollout_artifacts = rollout_worker.execute_train_episode_batch(n=batch_size, it=it, dataset=self.dataset)
            batch_size = self.args.gen_episodes_per_step if it > self.args.warmup_iter else 0
            p_bar_log = {}
            if batch_size > 0:
                if self.args.radius_option == "none":  # default GFN-AL
                    seqs = generator.decode(batch_size, max_len=self.args.max_len, random_action_prob=self.args.gen_random_action_prob , temp=self.args.gen_sampling_temperature)
                    vals = self.oracle(seqs.tolist()).reshape(-1)
                    rs = proxy(seqs.cpu())
                    try:
                        p_bar_log = {'gen_oracle': vals.mean(), 'gen_proxy': rs.mean().item()}
                        # p_bar.write(f"Oracle: {vals.mean()}, Proxy: {rs.mean().item()}")
                        # p_bar.set_postfix({'oracle': vals.mean(), 'proxy': rs.mean().item(), 'loss': loss})
                    except:
                        pass
                else:
                    # radius = 0.0 #0.5 + 0.5 * (t+1) / self.args.num_rounds
                    x, y = self.dataset.weighted_sample(batch_size, self.args.rank_coeff)
                    if self.args.acq_fn.lower() == "ucb":
                        with torch.no_grad():
                            rs, mu, sigma = proxy(x, return_all=True)
                    else:
                        with torch.no_grad():
                            rs = proxy(x)
                            sigma = torch.zeros(batch_size).to(self.args.device)
                    radius = get_current_radius(it, t, self.args, rs=rs, y=y, sigma=sigma)
                    # import pdb; pdb.set_trace()
                    guide = torch.tensor(np.array(x)).to(self.args.device)
                    seqs = generator.decode(batch_size, max_len=self.args.gen_max_len, guide_seqs=guide, explore_radius=radius, temp=self.args.gen_sampling_temperature)
                    # print(np.mean(y), np.mean(self.oracle(seqs.tolist()).reshape(-1)))
                    vals = self.oracle(seqs.tolist()).reshape(-1)
                    rs = proxy(seqs.cpu())
                    try:
                        p_bar_log = {'gen_oracle': vals.mean(), 'gen_proxy': rs.mean().item(), 'sigma': sigma.mean().item(), 'radius': radius.mean().item()}
                        # p_bar.set_postfix({'oracle': vals.mean(), 'proxy': rs.mean().item(), 'sigma': sigma.mean().item(), 'radius': radius.mean().item()})
                    except:
                        pass

            # offline data (both)
            off_batch_size = self.args.gen_data_sample_per_step if batch_size > 0 else self.args.gen_data_sample_per_step + self.args.gen_episodes_per_step
            if self.args.use_rank_based_off:
                off_x, off_y = self.dataset.weighted_sample(off_batch_size, self.args.rank_coeff)  # rank-based? weighted_sample(batch_size) 
            else:
                off_x, off_y = self.dataset.sample(off_batch_size)
            
            if proxy is None:
                seqs = torch.tensor(np.array(off_x)).to(self.args.device)
                rs = torch.tensor(off_y).to(self.args.device)
                # loss = generator.train_step(seqs, torch.tensor(y).to(self.args.device))
            elif batch_size == 0:
                seqs = torch.tensor(np.array(off_x)).to(self.args.device)
                rs = proxy(seqs.cpu())
                vals = self.oracle(seqs.tolist()).reshape(-1)
                # rs = torch.tensor(off_y).to(self.args.device)
            else:
                seqs = torch.cat([seqs, torch.tensor(np.array(off_x)).to(self.args.device)], dim=0)
                
                if self.args.acq_fn.lower() == "ucb":
                    with torch.no_grad():
                        rs, mu, sigma = proxy(seqs.cpu(), return_all=True) # proxy.l2r requires numpy array
                else:
                    with torch.no_grad():
                        rs = proxy(seqs.cpu())
            # import pdb; pdb.set_trace()
            loss = generator.train_step(seqs, rs)
            # loss = generator.train_step(seqs, self.l2r(rs))
            p_bar_log['off_oracle'] = np.mean(off_y)
            p_bar_log['proxy'] = rs.mean().item()
            p_bar_log['loss'] = loss
            p_bar.set_postfix(p_bar_log)
            # p_bar.set_postfix({'oracle': vals.mean(), 'proxy': rs.mean().item(), 'loss': loss})
            # loss = generator.train_step(seqs, torch.from_numpy(self.l2r(rs)).to(self.args.device))
            
            losses.append(loss)
            wandb_log = {"generator_total_loss": loss}
            if self.args.use_wandb:
                wandb.log(wandb_log)
        return losses
    
    def propose_sequences(self, generator, proxy, t=0):
        print("Generating samples")
        samples, scores = [], []
        batch_size = self.args.num_queries_per_round
        num_iter = self.args.sample_width #* self.args.chain_length if self.args.chain_length > 0 else 100
        for _ in range(num_iter):
            if self.args.radius_option != "none":
                x, y = self.dataset.weighted_sample(batch_size, self.args.rank_coeff)
                # import pdb; pdb.set_trace()
                guide = torch.tensor(np.array(x)).to(self.args.device)
                    
                if self.args.acq_fn.lower() == "ucb":
                    with torch.no_grad():
                        rs, _, sigma = proxy(x, return_all=True)
                else:
                    with torch.no_grad():
                        rs = proxy(x)
                        sigma = torch.zeros(batch_size).to(self.args.device)
                radius = get_current_radius(self.args.gen_num_iterations, t, self.args, rs=rs, y=y, sigma=sigma)
                    
            else:
                guide = None
                radius = 1.0
            # import pdb; pdb.set_trace()
            seqs = generator.decode(batch_size, max_len=self.args.gen_max_len, guide_seqs=guide, explore_radius=radius, temp=self.args.gen_sampling_temperature)
            
            if self.args.filter:
                # seqs = self._filter_samples(seqs, self.dataset.train)
                # seqs = self._filter_samples(seqs, self.dataset.valid)
                seqs = self._filter_samples(seqs, samples)
                
            with torch.no_grad():
                rs = proxy(seqs.cpu()).reshape(-1)
            samples.extend(seqs.tolist())
            scores.extend(rs.tolist())
            guide = None
        
        idx_pick = np.argsort(scores)[::-1][:self.args.num_queries_per_round]
        return np.array(samples)[idx_pick] #.squeeze(1)
    
    def _filter_samples(self, samples, reference_set):
        filtered_samples = []
        for sample in samples:
            str_sample = "".join([self.oracle.itos[i.item()] for i in sample])
            similar = False
            for example in reference_set:
                str_example = "".join([self.oracle.itos[i] for i in example])
                if is_similar(str_sample, str_example, "edit", 2.0):
                    similar = True
                    break
            if not similar:
                filtered_samples.append(sample)
        return torch.stack(filtered_samples)

    def propose_sequences_with_mh(self, generator, proxy, t=0):
        print("Generating samples")
        samples, scores = [], []
        batch_size = self.args.num_queries_per_round
        guide = None
        num_iter = self.args.sample_width #* self.args.chain_length if self.args.chain_length > 0 else 100
        for k in range(num_iter):
            if self.args.radius_option != "none":
                if guide is None:
                    x, y = self.dataset.weighted_sample(batch_size, self.args.rank_coeff)
                    guide = torch.tensor(np.array(x)).to(self.args.device)
                    chain_step = 0
                else:
                    x = guide.tolist()
                    chain_step += 1
                    
                if self.args.acq_fn.lower() == "ucb":
                    with torch.no_grad():
                        rs, _, sigma = proxy(x, return_all=True)
                else:
                    with torch.no_grad():
                        rs = proxy(x)
                        sigma = torch.zeros(batch_size).to(self.args.device)
                if chain_step > 0:
                    radius = 0.1
                else:
                    radius = get_current_radius(self.args.gen_num_iterations, t, self.args, rs=rs, y=y, sigma=sigma)
                    
            else:
                guide = None
                radius = 1.0
            # import pdb; pdb.set_trace()
            if self.args.use_mutation_kernel and chain_step > 0:
                import pdb; pdb.set_trace()
                # seqs = 
            else:
                seqs = generator.decode(batch_size, max_len=self.args.gen_max_len, guide_seqs=guide, explore_radius=radius, temp=self.args.gen_sampling_temperature)
            
            if self.args.chain_length > 0:
                with torch.no_grad():
                    new_rs = proxy(guide.cpu()).reshape(-1) # proxy.l2r requires numpy array
                    
                    log_p = generator.get_log_prob(seqs)
                    ref_log_p = generator.get_log_prob(guide)
                    # accept_mask = (log_p - ref_log_p) > 0  # log prob is reward
                    lp_update = log_p - ref_log_p
                    update_dist = torch.distributions.Bernoulli(logits=lp_update/5)  # + delta_logp_traj)
                    accept_mask = update_dist.sample().bool()
                    # accept_mask += (new_rs.view(-1) / rs.view(-1)) > 1.
                    # accept_mask += (torch.rand(accept_mask.shape) < 0.2).to(accept_mask.device)
                    # accept_mask += (torch.rand(accept_mask.shape) < 0.5 *(1 - 0.1 * t)).to(accept_mask.device)
                    guide[accept_mask] = seqs[accept_mask]
                    rs = proxy(guide.cpu()).reshape(-1) # proxy.l2r requires numpy array
                if (k+1) % self.args.chain_length == 0:
                    samples.extend(guide[accept_mask].tolist())
                    scores.extend(rs[accept_mask].tolist())
                    guide = None
            else:
                with torch.no_grad():
                    rs = proxy(seqs.cpu()).reshape(-1)
                samples.extend(seqs.tolist())
                scores.extend(rs.tolist())
                guide = None
        
        idx_pick = np.argsort(scores)[::-1][:self.args.num_queries_per_round]
        return np.array(samples)[idx_pick] #.squeeze(1)

