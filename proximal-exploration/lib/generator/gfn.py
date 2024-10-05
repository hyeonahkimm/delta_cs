import torch
import torch.nn.functional as F

from lib.nn.mlp import MLP

LOGINF = 1000


class TBGFlowNetGenerator():
    def __init__(self, args, tokenizer):
        super().__init__(args)
        self.args = args
        self.gen_clip = 10
        self.leaf_coef = 25 #args.gen_leaf_coef
        self.out_coef = 10 #args.gen_output_coef
        self.reward_exp_min = 1e-32
        self.loss_eps = torch.tensor(float(1e-5)).to(args.device)
        self.pad_tok = 1
        self.num_tokens = 20
        if args.task == "avGFP":
            self.max_len = 238
        elif args.task == "AAV":
            self.max_len = 28
            
        self.tokenizer=tokenizer
        self.model = MLP(num_tokens=self.num_tokens, 
                                num_outputs=self.num_tokens, 
                                num_hid=1024,
                                num_layers=2,
                                max_len=self.max_len,
                                dropout=0,
                                partition_init=args.gen_partition_init,
                                causal=args.gen_do_explicit_Z)
        self.model.to(args.device)
        self.opt = torch.optim.Adam(self.model.model_params(), self.args.gen_lr, weight_decay=0.,
                            betas=(0.9, 0.999))
        self.opt_Z = torch.optim.Adam(self.model.Z_param(), 1e-3, weight_decay=0.,
                            betas=(0.9, 0.999))
        self.device = args.device
        self.logsoftmax = torch.nn.LogSoftmax(1)
        self.logsoftmax2 = torch.nn.LogSoftmax(2)

    def train_step(self, input_batch):
        strs, r = zip(*input_batch["bulk_trajs"])
        loss, info = self.get_loss(strs, r)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gen_clip)
        self.opt.step()
        self.opt_Z.step()
        self.opt.zero_grad()
        self.opt_Z.zero_grad()
        return loss, info

    @property
    def Z(self):
        return self.model.Z
    
    def get_loss(self, strs, r):
        # strs, r = zip(*batch["bulk_trajs"])
        
        s = self.tokenizer.process(strs).to(self.device)
        r = torch.tensor(r).to(self.device).clamp(min=0)
        r = torch.nan_to_num(r, nan=0, posinf=0, neginf=0)
        
        inp_x = F.one_hot(s, num_classes=self.num_tokens+1)[:, :, :-1].to(torch.float32)
        inp = torch.zeros(s.shape[0], self.max_len, self.num_tokens)
        inp[:, :inp_x.shape[1], :] = inp_x
        x = inp.reshape(s.shape[0], -1).to(self.device).detach()
        if self.args.task == "amp":
            lens = [self.max_len for i in s]
        else:
            lens = [len(i) for i in strs]
        pol_logits = self.logsoftmax2(self.model(x, None, return_all=True, lens=lens))[:-1]
        
        if self.args.task == "amp" and s.shape[1] != self.max_len:
            s = F.pad(s, (0, self.max_len - s.shape[1]), "constant", 21)
            mask = s.eq(21)
        else:
            mask = s.eq(self.num_tokens)
        s = s.swapaxes(0, 1)
        n = (s.shape[0] - 1) * s.shape[1]
        seq_logits = (pol_logits
                        .reshape((n, self.num_tokens))[torch.arange(n, device=self.device),(s[1:,].reshape((-1,))).clamp(0, self.num_tokens-1)]
                        .reshape(s[1:].shape)
                        * mask[:,1:].swapaxes(0,1).logical_not().float()).sum(0)
        # p(x) = R/Z <=> log p(x) = log(R) - log(Z) <=> log p(x) - log(Z)
        loss = (self.model.Z + seq_logits - r.clamp(min=self.reward_exp_min).log()).pow(2).mean()
        
        return loss, {}

    def forward(self, x, lens, return_all=False, coef=1, pad=2):
        inp_x = F.one_hot(x, num_classes=self.num_tokens+1)[:, :, :-1].to(torch.float32)
        inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens)
        inp[:, :inp_x.shape[1], :] = inp_x
        inp = inp.reshape(x.shape[0], -1).to(self.device)
        out = self.model(inp, None, lens=lens, return_all=return_all) * self.out_coef
        return out
        