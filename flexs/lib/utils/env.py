import torch
import numpy as np

class Vocab:
    def __init__(self, alphabet) -> None:
        self.stoi = {}
        self.itos = {}
        for i, alphabet in enumerate(alphabet):
            self.stoi[alphabet] = i
            self.itos[i] = alphabet

class TokenizerWrapper:
    def __init__(self, vocab, dummy_process):
        self.vocab = vocab
        self.dummy_process = dummy_process
        self.eos_token = '%'
    
    def process(self, x):
        lens = [len(x[i]) for i in range(len(x))]
        if self.dummy_process:
            max_len = max(lens)
            if max_len != sum(lens) / len(lens):
                for i in range(len(x)):
                    if len(x[i]) == max_len:
                        pass
                    try:
                        x[i] = x[i] + [len(self.stoi.keys())] * (max_len - len(x[i]))
                    except:
                        import pdb; pdb.set_trace();
        else:
            ret_val = []
            max_len = max(lens)
            for i in range(len(x)):
                # process
                temp = [self.stoi[ch] for ch in x[i]]
                if max_len != sum(lens) / len(lens):
                    if len(temp) == max_len:
                        pass
                    try:
                        temp = temp + [len(self.stoi.keys())] * (max_len - len(temp))
                    except:
                        import pdb; pdb.set_trace();
                ret_val.append(temp)
            x = ret_val
        # import pdb; pdb.set_trace()
        x = np.array(x)
        return torch.tensor(x, dtype=torch.long)
    
    def decode(self, seqs):
        strings = []
        for seq in seqs:
            strings.append(''.join([self.itos[i] for i in seq.tolist()]))
        return strings
    
    def encode(self, strings):
        seqs = []
        for string in strings:
            seqs.append([self.stoi[c] for c in string])
        return seqs

    @property
    def itos(self):
        return self.vocab.itos

    @property
    def stoi(self):
        return self.vocab.stoi


def get_tokenizer(args, alphabet):
    # if args.task == "amp":
    #     alphabet = ['%', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    #     # %: EOS
    # elif args.task == "tfbind":
    #     alphabet = ['A', 'C', 'G', 'T']  # design-bench: ['A', 'C', 'G', 'T']  (not used)
    # elif args.task.startswith("rna"):
    #     alphabet = ['A', 'C', 'G', 'U']
    # elif args.task in ["gfp", "aav"]:
    #     alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    vocab = Vocab(alphabet)
    tokenizer = TokenizerWrapper(vocab, dummy_process=(args.task != "amp"))
    return tokenizer