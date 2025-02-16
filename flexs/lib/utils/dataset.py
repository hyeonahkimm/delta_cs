import numpy as np
import torch


class BioSeqDataset():
    def __init__(self, args, tokenizer, init_data):
        self.args = args
        self.rng = np.random.RandomState(142857)
        # self._load_dataset(args.task)
        # self.val_added = len(self.valid)
        self.tokenizer = tokenizer
    
        seqs, scores = [], []
        for x, score in zip(*(init_data)):
            seqs.append([self.tokenizer.stoi[c] for c in x])
            scores.append(score)
        self.train = np.array(seqs)
        self.train_scores = np.array(scores)
        
        self.train_added = len(self.train)

    def sample(self, n):
        indices = np.random.randint(0, len(self.train), n)
        return ([self.train[i] for i in indices],
                [self.train_scores[i] for i in indices])
        
    def weighted_sample(self, n, rank_coefficient=0.01):
        ranks = np.argsort(np.argsort(-1 * self.train_scores))
        weights = 1.0 / (rank_coefficient * len(self.train_scores) + ranks)
            
        indices = list(torch.utils.data.WeightedRandomSampler(
            weights=weights, num_samples=n, replacement=True
            ))
        
        return ([self.train[i] for i in indices],
                [self.train_scores[i] for i in indices])

    def add(self, batch):
        samples, scores = batch
        train, val = [], []
        train_seq, val_seq = [], []
        for x, score in zip(samples, scores):
            train_seq.append([self.tokenizer.stoi[c] for c in x])
            # train_seq.append(np.array([self.tokenizer.stoi[c] for c in x]))
            train.append(score)
        
        self.train_scores = np.concatenate((self.train_scores, train), axis=0).reshape(-1)
        self.train = np.concatenate((self.train, train_seq), axis=0)
    
    def _tostr(self, seqs):
        return ["".join([str(i) for i in x]) for x in seqs]

    def _top_k(self, data, k):
        indices = np.argsort(data[1])[::-1][:k]
        topk_scores = data[1][indices]
        topk_prots = np.array(data[0])[indices]
        return self._tostr(topk_prots), topk_scores

    def top_k(self, k):
        data = (self.train, self.train_scores)
        return self._top_k(data, k)

    def top_k_collected(self, k):
        # scores = np.concatenate((self.train_scores[self.train_added:], self.valid_scores[self.val_added:]))
        # seqs = np.concatenate((self.train[self.train_added:], self.valid[self.val_added:]), axis=0)
        data = (self.train_scores[self.train_added:], self.train[self.train_added:])
        return self._top_k(data, k)

