import numpy as np

ensemble_rules = {
    'mean': lambda x: np.mean(x, axis=0),
    'lcb': lambda x: np.mean(x, axis=0) - np.std(x, axis=0),
    'ucb': lambda x: np.mean(x, axis=0) + np.std(x, axis=0),
    'std': lambda x: np.std(x, axis=0),
}

class Ensemble:
    def __init__(self, models, ensemble_rule):
        self.models = models
        self.ensemble_func = ensemble_rules[ensemble_rule]
        self.std = ensemble_rules['std']
    
    def train(self, sequences, labels):
        for model in self.models:
            model.train(sequences, labels)
            
    def train_prioritized(self, sequences, labels, rank_coefficient=0.01):
        for model in self.models:
            model.train_prioritized(sequences, labels, rank_coefficient)

    def get_fitness(self, sequences, return_std=False):
        # Input:  - sequences:   [batch_size, sequence_length]
        # Output: - predictions: [batch_size]

        if return_std:
            ys = [model.get_fitness(sequences) for model in self.models]
            
            return self.ensemble_func(ys), self.std(ys)
        
        return self.ensemble_func([model.get_fitness(sequences) for model in self.models])
