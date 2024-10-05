import time
import numpy as np
import pandas as pd
import wandb
import itertools

from lib.utils.dataset import BioSeqDataset
from utils.seq_utils import hamming_distance

from polyleven import levenshtein


def mean_pairwise_distances(seqs):
    dists = []
    for pair in itertools.combinations(seqs, 2):
        dists.append(hamming_distance(*pair))
    return np.mean(dists)


def mean_distances_from_wt(seqs, wt):
    dists = []
    for seq in seqs:
        dists.append(hamming_distance(seq, wt))
    return np.mean(dists)

def mean_novelty(seqs, ref_seqs):
    novelty = [min([hamming_distance(seq, ref) for ref in ref_seqs]) for seq in seqs]
    return np.mean(novelty)

class Runner:
    """
        The interface of landscape/model/explorer is compatible with FLEXS benchmark.
        - Fitness Landscape EXploration Sandbox (FLEXS)
          https://github.com/samsinai/FLEXS
    """
    
    def __init__(self, args):
        self.num_rounds = args.num_rounds
        self.num_queries_per_round = args.num_queries_per_round
        self.method = args.alg
        self.use_wandb = args.use_wandb
        self.args = args

    def run(self, landscape, starting_sequence, model, explorer, starting_dataset=None):
        self.results = pd.DataFrame()
        if starting_dataset is None:
            starting_fitness = landscape.get_fitness([starting_sequence])[0]
            self.update_results(0, [starting_sequence], [starting_fitness])
        else:
            self.update_results(0, starting_dataset[0].tolist(), starting_dataset[1].tolist())
        
        if self.method in ['gfn-al', 'gfn_seq_editor']: #update dataset too
            explorer.dataset = BioSeqDataset(explorer.args, explorer.tokenizer, init_data=starting_dataset)
        
        for round in range(1, self.num_rounds+1):
            round_start_time = time.time()
            
            if self.args.use_rank_based_proxy_training:
                # model.train(self.sequence_buffer, self.fitness_buffer)
                model.train_prioritized(self.sequence_buffer, self.fitness_buffer)
            elif self.method is not 'gfn_seq_editor':
                model.train(self.sequence_buffer, self.fitness_buffer)
                
            sequences, model_scores = explorer.propose_sequences(self.results)
            assert len(sequences) <= self.num_queries_per_round
            true_scores = landscape.get_fitness(sequences)

            round_running_time = time.time()-round_start_time
            self.update_results(round, sequences, true_scores, starting_sequence, round_running_time)
            # self.update_results(round, sequences, true_scores, starting_sequence, round_running_time)
            
            if self.method == 'gfn-al': #update dataset too
                explorer.dataset.add((sequences, true_scores))
                explorer.round = round
              
        if self.use_wandb:
            # Convert pandas DataFrame to a wandb.Table and log it
            wandb_table = wandb.Table(dataframe=self.results)
            wandb.log({"rst": wandb_table})
        self.results.to_csv(f'./results/{self.args.task}/{self.method}_{self.args.name}_{self.args.seed}.csv', index=False)
    
    def update_results(self, round, sequences, true_scores, wt=None, running_time=0.0):
        self.results = self.results.append(
            pd.DataFrame({
                "round": round,
                "sequence": sequences,
                "true_score": true_scores
            })
        )
        print('round: {}  max fitness score: {:.3f}  running time: {:.2f} (sec)'.format(round, self.results['true_score'].max(), running_time))
        if self.use_wandb:
            top100 = self.results.nlargest(self.args.num_queries_per_round, 'true_score')
            div100 = mean_pairwise_distances(top100['sequence'])
            avg100 = top100['true_score'].mean()
            initial = self.results[self.results['round']==0]['sequence']
            novelty100 = mean_novelty(top100['sequence'], initial)
            median = np.percentile(top100['true_score'], 50)
            dist100_from_wt = mean_distances_from_wt(top100['sequence'], wt) if wt is not None else 0
            queried_dist100 = mean_distances_from_wt(sequences, wt) if wt is not None else 0
            queried_div = mean_pairwise_distances(sequences)
            queried_score = np.mean(true_scores)
            wandb.log({'round': round,  'running_time': running_time,
                       'max_fitness': self.results['true_score'].max(), 'median_topK': median, 'avg_topK': avg100,
                       'div100': div100, 'novelty100': novelty100, 'dist100_from_wt': dist100_from_wt,
                       'queried_div': queried_div, 'queried_score': queried_score, 'queried_dist_from_wt': queried_dist100})
    
    @property
    def sequence_buffer(self):
        return self.results['sequence'].to_numpy()

    @property
    def fitness_buffer(self):
        return self.results['true_score'].to_numpy()

