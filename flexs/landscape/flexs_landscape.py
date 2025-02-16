
import flexs

from . import register_landscape

@register_landscape("flexs")
class FLEXS_Landscape:
    """
        A TAPE-based oracle model to simulate protein fitness landscape.
    """
    
    def __init__(self, args):
        
        if args.task == 'tfbind':
            self.problem = flexs.landscapes.tf_binding.registry()['SIX6_REF_R1']
            self.landscape = flexs.landscapes.TFBinding(**self.problem['params'])
            self.starting_sequence = self.problem['starts'][2]
        elif args.task.startswith('rna'):
            self.problem = flexs.landscapes.rna.registry()['L14_'+ args.task.upper()]  # L14_RNA1, L14_RNA2, L14_RNA3
            self.landscape = flexs.landscapes.RNABinding(**self.problem['params'])
            self.starting_sequence = None
        elif args.task == 'gfp':
            self.landscape = flexs.landscapes.BertGFPBrightness()
            self.starting_sequence = self.landscape.gfp_wt_sequence
        elif args.task == 'aav':
            self.problem = flexs.landscapes.additive_aav_packaging.registry()['liver']
            self.landscape = flexs.landscapes.AdditiveAAVPackaging(**self.problem['params'])
            self.starting_sequence = self.landscape.wild_type

    def get_fitness(self, sequences):
        # Input:  - sequences:      [query_batch_size, sequence_length]
        # Output: - fitness_scores: [query_batch_size]
        return self.landscape.get_fitness(sequences).tolist()