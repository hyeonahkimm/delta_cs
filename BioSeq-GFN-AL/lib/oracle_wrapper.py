import numpy as np

from clamp_common_eval.defaults import get_test_oracle
# import design_bench
import flexs


def get_oracle(args):
    if args.task == "amp":
        return AMPOracleWrapper(args)
    elif args.task == "gfp":
        return avGFPWrapper(args)
    elif args.task == "aav":
        return AAVWrapper(args)
    elif args.task == "tfbind":
        return TFBind8Wrapper(args)
    elif args.task.startswith("rna"):
        return RNAWrapper(args)


class AMPOracleWrapper:
    def __init__(self, args):
        self.oracle = get_test_oracle(args.oracle_split, 
                                        model=args.oracle_type, 
                                        feature=args.oracle_features, 
                                        dist_fn="edit", 
                                        norm_constant=args.medoid_oracle_norm)
        self.oracle.to(args.device)

    def __call__(self, x, batch_size=256):
        scores = []
        for i in range(int(np.ceil(len(x) / batch_size))):
            s = self.oracle.evaluate_many(x[i*batch_size:(i+1)*batch_size])
            if type(s) == dict:
                scores += s["confidence"][:, 1].tolist()
            else:
                scores += s.tolist()
        return np.float32(scores)


# class GFPWrapper:
#     def __init__(self, args):
#         import pdb; pdb.set_trace()
#         self.task = design_bench.make('GFP-Transformer-v0')

#     def __call__(self, x, batch_size=256):
#         scores = []
#         for i in range(int(np.ceil(len(x) / batch_size))):
#             s = self.task.predict(np.array(x[i*batch_size:(i+1)*batch_size])).reshape(-1)
#             scores += s.tolist()
#         return np.float32(scores)

# class TFBind8Wrapper:
#     def __init__(self, args):
#         self.task = design_bench.make('TFBind8-Exact-v0')

#     def __call__(self, x, batch_size=256):
#         scores = []
#         for i in range(int(np.ceil(len(x) / batch_size))):
#             s = self.task.predict(np.array(x[i*batch_size:(i+1)*batch_size]))
#             scores += s.tolist()
#         return np.array(scores)

########### based on flex codes ############
class TFBind8Wrapper:
    def __init__(self, args):
        self.alphabet = ['T', 'G', 'C', 'A']  # based on design bench alphabet
        self.itos = {idx: value for idx, value in enumerate(self.alphabet)}
        # self.task = design_bench.make('TFBind8-Exact-v0')
        tf_binding_problem = flexs.landscapes.tf_binding.registry()['SIX6_REF_R1']
        self.landscape = flexs.landscapes.TFBinding(**tf_binding_problem['params'])

    def __call__(self, x):
        seqs = []
        for state in x:
            seqs.append(''.join([self.itos[i] for i in state]))
        
        # scores = self.landscape.get_fitness(seqs)
        # scores[scores < 0.3] = 0
        # return scores
        return self.landscape.get_fitness(seqs)
    
class RNAWrapper:
    def __init__(self, args):
        self.alphabet = ['U', 'G', 'C', 'A']
        self.itos = {idx: value for idx, value in enumerate(self.alphabet)}
        problem = flexs.landscapes.rna.registry()['L14_'+ args.task.upper()]  # L14_RNA1, L14_RNA2, L14_RNA3
        self.landscape = flexs.landscapes.RNABinding(**problem['params'])
        
    def __call__(self, x):
        seqs = []
        for state in x:
            seqs.append(''.join([self.itos[i] for i in state]))
        
        return self.landscape.get_fitness(seqs)
    

class avGFPWrapper:
    def __init__(self, args):
        self.alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        self.itos = {idx: value for idx, value in enumerate(self.alphabet)}
        self.stoi = {value: idx for idx, value in enumerate(self.alphabet)}
        self.landscape = flexs.landscapes.BertGFPBrightness()
        self.wt = self.landscape.gfp_wt_sequence

    def __call__(self, x):
        seqs = []
        for state in x:
            seqs.append(''.join([self.itos[i] for i in state]))
        
        return self.landscape.get_fitness(np.array(seqs))


class AAVWrapper:
    def __init__(self, args):
        self.alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        self.itos = {idx: value for idx, value in enumerate(self.alphabet)}
        self.stoi = {value: idx for idx, value in enumerate(self.alphabet)}
        additive_aav_problem = flexs.landscapes.additive_aav_packaging.registry()['liver']
        self.landscape = flexs.landscapes.AdditiveAAVPackaging(**additive_aav_problem['params'])
        self.wt = self.landscape.wild_type

    def __call__(self, x):
        seqs = []
        for state in x:
            seqs.append(''.join([self.itos[i] for i in state]))
        seqs = np.array(seqs)
        return self.landscape.get_fitness(seqs)
