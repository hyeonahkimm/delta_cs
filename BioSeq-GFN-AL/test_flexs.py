
import pprint

import flexs
from flexs import baselines
import flexs.utils.sequence_utils as s_utils


def test_aav():
    additive_aav_problem = flexs.landscapes.additive_aav_packaging.registry()['liver']
    additive_aav_landscape = flexs.landscapes.AdditiveAAVPackaging(**additive_aav_problem['params'])
    additive_aav_wt = additive_aav_landscape.wild_type

    import pdb; pdb.set_trace() 
    pprint.pprint(additive_aav_problem)
    print(f'WT: {additive_aav_wt}, fitness: {additive_aav_landscape.get_fitness([additive_aav_wt])}')   


def test_rna_binding():
    rna_problem = flexs.landscapes.rna.registry()['L14_RNA1']
    rna_landscape = flexs.landscapes.RNABinding(**rna_problem['params'])
    pprint.pprint(rna_problem)
    
    starting_sequence = rna_problem['starts'][1]
    print(f'WT: {starting_sequence}, fitness: {rna_landscape.get_fitness([starting_sequence])}')


def test_rosetta():
    rosetta_problem = flexs.landscapes.rosetta.registry()['3mx7']
    protein_landscape = flexs.landscapes.RosettaFolding(**rosetta_problem['params'])
    protein_wt = protein_landscape.wt_pose.sequence()

    print()
    pprint.pprint(rosetta_problem)
    print(f'\nWT sequence: {protein_wt}, fitness: {protein_landscape.get_fitness([protein_wt])}')
    
    
def test_tfbinding():
    tf_binding_problem = flexs.landscapes.tf_binding.registry()['SIX6_REF_R1']
    tf_binding_landscape = flexs.landscapes.TFBinding(**tf_binding_problem['params'])
    tf_binding_start = tf_binding_problem['starts'][5]

    pprint.pprint(tf_binding_problem)
    print(f'\nSequence: {tf_binding_start}, fitness: {tf_binding_landscape.get_fitness([tf_binding_start])}')
    

def test_gfp():
    # landscape = flexs.landscapes.BertGFPBrightness()

    # seq_length = len(landscape.gfp_wt_sequence)
    # test_seqs = s_utils.generate_random_sequences(seq_length, 100, s_utils.AAS)
    # landscape.get_fitness(test_seqs)

    # # Clean up downloaded model
    # shutil.rmtree("fluorescence-model")
    bert_gfp_landscape = flexs.landscapes.BertGFPBrightness()
    bert_gfp_wt = bert_gfp_landscape.gfp_wt_sequence
    
    print(f'WT sequence: {bert_gfp_wt}, fitness: {bert_gfp_landscape.get_fitness([bert_gfp_wt])}')
    

if __name__ == '__main__':
    print('Testing Flexs')
    print('Testing AAV...')
    test_aav()
    # print('Testing RNA binding...')
    # test_rna_binding()
    # print('Testing TF binding...')
    # test_tfbinding()
    print('Testing GFP...')
    test_gfp()
    print('Testing Rosetta...')
    test_rosetta()
    