import itertools
import wandb
import numpy as np
from polyleven import levenshtein


def is_similar(seq_a, seq_b, dist_type="edit", threshold=0):
    if dist_type == "edit":
        return edit_dist(seq_a, seq_b) < threshold
    return False


def edit_dist(seq1, seq2):
    return levenshtein(seq1, seq2) / 1


def mean_pairwise_distances(seqs):
    dists = []
    for pair in itertools.combinations(seqs, 2):
        dists.append(edit_dist(*pair))
    return np.mean(dists)


def mean_novelty(seqs, ref_seqs):
    novelty = [min([edit_dist(seq, ref) for ref in ref_seqs]) for seq in seqs]
    return np.mean(novelty)


def log_overall_metrics(args, dataset, t, collected=False, new_seqs=[], seed_seqs=[], query=None, wt=None):
    top100 = dataset.top_k(128)
    dist100 = mean_pairwise_distances(top100[0])
    ref_seqs, _ = dataset.get_ref_data()
    novelty100 = mean_novelty(top100[0], ref_seqs)
    edit = np.mean([edit_dist(s, ss) for s, ss in zip(new_seqs, seed_seqs)]) if len(seed_seqs) > 0 else 0
    # print("Avg. Edit Dist from seed sequences", edit)
    print("========= Round", t, "=========")
    print("Scores, 128", np.max(top100[1]), np.mean(top100[1]))
    print("Dist, 128", dist100)
    print("Novelty, 128", novelty100)
    log = {'top-128': np.mean(top100[1]),
           'dist-128': dist100,
           'nov-128': novelty100,
           'edit_dist': edit,
           'round': t}
    if collected:
        top100 = dataset.top_k_collected(128)
        dist100 = mean_pairwise_distances(top100[0])
        if wt is not None:
            dist_from_wt = mean_novelty(top100[0][0], [wt])
            print("Collected Dist from WT", dist_from_wt)
            log["collected_dist_from_wt"] = dist_from_wt
        novelty100 = mean_novelty(top100[0], ref_seqs)
        print("Collected Scores, 128, max, 50 pl", np.mean(top100[1]), np.max(top100[1]), np.percentile(top100[1], 50))
        print("Collected Dist and Novelty, 128", dist100, novelty100)
        log["collected_top-128"] = np.mean(top100[1])
        log["collected_max"] = np.max(top100[1])
        log["collected_50pl"] = np.percentile(top100[1], 50)
        log["collected_dist-128"] = dist100
        log["collected_novelty-128"] = novelty100
    if query is not None:
        log["queried"] = query[1].mean()
    
    if args.use_wandb:
        wandb.log(log)