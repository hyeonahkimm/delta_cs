import numpy as np

if __name__ == '__main__':
    x_all = np.load(f"./dataset/tfbind/tfbind-x-all.npy")
    y_all = np.load(f"./dataset/tfbind/tfbind-y-all.npy")
    
    for wt_idx in range(6,100):
        # wt_idx = np.random.choice(len(x_all))
        wt = x_all[wt_idx]
        wt_score = y_all[wt_idx]  # 0.4312
        collected_samples = [wt]
        collected_scores = [wt_score]
    
        local_max = 0.
        for x, y in zip(collected_samples, collected_scores):
            for i, seq in enumerate(x_all):
                if 0 < ((seq - x) != 0).sum() < 3 and y_all[i] < wt_score:
                    collected_samples.append(seq)
                    collected_scores.append(y_all[i])
                elif 0 < ((seq - x) != 0).sum() < 3 and y_all[i] > local_max:
                    local_max = y_all[i]
            if len(collected_samples) > 1000:
                break
        print(f"local_max: {local_max}")
        if local_max < 0.95: 
            break
        
    # np.save('dataset/tfbind/local-tfbind-x-init2.npy', np.array(collected_samples)) #0 .9322764194120315
    # np.save('dataset/tfbind/local-tfbind-y-init2.npy', collected_scores)