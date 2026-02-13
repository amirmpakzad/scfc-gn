from dataclasses import dataclass

import numpy as np 
from scipy.stats import spearmanr

@dataclass
class Corr:
    pearson: float = 0
    spearman_rho: float = 0
    spearman_pval: float = 0
    mi_rho: float = 0
    mi_pval: float = 0


def compute_corr(feat : Features) -> Corr:
    pearson = np.corrcoef(feat.fc_upper, feat.sc_upper)[0, 1]
    r_spearman, p_spearman = spearmanr(feat.fc_upper, feat.sc_upper)
    _, _, r_mi, p_mi = mi_test(feat.fc_upper, feat.sc_upper)
    corr = Corr(
        pearson = pearson.item(),
        spearman_rho = r_spearman.item(),
        spearman_pval= p_spearman.item(),
        mi_rho= r_mi,
        mi_pval= p_mi 
    )
    return corr