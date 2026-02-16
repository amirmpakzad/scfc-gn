from dataclasses import dataclass

@dataclass
class Corr:
    pearson: float = 0
    spearman_rho: float = 0
    spearman_pval: float = 0
    mi_rho: float = 0
    mi_pval: float = 0