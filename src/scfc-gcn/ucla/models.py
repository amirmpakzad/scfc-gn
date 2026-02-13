from dataclasses import dataclass
import numpy as np
from enum import Enum

 
@dataclass(frozen=True)
class Matrices:
    fc_matrix: np.ndarray
    sc_matrix: np.ndarray
    xyz_matrix: np.ndarray
    region_matrix : np.ndarray


@dataclass(frozen=True)
class Features:
    sc_upper: np.ndarray
    fc_upper: np.ndarray
    

class NetworkType(Enum):
    WHOLE = "whole brain"
    DMN = "default mode netwrok"
    VISUAL = "visual network"
    SMN = "somato"


@dataclass
class Corr:
    pearson: float = 0
    spearman_rho: float = 0
    spearman_pval: float = 0
    mi_rho: float = 0
    mi_pval: float = 0


@dataclass
class NetworkData:
    matrices : Matrices
    features : Features
    type: NetworkType
    corr: Corr|None = None


class SubjectGroup(Enum):
    CONTROL = "control"
    AUTISM = "autism"
    ALL = "all"

@dataclass
class FilesPaths:
    sc_path : str
    fc_path : str
    xyz_path : str
    region_path : str


@dataclass
class Subject:
    id:str
    group: SubjectGroup
    files_paths: FilesPaths
    whole_data: NetworkData | None = None
    smn_network: NetworkData | None = None
    dmn_network: NetworkData | None = None
    visual_network: NetworkData | None = None