from glob import glob
from typing import Tuple, List
from models import *
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence
from pathlib import Path
from .utils import read_lines_txt
from .matrices import get_matrices, get_upper_tri


__all__ = [
    "get_all_subjects",
    "create_sub_graph_matricies",
    "get_corr",
    "boxplot",
]

dti_pattern = "data/*_DTI_connectivity_matrix_file.txt"
fc_pattern  = "data/*_rsfMRI_connectivity_matrix_file.txt"
xyz_pattern = "data/*_DTI_region_xyz_centers_file.txt"
region_pattern = "data/*_DTI_region_names_full_file.txt"


def get_subject_id(fn : str, remove_suff : bool = False):
    nm = fn.split("_")[0]
    s = Path(nm).parts[1]
    # s = nm.split('/')[1]
    if remove_suff and s.endswith(("B", "C", "D")):
        s = s[:-1]
    return s


def get_file_maps(remove_suffix: bool= True):
    dti_files, fc_files, xyz_files, region_files = get_files()
    dti_map = {get_subject_id(f, remove_suffix): f for f in dti_files}
    fc_map = {get_subject_id(f, remove_suffix) : f for f in fc_files}
    xyz_map = {get_subject_id(f, remove_suffix): f for f in xyz_files}
    region_map = {get_subject_id(f, remove_suffix): f for f in region_files}
    return dti_map, fc_map, xyz_map, region_map


def get_files():
    dti_files = glob(dti_pattern)
    fc_files = glob(fc_pattern)
    xyz_files = glob(xyz_pattern)
    region_files = glob(region_pattern)
    return dti_files, fc_files, xyz_files, region_files



def get_subject_group_by_id(subject_id: str) -> SubjectGroup:
    prefix = subject_id.strip().upper()

    if prefix.startswith("TD"):
        return SubjectGroup.CONTROL
    elif prefix.startswith("ASD"):
        return SubjectGroup.AUTISM
    elif prefix.startswith("ALL"):
        return SubjectGroup.ALL
    else:
        raise ValueError(f"Unknown subject id prefix: {subject_id}")
    

def get_subjects_map():
    dti_map, fc_map, xyz_map, region_map = get_file_maps()
    subjects = sorted(
        set(dti_map.keys()) &
        set(fc_map.keys()) &
        set(xyz_map.keys()) &
        set(region_map.keys())
    )
    print("Subjects with SC+FC+XYZ:", len(subjects) - 1)
    return subjects






def get_mask(sc_mat):
    n = sc_mat.shape[0]
    mask = np.triu(np.ones((n, n), dtype = bool), k =1)
    nonZero_sc_edge_mask = sc_mat != 0 
    final_mask = mask & nonZero_sc_edge_mask
    return final_mask


def get_all_subjects(remove_suffix: bool= True) -> Tuple[List[Subject], Subject]:
    subjects = get_subjects_map()
    subjects_list = []
    group_subject = None

    for i in range(len(subjects)):
        id = subjects[i]

        group = get_subject_group_by_id(id)

        maps = get_file_maps(remove_suffix)
        files_path = FilesPaths(maps[0][id], maps[1][id], maps[2][id], maps[3][id])
        
        matrices = get_matrices(files_path)
        final_mask = get_mask(matrices.sc_matrix)
        features = get_upper_tri(matrices, final_mask)
        
        whole_data = NetworkData(matrices = matrices, features= features,
                                    type=NetworkType.WHOLE)

        sub = Subject(id=id, group=group, files_paths= files_path, whole_data= whole_data)

        if  id.startswith("All"):
            group_subject = sub
        else :
            subjects_list.append(sub)

    return subjects_list, group_subject



def create_idx(names : Sequence, *, dmn_keys, smn_keys, visual_keys, log = True):
    idx_dmn = [i for i, name in enumerate(names) if any(k in name for k in dmn_keys)]
    idx_smn = [i for i, name in enumerate(names) if any(k in name for k in smn_keys)]
    idx_visual = [i for i, name in enumerate(names) if any(k in name for k in visual_keys)]

    if log:
        print(f'{len(idx_dmn)}, {len(idx_smn)}, {len(idx_visual)}')

    return idx_dmn, idx_smn, idx_visual


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

def get_corr(all_subs: list[Subject]) -> list[Subject]:
    for sub in all_subs:
        for attr in ("whole_data", "dmn_network", "smn_network", "visual_network"):
            net = getattr(sub, attr, None)
            if net is None:
                continue
            net.corr = compute_corr(net.features)
    return all_subs


def induced_submatrix(C, idx):
    idx = np.array(idx, dtype=int)
    return C[np.ix_(idx, idx)]


def get_subgraph(sc_mat, fc_mat, region_names, xyz, idx, type):
    sc_sub_net = induced_submatrix(sc_mat, idx)
    fc_sub_net = induced_submatrix(fc_mat, idx)
    region_sub = [region_names[i] for i in idx]
    xyz_sub = xyz[idx, :]
    matrices = Matrices(fc_matrix= sc_sub_net, sc_matrix= fc_sub_net,
                        xyz_matrix= xyz_sub, region_matrix= region_sub)
    mask = get_mask(sc_sub_net)
    sc_values = sc_sub_net[mask].astype(float)
    fc_values = fc_sub_net[mask].astype(float)
    features = Features(sc_upper = sc_values, fc_upper= fc_values)
    network_data = NetworkData(
        matrices= matrices,
        features= features,
        type = type
    )
    return network_data


def create_sub_graph_matricies(
        all_subs: list[Subject],
        *,
        dmn_keys, 
        visual_keys,
        smn_keys):
    
    subs = all_subs
    for i in range(len(subs)):
        sub = subs[i]
        data = sub.whole_data
        sc_mat = data.matrices.sc_matrix
        fc_mat = data.matrices.fc_matrix
        region_mat = data.matrices.region_matrix
        xyz = data.matrices.xyz_matrix
        idx_dmn, idx_smn, idx_visual = create_idx(data.matrices.region_matrix,
                                                          dmn_keys= dmn_keys,
                                                          visual_keys= visual_keys,
                                                          smn_keys= smn_keys,
                                                          log= False)
        
        sub.dmn_network = get_subgraph(sc_mat, fc_mat, region_mat, xyz,
                                        idx_dmn, NetworkType.DMN)
        sub.smn_network = get_subgraph(sc_mat, fc_mat, region_mat, xyz,
                                       idx_smn, NetworkType.SMN)
        sub.visual_network = get_subgraph(sc_mat, fc_mat, region_mat, xyz,
                                          idx_visual, NetworkType.VISUAL)
        
    return subs


def mi_coupling_node_permutation(SC, FC, mask, permutations=300, seed=7):
    rng = np.random.default_rng(seed)

    sc_edges = SC[mask].astype(float)
    fc_edges = FC[mask].astype(float)

    observed = mutual_information_test(sc_edges, fc_edges)

    count = 0
    n = SC.shape[0]
    for _ in range(permutations):
        p = rng.permutation(n)
        FCp = FC[np.ix_(p, p)]
        fc_p_edges = FCp[mask].astype(float)
        if mutual_information_test(sc_edges, fc_p_edges) >= observed:
            count += 1

    pval = (count + 1) / (permutations + 1)
    return observed, pval




def mutual_information_test(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows")
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]

    xy = np.concatenate([x, y], axis=1)
    sx = np.cov(x, rowvar=False, bias=True)
    sy = np.cov(y, rowvar=False, bias=True)
    sxy = np.cov(xy, rowvar=False, bias=True)

    if sx.ndim == 0:
        sx = np.array([[float(sx)]])
    if sy.ndim == 0:
        sy = np.array([[float(sy)]])
    if sxy.ndim == 0:
        sxy = np.array([[float(sxy)]])

    sx += eps * np.eye(sx.shape[0])
    sy += eps * np.eye(sy.shape[0])
    sxy += eps * np.eye(sxy.shape[0])

    sign_x, logdet_x = np.linalg.slogdet(sx)
    sign_y, logdet_y = np.linalg.slogdet(sy)
    sign_xy, logdet_xy = np.linalg.slogdet(sxy)
    if sign_x <= 0 or sign_y <= 0 or sign_xy <= 0:
        return 0.0
    return 0.5 * (logdet_x + logdet_y - logdet_xy)


def mutual_information_p_value(
    x: np.ndarray,
    y: np.ndarray,
    permutations: int = 1000,
    random_state: int | None = None,
    eps: float = 1e-12,
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows")
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if permutations < 1:
        raise ValueError("permutations must be >= 1")

    rng = np.random.default_rng(random_state)
    observed = mutual_information_test(x, y, eps=eps)
    count = 0
    for _ in range(permutations):
        perm = rng.permutation(y.shape[0])
        if mutual_information_test(x, y[perm], eps=eps) >= observed:
            count += 1
    return (count + 1) / (permutations + 1)


def mi_to_rho(mi_nats: float) -> float:
    """Convert Gaussian MI (nats) to equivalent correlation magnitude rho in [0,1]."""
    mi_nats = max(float(mi_nats), 0.0)
    # rho = sqrt(1 - exp(-2I))
    val = 1.0 - np.exp(-2.0 * mi_nats)
    # numerical safety
    if val < 0:
        val = 0.0
    if val > 1:
        val = 1.0
    return float(np.sqrt(val))


def mi_test(x, y, permutations=300, random_state=7, eps=1e-12, log=True):
    mi = mutual_information_test(x, y, eps=eps)  # nats
    rho = mi_to_rho(mi)
    mi_p = mutual_information_p_value(x, y, permutations=permutations, random_state=random_state, eps=eps)

    
    mi_bits = mi / np.log(2)

    if log:
        print(
            f"MI={mi:.6f} nats ({mi_bits:.6f} bits) | rho_equiv={rho:.4f} | p_perm={mi_p:.4f} "
            f"(permutations={permutations})"
        )
    return  mi,  mi_bits, rho, mi_p
