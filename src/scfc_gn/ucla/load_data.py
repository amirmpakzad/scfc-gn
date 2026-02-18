from glob import glob
from typing import Tuple, List
from .models import *
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence
from pathlib import Path
from .utils import read_lines_txt, get_files
from .matrices import get_matrices, get_upper_tri
from scfc_gn.corr.mutual_information import mi_test


__all__ = [
    "get_all_subjects",
    "create_sub_graph_matricies",
    "get_corr",
    "boxplot",
]


def get_subject_id(fn : str, remove_suff : bool = False):
    nm = fn.split("_")[0]
    s = Path(nm).parts[1]
    # s = nm.split('/')[1]
    if remove_suff and s.endswith(("B", "C", "D")):
        s = s[:-1]
    return s


def get_file_maps(pattern: FilePattern, remove_suffix: bool= True):
    dti_files, fc_files, xyz_files, region_files = get_files(pattern)
    dti_map = {get_subject_id(f, remove_suffix): f for f in dti_files}
    fc_map = {get_subject_id(f, remove_suffix) : f for f in fc_files}
    xyz_map = {get_subject_id(f, remove_suffix): f for f in xyz_files}
    region_map = {get_subject_id(f, remove_suffix): f for f in region_files}
    return dti_map, fc_map, xyz_map, region_map



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
    

def get_subjects_map(pattern: FilePattern):
    dti_map, fc_map, xyz_map, region_map = get_file_maps(pattern)
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


def get_all_subjects(pattern: FilePattern, remove_suffix: bool= True) -> Tuple[List[Subject], Subject]:
    subjects = get_subjects_map(pattern)
    subjects_list = []
    group_subject = None

    for i in range(len(subjects)):
        id = subjects[i]

        group = get_subject_group_by_id(id)

        maps = get_file_maps(pattern, remove_suffix)
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


