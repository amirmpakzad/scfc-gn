import numpy as np

from .models import FilesPaths, Matrices, Features
from .utils import read_lines_txt


def get_matrices(paths: FilesPaths) -> Matrices:
    sc_matrix = np.loadtxt(paths.sc_path)
    fc_matrix = np.loadtxt(paths.fc_path)
    xyz_matrix = np.loadtxt(paths.xyz_path)

    np.fill_diagonal(sc_matrix, 0.0)
    np.fill_diagonal(fc_matrix, 0.0)

    if not np.isfinite(fc_matrix).all():
        bad = np.argwhere(~np.isfinite(fc_matrix))
        raise ValueError(f"Non-finite FC entries beyond diagonal in {paths.fc_path}. "
                         f"Example positions: {bad[:10].tolist()}")

    region_matrix = read_lines_txt(paths.region_path)
    return Matrices(sc_matrix= sc_matrix, fc_matrix= fc_matrix,
                    xyz_matrix= xyz_matrix, region_matrix= region_matrix)


def get_upper_tri(matrices : Matrices, mask) -> Features:
    sc_values = matrices.sc_matrix[mask].astype(float)
    fc_values = matrices.fc_matrix[mask].astype(float)
    return Features(sc_upper = sc_values, fc_upper= fc_values)


def handle_inf(t):
     return np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)