import numpy as np

from .models import FilesPaths, Matrices, Features
from .utils import read_lines_txt


def get_matrices(paths: FilesPaths) -> Matrices:
    sc_matrix = np.loadtxt(paths.sc_path)
    fc_matrix = np.loadtxt(paths.fc_path)
    xyz_matrix = np.loadtxt(paths.xyz_path)
    region_matrix = read_lines_txt(paths.region_path)
    return Matrices(sc_matrix= sc_matrix, fc_matrix= fc_matrix,
                    xyz_matrix= xyz_matrix, region_matrix= region_matrix)


def get_upper_tri(matrices : Matrices, mask) -> Features:
    sc_values = matrices.sc_matrix[mask].astype(float)
    fc_values = matrices.fc_matrix[mask].astype(float)
    return Features(sc_upper = sc_values, fc_upper= fc_values)
    