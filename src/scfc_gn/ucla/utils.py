import glob
from .models import FilePattern

def read_lines_txt(path, encoding="utf-8"):
    with open(path, "r", encoding=encoding, errors="replace") as f:
        return [line.strip() for line in f if line.strip()]


def get_files(pattern: FilePattern):
    dti_files = glob.glob(pattern.dti_files)
    fc_files = glob.glob(pattern.fc_files)
    xyz_files = glob.glob(pattern.xyz_files)
    region_files = glob.glob(pattern.region_files)
    return dti_files, fc_files, xyz_files, region_files


