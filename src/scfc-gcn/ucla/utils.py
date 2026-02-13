def read_lines_txt(path, encoding="utf-8"):
    with open(path, "r", encoding=encoding, errors="replace") as f:
        return [line.strip() for line in f if line.strip()]