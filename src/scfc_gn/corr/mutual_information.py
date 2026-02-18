import numpy as np


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