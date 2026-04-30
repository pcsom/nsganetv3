"""Large child pool generation for surrogate-assisted candidate selection."""
import numpy as np


def stable_arch_key(arch):
    return (tuple(arch['ks']), tuple(arch['e']), tuple(arch['d']), int(arch['r']))


def chromosome_bounds(search_space):
    n_var = 46
    xl = np.zeros(n_var, dtype=np.int64)
    xu = 2 * np.ones(n_var, dtype=np.int64)
    xu[-1] = len(search_space.resolution) - 1
    return xl, xu


def two_point_crossover(rng, p1, p2):
    n = len(p1)
    if n < 2:
        return p1.copy()
    c1, c2 = sorted(rng.choice(n, size=2, replace=False))
    child = p1.copy()
    child[c1:c2] = p2[c1:c2]
    return child


def mutate_int_bounds(rng, x, xl, xu, prob_per_gene):
    y = x.copy()
    for i in range(len(x)):
        if rng.random() < prob_per_gene:
            y[i] = rng.integers(int(xl[i]), int(xu[i]) + 1)
    return y


def generate_unique_child_pool(
    nd_X,
    search_space,
    pool_size,
    rng,
    crossover_prob=0.9,
    mutation_prob=None,
    max_attempts_multiplier=100,
):
    """
    Sample many offspring by mating parents from nd_X (integer chromosomes).
    Deduplicates exact chromosomes to avoid redundant surrogate work.
    """
    xl, xu = chromosome_bounds(search_space)
    if mutation_prob is None:
        mutation_prob = 1.0 / len(xl)

    seen = set()
    rows = []
    max_attempts = max(pool_size * max_attempts_multiplier, pool_size + 1)
    attempts = 0

    nd_X = np.asarray(nd_X, dtype=np.int64)
    if len(nd_X) == 0:
        raise ValueError("nd_X is empty; need at least one parent architecture")

    while len(rows) < pool_size and attempts < max_attempts:
        attempts += 1
        i, j = rng.integers(0, len(nd_X), size=2)
        if i == j and len(nd_X) > 1:
            continue
        p1, p2 = nd_X[i], nd_X[j]
        if rng.random() < crossover_prob:
            child = two_point_crossover(rng, p1, p2)
        else:
            child = p1.copy()
        child = mutate_int_bounds(rng, child, xl, xu, mutation_prob)
        child = np.clip(child, xl, xu).astype(np.int64)
        key = tuple(child.tolist())
        if key in seen:
            continue
        seen.add(key)
        rows.append(child)

    return np.array(rows, dtype=np.int64)
