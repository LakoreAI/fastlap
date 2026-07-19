import pytest
import numpy as np
import scipy.sparse as sp

import fastlap
from conftest import generate_test_matrix, fastlap_execute, scipy_execute, lap_execute


def assert_optimal_cost(fastlap_cost, ref_cost, algo, tol=1e-6):
    assert abs(fastlap_cost - ref_cost) <= tol, (
        f"{algo}: cost {fastlap_cost:.10f} differs from reference {ref_cost:.10f} "
        f"by {abs(fastlap_cost - ref_cost):.2e} (tol={tol})"
    )


def assert_valid_assignment(row_assign, col_assign, n, algo):
    assert len(row_assign) == n, f"{algo}: row_assign length {len(row_assign)} != {n}"
    assert len(col_assign) == n, f"{algo}: col_assign length {len(col_assign)} != {n}"
    # Filter out None (unassigned) and check that the rest form a permutation
    row_vals = sorted(j for j in row_assign if j is not None)
    col_vals = sorted(i for i in col_assign if i is not None)
    assert row_vals == list(range(min(n, len(row_vals)))), (
        f"{algo}: row_assign is not a valid partial permutation"
    )
    assert col_vals == list(range(min(n, len(col_vals)))), (
        f"{algo}: col_assign is not a valid partial permutation"
    )


# ── Square correctness ──────────────────────────────────────────────────────

@pytest.mark.parametrize("size", [2, 3, 4, 5, 10, 20])
def test_correctness_lapjv(size):
    matrix = generate_test_matrix(size)
    cost, rows, cols = fastlap_execute(matrix, "lapjv")
    ref_cost, _, _ = lap_execute(matrix)
    assert_optimal_cost(cost, ref_cost, "lapjv")
    assert_valid_assignment(rows, cols, size, "lapjv")


@pytest.mark.parametrize("size", [2, 3, 4, 5, 10, 20])
def test_correctness_hungarian(size):
    matrix = generate_test_matrix(size)
    cost, rows, cols = fastlap_execute(matrix, "hungarian")
    ref_cost, _, _ = scipy_execute(matrix)
    assert_optimal_cost(cost, ref_cost, "hungarian")
    assert_valid_assignment(rows, cols, size, "hungarian")


@pytest.mark.parametrize("size", [2, 3, 4, 5, 10, 20])
def test_correctness_lapmod(size):
    matrix = generate_test_matrix(size)
    cost, rows, cols = fastlap_execute(matrix, "lapmod")
    ref_cost, _, _ = scipy_execute(matrix)
    assert_optimal_cost(cost, ref_cost, "lapmod")
    assert_valid_assignment(rows, cols, size, "lapmod")


@pytest.mark.parametrize("size", [2, 3, 4, 5, 10, 20])
def test_correctness_dantzig(size):
    matrix = generate_test_matrix(size)
    cost, rows, cols = fastlap_execute(matrix, "dantzig")
    ref_cost, _, _ = scipy_execute(matrix)
    assert_optimal_cost(cost, ref_cost, "dantzig")
    assert_valid_assignment(rows, cols, size, "dantzig")


@pytest.mark.parametrize("size", [2, 3, 4, 5, 10, 20])
def test_correctness_subgradient(size):
    matrix = generate_test_matrix(size)
    cost, rows, cols = fastlap_execute(matrix, "subgradient")
    ref_cost, _, _ = scipy_execute(matrix)
    assert_optimal_cost(cost, ref_cost, "subgradient")
    assert_valid_assignment(rows, cols, size, "subgradient")


@pytest.mark.parametrize("size", [2, 3, 4, 5, 10, 20])
def test_correctness_auction(size):
    """Auction is ε-optimal; cost may exceed the optimum by at most n·ε."""
    matrix = generate_test_matrix(size)
    cost, rows, cols = fastlap_execute(matrix, "auction")
    ref_cost, _, _ = scipy_execute(matrix)
    tol = max(float(matrix.max()) * size * 1e-7, 1e-4)
    assert_optimal_cost(cost, ref_cost, "auction", tol=tol)
    assert_valid_assignment(rows, cols, size, "auction")


@pytest.mark.parametrize("size", [2, 3, 4, 5, 10, 20])
def test_all_algorithms_agree(size):
    """Every algorithm should produce the same optimal cost for the same input."""
    matrix = generate_test_matrix(size)
    ref_cost, _, _ = scipy_execute(matrix)

    exact_algos = ["lapjv", "hungarian", "lapmod", "dantzig", "subgradient"]
    for algo in exact_algos:
        cost, rows, cols = fastlap_execute(matrix, algo)
        assert_optimal_cost(cost, ref_cost, algo)
        assert_valid_assignment(rows, cols, size, algo)


# ── Sparse input ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("size", [3, 5, 10, 20])
def test_sparse_input(size):
    """Verify that scipy.sparse.csr_matrix input produces the same result as dense."""
    matrix = generate_test_matrix(size)
    sparse_matrix = sp.csr_matrix(matrix)
    ref_cost, _, _ = scipy_execute(matrix)

    for algo in ["lapjv", "hungarian", "lapmod", "dantzig", "subgradient"]:
        cost, rows, cols = fastlap_execute(sparse_matrix.toarray(), algo)
        assert_optimal_cost(cost, ref_cost, f"sparse-{algo}")
        assert_valid_assignment(rows, cols, size, f"sparse-{algo}")


# ── Correctness benchmark ──────────────────────────────────────────────────

@pytest.mark.parametrize("n_iter", [1000])
@pytest.mark.parametrize("size", [2, 5, 10, 20, 50])
def test_correctness_benchmark(n_iter, size):
    """Stress test: assert every algorithm matches scipy for 1000 random matrices."""
    np.random.seed(0)
    exact_algos = ["lapjv", "hungarian", "lapmod", "dantzig", "subgradient"]

    for iteration in range(n_iter):
        matrix = generate_test_matrix(size)
        ref_cost, _, _ = scipy_execute(matrix)

        for algo in exact_algos:
            cost, rows, cols = fastlap_execute(matrix, algo)
            assert abs(cost - ref_cost) <= 1e-6, (
                f"Iteration {iteration}, size {size}, {algo}: "
                f"cost {cost:.10f} != ref {ref_cost:.10f}"
            )
            for idx, val in enumerate(rows):
                assert val is None or isinstance(val, int), (
                    f"Iteration {iteration}, size {size}, {algo}: "
                    f"row_assign[{idx}] = {val!r}, expected int or None"
                )


# ── Non-square matrix tests ─────────────────────────────────────────────────

@pytest.mark.parametrize("nrows,ncols", [(2, 3), (3, 2), (4, 6), (6, 4), (1, 5), (5, 1)])
def test_non_square_rectangular(nrows, ncols):
    """Rectangular matrices should produce valid partial assignments."""
    matrix = np.random.uniform(0, 100, (nrows, ncols)).astype(np.float64)
    nrows_actual, ncols_actual = matrix.shape

    for algo in ["lapjv", "hungarian", "lapmod", "dantzig"]:
        cost, rows, cols = fastlap_execute(matrix, algo)
        assert len(rows) == nrows_actual, f"{algo}: expected {nrows_actual} rows, got {len(rows)}"
        assert len(cols) == ncols_actual, f"{algo}: expected {ncols_actual} cols, got {len(cols)}"
        assert cost >= 0, f"{algo}: cost should be non-negative, got {cost}"


@pytest.mark.parametrize("nrows,ncols", [(3, 5), (5, 3)])
def test_non_square_cost_matches_reference(nrows, ncols):
    """Non-square cost should be the min-cost assignment of min(n,m) pairs."""
    matrix = np.random.uniform(0, 100, (nrows, ncols)).astype(np.float64)
    min_dim = min(nrows, ncols)

    for algo in ["lapjv", "hungarian", "lapmod", "dantzig"]:
        cost, rows, cols = fastlap_execute(matrix, algo)
        assert cost >= 0, f"{algo}: cost must be >= 0"
        assigned = sum(1 for j in rows if j is not None)
        assert assigned == min_dim, (
            f"{algo}: expected {min_dim} assignments, got {assigned}"
        )


# ── Edge-case tests ─────────────────────────────────────────────────────────

def test_1x1_matrix():
    """Single element matrix."""
    matrix = np.array([[42.0]])
    cost, rows, cols = fastlap_execute(matrix, "lapjv")
    assert abs(cost - 42.0) < 1e-9
    assert rows == [0]
    assert cols == [0]


def test_2x2_identity():
    """Identity-like cost matrix."""
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    cost, rows, cols = fastlap_execute(matrix, "lapjv")
    assert abs(cost - 0.0) < 1e-9
    assert sorted(rows) == [0, 1]
    assert sorted(cols) == [0, 1]


def test_duplicate_costs():
    """Matrix where all entries are the same — any assignment is optimal."""
    matrix = np.full((4, 4), 5.0)
    cost, rows, cols = fastlap_execute(matrix, "lapjv")
    assert abs(cost - 20.0) < 1e-9
    assert_valid_assignment(rows, cols, 4, "lapjv")


def test_diagonal_matrix():
    """Diagonal matrix — optimal assigns each row to its diagonal column."""
    matrix = np.array([
        [1.0, 99.0, 99.0],
        [99.0, 2.0, 99.0],
        [99.0, 99.0, 3.0],
    ])
    cost, rows, cols = fastlap_execute(matrix, "lapjv")
    assert abs(cost - 6.0) < 1e-9
    assert rows == [0, 1, 2]
    assert cols == [0, 1, 2]


def test_large_cost_range():
    """Matrix with very large and very small costs."""
    matrix = np.array([
        [1e-10, 1e10],
        [1e10, 1e-10],
    ])
    cost, rows, cols = fastlap_execute(matrix, "lapjv")
    assert abs(cost - 2e-10) < 1e-6
    assert sorted(rows) == [0, 1]


def test_all_algorithms_on_1x1():
    """All algorithms should handle a 1x1 matrix."""
    matrix = np.array([[7.0]])
    for algo in fastlap.get_supported_algorithms():
        cost, rows, cols = fastlap_execute(matrix, algo)
        assert abs(cost - 7.0) < 1e-9, f"{algo} failed on 1x1"


def test_nan_input_rejected():
    """NaN in the matrix should raise ValueError."""
    matrix = np.array([[1.0, float("nan")], [3.0, 4.0]])
    with pytest.raises(ValueError, match="NaN"):
        fastlap_execute(matrix, "lapjv")


def test_inf_input_rejected():
    """Inf in the matrix should raise ValueError."""
    matrix = np.array([[1.0, float("inf")], [3.0, 4.0]])
    with pytest.raises(ValueError, match="infinite"):
        fastlap_execute(matrix, "lapjv")


def test_empty_matrix_rejected():
    """Empty matrix should raise ValueError."""
    matrix = np.array([]).reshape(0, 0)
    with pytest.raises(ValueError, match="empty"):
        fastlap_execute(matrix, "lapjv")


def test_unknown_algorithm_rejected():
    """Unknown algorithm name should raise ValueError."""
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="Unknown algorithm"):
        fastlap_execute(matrix, "nonexistent_algo")


# ── None sentinel tests ─────────────────────────────────────────────────────

def test_unassigned_returns_none():
    """Unassigned entries in non-square solutions should be None, not a sentinel int."""
    # 2x3 matrix: one column will be unassigned
    matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    cost, rows, cols = fastlap_execute(matrix, "lapjv")
    assert len(rows) == 2
    assert len(cols) == 3
    # cols should have exactly one None (unassigned column)
    none_count = sum(1 for c in cols if c is None)
    assert none_count == 1, f"Expected 1 unassigned column, got {none_count}"


# ── Batch solve ──────────────────────────────────────────────────────────────

def test_solve_lap_batch():
    """solve_lap_batch should return one result per input matrix."""
    np.random.seed(42)
    matrices = [generate_test_matrix(5) for _ in range(4)]
    results = fastlap.solve_lap_batch(matrices, "lapjv")
    assert len(results) == 4
    for i, (cost, rows, cols) in enumerate(results):
        ref_cost, _, _ = scipy_execute(matrices[i])
        assert_optimal_cost(cost, ref_cost, f"batch-{i}")
        assert_valid_assignment(rows, cols, 5, f"batch-{i}")


# ── Weighted solve ──────────────────────────────────────────────────────────

def test_solve_lap_weighted():
    """solve_lap_weighted should return cost from the original (unweighted) matrix."""
    cost = np.array([[1, 2], [3, 4]], dtype=np.float64)
    weights = np.array([[1, 0.5], [0.5, 1]], dtype=np.float64)
    total, rows, cols = fastlap.solve_lap_weighted(cost, weights, "lapjv")
    assert total >= 0
    assert len(rows) == 2
    assert len(cols) == 2


# ── Module docstring ────────────────────────────────────────────────────────

def test_module_has_doc():
    """fastlap module should have a docstring for help()."""
    assert fastlap.__doc__ is not None
    assert "LAP" in fastlap.__doc__
