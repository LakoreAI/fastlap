import pytest
import numpy as np
import scipy.sparse as sp

from conftest import generate_test_matrix, fastlap_execute, scipy_execute, lap_execute


def assert_optimal_cost(fastlap_cost, ref_cost, algo, tol=1e-6):
    assert abs(fastlap_cost - ref_cost) <= tol, (
        f"{algo}: cost {fastlap_cost:.10f} differs from reference {ref_cost:.10f} "
        f"by {abs(fastlap_cost - ref_cost):.2e} (tol={tol})"
    )


def assert_valid_assignment(row_assign, col_assign, n, algo):
    assert len(row_assign) == n, f"{algo}: row_assign length {len(row_assign)} != {n}"
    assert len(col_assign) == n, f"{algo}: col_assign length {len(col_assign)} != {n}"
    assert sorted(row_assign) == list(range(n)), f"{algo}: row_assign is not a permutation"
    assert sorted(col_assign) == list(range(n)), f"{algo}: col_assign is not a permutation"


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


ALL_ALGORITHMS = ["lapjv", "hungarian", "lapmod", "dantzig", "subgradient", "auction"]


@pytest.mark.parametrize("size", [3, 4, 5, 10, 20])
@pytest.mark.parametrize("algo", ALL_ALGORITHMS)
def test_sparse_matches_dense(algo, size):
    """A scipy.sparse.csr_matrix input must yield the same result as its dense form."""
    matrix = generate_test_matrix(size)
    csr = sp.csr_matrix(matrix)

    dense_cost, dense_rows, dense_cols = fastlap_execute(matrix, algo)
    sparse_cost, sparse_rows, sparse_cols = fastlap_execute(csr, algo)

    assert abs(dense_cost - sparse_cost) < 1e-9, (
        f"{algo} size={size}: sparse cost {sparse_cost} != dense cost {dense_cost}"
    )
    assert sparse_rows == dense_rows, f"{algo} size={size}: sparse rows differ from dense"
    assert sparse_cols == dense_cols, f"{algo} size={size}: sparse cols differ from dense"


def test_sparse_implicit_zeros_match_toarray():
    """Unstored CSR entries are treated as 0.0, matching csr.toarray()."""
    matrix = np.array(
        [[5.0, 0.0, 3.0], [0.0, 2.0, 0.0], [4.0, 0.0, 1.0]], dtype=np.float64
    )
    csr = sp.csr_matrix(matrix)
    for algo in ALL_ALGORITHMS:
        sparse_cost, _, _ = fastlap_execute(csr, algo)
        dense_cost, _, _ = fastlap_execute(csr.toarray(), algo)
        assert abs(sparse_cost - dense_cost) < 1e-9, (
            f"{algo}: sparse {sparse_cost} != toarray {dense_cost}"
        )
