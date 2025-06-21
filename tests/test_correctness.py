import pytest

from conftest import generate_test_matrix, fastlap_execute, scipy_execute, lap_execute


def compare_assignments(
    row_ind1, col_ind1, row_ind2, col_ind2, cost1, cost2, algo1, algo2, tol=1e-8
):
    """Compare assignments and costs between two algorithms."""
    assert abs(cost1 - cost2) < tol, (
        f"Cost mismatch: {algo1} ({cost1}) vs {algo2} ({cost2})"
    )
    mapping1 = dict(zip(row_ind1, col_ind1))
    mapping2 = dict(zip(row_ind2, col_ind2))
    assert mapping1 == mapping2, f"Assignment mismatch: {algo1} vs {algo2}"


@pytest.mark.parametrize("size", [2, 3, 4, 5])
def test_correctness_hungarian(size):
    """Test fastlap correctness against SciPy for various matrix sizes."""
    matrix = generate_test_matrix(size)

    # fastlap
    fastlap_cost, fastlap_rows, fastlap_cols = fastlap_execute(matrix, "hungarian")

    # SciPy
    scipy_cost, scipy_rows, scipy_cols = scipy_execute(matrix)

    # Compare
    compare_assignments(
        fastlap_rows,
        fastlap_cols,
        scipy_rows,
        scipy_cols,
        fastlap_cost,
        scipy_cost,
        f"fastlap.hungarian",
        "scipy",
    )


@pytest.mark.parametrize("size", [2, 3, 4, 5])
def test_correctness_lapjv(size):
    """Test fastlap correctness against SciPy for various matrix sizes."""
    matrix = generate_test_matrix(size)

    # fastlap
    fastlap_cost, fastlap_rows, fastlap_cols = fastlap_execute(matrix, "lapjv")

    # lap
    lap_cost, lap_rows, lap_cols = lap_execute(matrix)

    # Compare
    compare_assignments(
        fastlap_rows,
        fastlap_cols,
        lap_rows,
        lap_cols,
        fastlap_cost,
        lap_cost,
        f"fastlap.lapjv",
        "lap",
    )
