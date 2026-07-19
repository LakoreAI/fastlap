#![allow(clippy::needless_range_loop)]
//! fastlap — High-performance LAP solver powered by Rust.
//!
//! Provides `solve_lap` for single matrices and `solve_lap_batch` for parallel
//! solving of many independent matrices.  All six algorithms (LAPJV, Hungarian,
//! LAPMOD, Dantzig, Auction, Subgradient) are exposed through a uniform API.

use pyo3::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub mod lap;
pub mod matrix;
pub mod types;
pub mod utils;

use crate::matrix::extract_matrix;
use crate::types::LapSolution;
use crate::utils::{solve_with, supported_algorithms};

// ---------------------------------------------------------------------------
// Python ↔ Rust conversion helpers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Public Python API
// ---------------------------------------------------------------------------

/// Solve a Linear Assignment Problem.
///
/// Parameters
/// ----------
/// cost_matrix : numpy.ndarray or scipy.sparse.csr_matrix
///     An (n x m) cost matrix. Square matrices are solved directly.
///     Rectangular matrices are padded internally and assignments to
///     padded rows/columns are reported as ``None``.
/// algorithm : str
///     One of: ``"lapjv"``, ``"hungarian"``, ``"lapmod"``,
///     ``"dantzig"``, ``"auction"``, ``"subgradient"``.
///
/// Returns
/// -------
/// tuple[float, list[int | None], list[int | None]]
///     ``(total_cost, row_assignments, col_assignments)``.
///     ``row_assign[i]`` is the column assigned to row i, or ``None``.
///     ``col_assign[j]`` is the row assigned to column j, or ``None``.
///
/// Raises
/// ------
/// ValueError
///     If the matrix is empty, non-rectangular, contains NaN/Inf,
///     or the algorithm name is not recognised.
/// TypeError
///     If the input is not a NumPy ndarray or scipy CSR matrix.
///
/// Examples
/// --------
/// >>> import fastlap
/// >>> cost = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
/// >>> total, rows, cols = fastlap.solve_lap(cost, algorithm="lapjv")
#[pyfunction]
#[pyo3(signature = (cost_matrix, algorithm))]
fn solve_lap<'py>(
    _py: Python<'py>,
    cost_matrix: &Bound<'py, PyAny>,
    algorithm: &str,
) -> PyResult<LapSolution> {
    let matrix = extract_matrix(cost_matrix)?;
    solve_with(matrix, algorithm).map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
}

/// Solve multiple independent Linear Assignment Problems in parallel.
///
/// Parameters
/// ----------
/// cost_matrices : list of numpy.ndarray or scipy.sparse.csr_matrix
///     A list of cost matrices to solve.
/// algorithm : str
///     Algorithm name (same as :func:`solve_lap`).
///
/// Returns
/// -------
/// list of tuple[float, list[int | None], list[int | None]]
///     One ``(total_cost, row_assignments, col_assignments)`` per matrix.
#[pyfunction]
#[pyo3(signature = (cost_matrices, algorithm))]
fn solve_lap_batch<'py>(
    py: Python<'py>,
    cost_matrices: &Bound<'py, PyAny>,
    algorithm: &str,
) -> PyResult<Vec<LapSolution>> {
    // Validate algorithm name once up-front.
    if !supported_algorithms().contains(&algorithm) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown algorithm '{}'. Supported: {}",
            algorithm,
            supported_algorithms().join(", ")
        )));
    }

    let matrices: Vec<Vec<Vec<f64>>> = cost_matrices
        .extract::<Vec<Bound<'py, PyAny>>>()?
        .iter()
        .map(|m| extract_matrix(m))
        .collect::<PyResult<_>>()?;

    let results: Vec<LapSolution> = py.allow_threads(|| {
        matrices
            .into_par_iter()
            .map(|matrix| solve_with(matrix, algorithm).unwrap())
            .collect()
    });

    Ok(results)
}

/// Solve a Linear Assignment Problem with optional per-entry weights.
///
/// The effective cost is ``weight[i][j] * cost_matrix[i][j]``.
///
/// Parameters
/// ----------
/// cost_matrix : numpy.ndarray or scipy.sparse.csr_matrix
///     An (n x m) cost matrix.
/// weights : numpy.ndarray or scipy.sparse.csr_matrix
///     Per-entry weights of the same shape as *cost_matrix*.
/// algorithm : str
///     Algorithm name (same as :func:`solve_lap`).
///
/// Returns
/// -------
/// tuple[float, list[int | None], list[int | None]]
///     ``(total_cost, row_assignments, col_assignments)`` where the
///     returned cost is the sum of the *original* (unweighted) costs.
#[pyfunction]
#[pyo3(signature = (cost_matrix, weights, algorithm))]
fn solve_lap_weighted<'py>(
    _py: Python<'py>,
    cost_matrix: &Bound<'py, PyAny>,
    weights: &Bound<'py, PyAny>,
    algorithm: &str,
) -> PyResult<LapSolution> {
    let costs = extract_matrix(cost_matrix)?;
    let w = extract_matrix(weights)?;

    if costs.len() != w.len() || (costs.is_empty() && w.is_empty()) || costs[0].len() != w[0].len()
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "cost_matrix and weights must have the same shape",
        ));
    }

    let weighted: Vec<Vec<f64>> = costs
        .iter()
        .zip(w.iter())
        .map(|(row_c, row_w)| {
            row_c
                .iter()
                .zip(row_w.iter())
                .map(|(c, ww)| c * ww)
                .collect()
        })
        .collect();

    let (_, row_assign, col_assign) = solve_with(weighted, algorithm)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    // Recompute cost from original unweighted matrix
    let total_cost: f64 = row_assign
        .iter()
        .enumerate()
        .filter_map(|(i, opt_j)| opt_j.map(|j| costs[i][j]))
        .sum();

    Ok((total_cost, row_assign, col_assign))
}

/// Return the list of supported algorithm names.
#[pyfunction]
fn get_supported_algorithms() -> Vec<&'static str> {
    supported_algorithms().to_vec()
}

/// High-performance LAP solver backed by Rust.
///
/// Provides:
/// - :func:`solve_lap` — solve a single assignment problem
/// - :func:`solve_lap_batch` — solve many in parallel
/// - :func:`solve_lap_weighted` — solve with per-entry cost scaling
/// - :func:`get_supported_algorithms` — list available algorithms
#[pymodule]
fn fastlap(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_lap, m)?)?;
    m.add_function(wrap_pyfunction!(solve_lap_batch, m)?)?;
    m.add_function(wrap_pyfunction!(solve_lap_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(get_supported_algorithms, m)?)?;
    Ok(())
}
