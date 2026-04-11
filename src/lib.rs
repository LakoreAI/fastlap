#![allow(clippy::needless_range_loop)]
use pyo3::prelude::*;

pub mod lap;
pub mod matrix;
pub mod types;
pub mod utils;

use crate::lap::{auction, dantzig, hungarian, lapjv, lapmod, subgradient};
use crate::matrix::extract_matrix;
use crate::types::LapSolution;
use crate::utils::supported_algorithms;

#[pyfunction]
fn solve_lap<'py>(
    _py: Python<'py>,
    cost_matrix: &Bound<'py, PyAny>,
    algorithm: &str,
) -> PyResult<LapSolution> {
    let matrix = extract_matrix(cost_matrix)?;

    match algorithm {
        "lapjv" => Ok(lapjv::solve(matrix)),
        "hungarian" => Ok(hungarian::solve(matrix)),
        "lapmod" => Ok(lapmod::solve(matrix)),
        "subgradient" => Ok(subgradient::solve(matrix)),
        "auction" => Ok(auction::solve(matrix)),
        "dantzig" => Ok(dantzig::solve(matrix)),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown algorithm. Supported algorithms: {}",
            supported_algorithms().join(", ")
        ))),
    }
}

#[pyfunction]
fn get_supported_algorithms() -> Vec<&'static str> {
    supported_algorithms().to_vec()
}

#[pymodule]
fn fastlap(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_lap, m)?)?;
    m.add_function(wrap_pyfunction!(get_supported_algorithms, m)?)?;
    Ok(())
}
