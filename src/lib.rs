use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;

pub mod lap;
use crate::lap::*;

#[pyfunction]
fn solve_lap<'py>(
    _py: Python<'py>,
    cost_matrix: &Bound<'py, PyArray2<f64>>, // Changed to use Bound
    algorithm: &str,
) -> PyResult<(f64, Vec<usize>, Vec<usize>)> {
    // Convert NumPy array to dense matrix
    let matrix: Vec<Vec<f64>> = cost_matrix
        .readonly()
        .as_array()
        .rows()
        .into_iter()
        .map(|row| row.iter().copied().collect::<Vec<f64>>())
        .collect();

    // Validate dense matrix
    if matrix.is_empty() || matrix.iter().any(|row| row.len() != matrix[0].len()) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Matrix must be non-empty and rectangular",
        ));
    }

    match algorithm {
        "lapjv" => Ok(lapjv(matrix)),
        "hungarian" => Ok(hungarian(matrix)),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Unknown algorithm. Supported algorithms: 'lapjv', 'hungarian'",
        )),
    }
}

#[pymodule]
fn fastlap(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_lap, m)?)?;
    Ok(())
}

pub fn fastlap_rust(_py: Python, matrix: &Vec<Vec<f64>>, algo: &str) -> PyResult<(f64, Vec<usize>, Vec<usize>)> {
    if matrix.is_empty() || matrix.iter().any(|row| row.len() != matrix[0].len()) {
        return Err(pyo3::exceptions::PyValueError::new_err("Matrix must be square and non-empty"));
    }

    let result = match algo {
        "lapjv" => lapjv(matrix.clone()),
        "hungarian" => hungarian(matrix.clone()),
        _ => return Err(pyo3::exceptions::PyValueError::new_err("Unsupported algorithm")),
    };

    Ok(result)
}

