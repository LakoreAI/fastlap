use numpy::PyReadonlyArray1;
use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use sprs::CsMat;

/// Convert a dense NumPy array to Vec<Vec<f64>>
pub fn extract_dense_matrix<'py>(
    cost_matrix: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Vec<Vec<f64>>> {
    let matrix: Vec<Vec<f64>> = cost_matrix
        .readonly()
        .as_array()
        .rows()
        .into_iter()
        .map(|row| row.iter().copied().collect::<Vec<f64>>())
        .collect();
    Ok(matrix)
}

/// Convert a scipy.sparse.csr_matrix to Vec<Vec<f64>>
pub fn extract_sparse_matrix<'py>(cost_matrix: &Bound<'py, PyAny>) -> PyResult<Vec<Vec<f64>>> {
    let indptr: PyReadonlyArray1<usize> = cost_matrix.getattr("indptr")?.extract()?;
    let indices: PyReadonlyArray1<usize> = cost_matrix.getattr("indices")?.extract()?;
    let data: PyReadonlyArray1<f64> = cost_matrix.getattr("data")?.extract()?;

    let shape: (usize, usize) = cost_matrix.getattr("shape")?.extract::<(usize, usize)>()?;

    let csr = CsMat::new(
        shape,
        indptr.as_slice()?.to_vec(),
        indices.as_slice()?.to_vec(),
        data.as_slice()?.to_vec(),
    );

    let dense: Vec<Vec<f64>> = (0..shape.0)
        .map(|i| {
            (0..shape.1)
                .map(|j| csr.get(i, j).copied().unwrap_or(f64::INFINITY))
                .collect()
        })
        .collect();

    Ok(dense)
}

/// Convert input (dense or CSR) to a validated dense matrix
pub fn extract_matrix<'py>(cost_matrix: &Bound<'py, PyArray2<f64>>) -> PyResult<Vec<Vec<f64>>> {
    // Try dense first
    if let Ok(array) = cost_matrix.downcast::<PyArray2<f64>>() {
        let matrix = extract_dense_matrix(&array.readonly())?;
        return validate_matrix(matrix);
    }

    // Try sparse (CSR)
    let is_csr = ["indptr", "indices", "data", "shape"]
        .iter()
        .all(|&attr| cost_matrix.hasattr(attr).unwrap_or(false));

    if is_csr {
        let matrix = extract_sparse_matrix(cost_matrix)?;
        return validate_matrix(matrix);
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Input must be a NumPy ndarray or scipy.sparse.csr_matrix",
    ))
}

/// Ensure matrix is rectangular and non-empty
pub fn validate_matrix(matrix: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    if matrix.is_empty() || matrix.iter().any(|row| row.len() != matrix[0].len()) {
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Matrix must be non-empty and rectangular",
        ))
    } else {
        Ok(matrix)
    }
}
