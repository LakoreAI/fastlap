use numpy::PyReadonlyArray1;
use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;

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

/// Convert a scipy.sparse.csr_matrix to a dense Vec<Vec<f64>>.
///
/// Unstored entries are filled with 0.0 so the result matches `csr.toarray()`.
/// scipy stores `indptr`/`indices` as int32 or int64 depending on matrix size,
/// so both index arrays are cast to int64 (and `data` to float64) before extraction.
pub fn extract_sparse_matrix<'py>(cost_matrix: &Bound<'py, PyAny>) -> PyResult<Vec<Vec<f64>>> {
    let (nrows, ncols): (usize, usize) = cost_matrix.getattr("shape")?.extract()?;

    let indptr_obj = cost_matrix
        .getattr("indptr")?
        .call_method1("astype", ("int64",))?;
    let indices_obj = cost_matrix
        .getattr("indices")?
        .call_method1("astype", ("int64",))?;
    let data_obj = cost_matrix
        .getattr("data")?
        .call_method1("astype", ("float64",))?;

    let indptr: PyReadonlyArray1<i64> = indptr_obj.extract()?;
    let indices: PyReadonlyArray1<i64> = indices_obj.extract()?;
    let data: PyReadonlyArray1<f64> = data_obj.extract()?;

    let indptr = indptr.as_slice()?;
    let indices = indices.as_slice()?;
    let data = data.as_slice()?;

    if indptr.len() != nrows + 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Invalid CSR matrix: indptr length must equal nrows + 1",
        ));
    }
    if indices.len() != data.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Invalid CSR matrix: indices and data must have equal length",
        ));
    }

    let mut dense = vec![vec![0.0f64; ncols]; nrows];
    for i in 0..nrows {
        let start = indptr[i] as usize;
        let end = indptr[i + 1] as usize;
        for k in start..end {
            let col = indices[k] as usize;
            if col >= ncols {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid CSR matrix: column index out of bounds",
                ));
            }
            dense[i][col] = data[k];
        }
    }

    Ok(dense)
}

/// Convert input (dense or CSR) to a validated dense matrix
pub fn extract_matrix<'py>(cost_matrix: &Bound<'py, PyAny>) -> PyResult<Vec<Vec<f64>>> {
    // Try dense first
    if let Ok(array) = cost_matrix.downcast::<PyArray2<f64>>() {
        let matrix = extract_dense_matrix(array)?;
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
