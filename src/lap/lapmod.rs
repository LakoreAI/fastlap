use crate::types::LapSolution;
use crate::utils::{pad_to_square, sap_solve, trim_solution};

/// Solves the LAP using LAPMOD — a LAPJV variant with infinity padding for non-square matrices.
///
/// Padding with a large-but-finite sentinel keeps the padded entries strictly
/// outside the feasible cost range, which is the approach used in sparse-aware formulations.
pub fn solve(matrix: Vec<Vec<f64>>) -> LapSolution {
    let nrows = matrix.len();
    if nrows == 0 {
        return (0.0, vec![], vec![]);
    }
    let fill = matrix
        .iter()
        .flatten()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(0.0)
        * 1e6
        + 1e9;
    let padded = pad_to_square(&matrix, fill);
    let (_, row_assign, col_assign) = sap_solve(&padded);
    trim_solution(&matrix, row_assign, col_assign)
}
