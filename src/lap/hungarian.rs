use crate::types::LapSolution;
use crate::utils::{pad_to_square, sap_solve, trim_solution};

/// Solves the LAP using the Kuhn-Munkres (Hungarian) algorithm.
///
/// Uses the O(n³) shortest-augmenting-path formulation with dual variables,
/// which is mathematically equivalent to the classical matrix-reduction method.
/// Non-square matrices are padded with a cost above the maximum real entry.
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
        + 1.0;
    let padded = pad_to_square(&matrix, fill);
    let (_, row_assign, col_assign) = sap_solve(&padded);
    trim_solution(&matrix, row_assign, col_assign)
}
