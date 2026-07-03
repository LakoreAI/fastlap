use crate::types::LapSolution;
use crate::utils::{pad_to_square, sap_solve, trim_solution};

/// Solves the LAP using the Jonker-Volgenant shortest-augmenting-path algorithm (LAPJV).
///
/// Non-square matrices are padded with a cost slightly above the maximum real cost so that
/// padded assignments are never preferred over real ones.
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
