use crate::types::LapSolution;
use crate::utils::{pad_to_square, sap_solve, trim_solution};

/// Solves the LAP using Dantzig's algorithm (simplex on the assignment polytope).
///
/// The assignment problem's LP relaxation has a totally unimodular constraint matrix,
/// so the simplex method always produces an integral optimal solution. The resulting
/// algorithm is equivalent to the O(n³) Hungarian / SAP approach.
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
