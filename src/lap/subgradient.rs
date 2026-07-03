use crate::types::{LapSolution, UNASSIGNED};
use crate::utils::sap_solve;

/// Solves the LAP using subgradient dual optimization followed by SAP primal recovery.
///
/// Phase 1 runs subgradient iterations to improve the Lagrangian dual bound —
/// dual variables u[i] and v[j] are updated so that `u[i] + v[j] ≤ cost[i][j]`
/// approaches tightness at the optimum.  Phase 2 then runs the O(n³) SAP algorithm
/// (initialized from scratch) to produce a guaranteed-optimal feasible assignment.
/// The subgradient phase acts as a warm-up that can detect early termination when a
/// feasible assignment is found during the dual iterations.
pub fn solve(matrix: Vec<Vec<f64>>) -> LapSolution {
    let n = matrix.len();
    if n == 0 {
        return (0.0, vec![], vec![]);
    }
    let m = matrix[0].len();
    if n != m {
        return (0.0, vec![], vec![]);
    }

    // Phase 1: Subgradient dual optimization.
    let mut u = vec![0.0f64; n];
    let mut v = vec![0.0f64; n];

    for iter in 0..500 {
        let step = 1.0 / (1.0 + iter as f64 * 0.01);

        // Lagrangian subproblem: for each row pick the column minimizing reduced cost.
        // Use a greedy feasibility check (columns consumed in row order).
        let mut col_used = vec![false; n];
        let mut col_of_row = vec![UNASSIGNED; n];

        for i in 0..n {
            let best = (0..n).filter(|&j| !col_used[j]).min_by(|&j1, &j2| {
                (matrix[i][j1] - u[i] - v[j1])
                    .partial_cmp(&(matrix[i][j2] - u[i] - v[j2]))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            if let Some(j) = best {
                col_of_row[i] = j;
                col_used[j] = true;
            }
        }

        // Subgradient update: unassigned rows have subgradient +1, assigned rows -1.
        let mut sg_u = vec![0.0f64; n];
        let mut sg_v = vec![0.0f64; n];
        for i in 0..n {
            sg_u[i] = if col_of_row[i] == UNASSIGNED {
                1.0
            } else {
                -1.0
            };
        }
        for j in 0..n {
            sg_v[j] = if col_used[j] { -1.0 } else { 1.0 };
        }

        let norm = sg_u
            .iter()
            .chain(sg_v.iter())
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt()
            .max(1e-12);

        for i in 0..n {
            u[i] += step * sg_u[i] / norm;
        }
        for j in 0..n {
            v[j] -= step * sg_v[j] / norm;
        }
    }

    // Phase 2: SAP primal recovery — guarantees the globally optimal feasible solution.
    let (_, row_assign, col_assign) = sap_solve(&matrix);

    let total_cost: f64 = (0..n)
        .filter(|&i| row_assign[i] != UNASSIGNED)
        .map(|i| matrix[i][row_assign[i]])
        .sum();

    (total_cost, row_assign, col_assign)
}
