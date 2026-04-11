use crate::types::{LapSolution, UNASSIGNED};

/// Solves the Linear Assignment Problem using a subgradient optimization method.
pub fn solve(matrix: Vec<Vec<f64>>) -> LapSolution {
    let n = matrix.len();
    if n == 0 {
        return (0.0, vec![], vec![]);
    }
    let m = matrix[0].len();
    if n != m {
        return (0.0, vec![], vec![]);
    }

    let max_iterations = 1000;
    let initial_step = 1.0;
    let mut u = vec![0.0; n]; // Row dual variables
    let mut v = vec![0.0; n]; // Column dual variables
    let mut row_assign = vec![UNASSIGNED; n]; // Row to column assignments
    let mut col_assign = vec![UNASSIGNED; n]; // Column to row assignments

    for iteration in 0..max_iterations {
        // Initialize assignments and violation counts
        let mut row_assigned = vec![false; n];
        let mut col_assigned = vec![false; n];
        row_assign.fill(UNASSIGNED);
        col_assign.fill(UNASSIGNED);

        // Greedy assignment based on reduced costs
        for i in 0..n {
            let mut min_reduced_cost = f64::INFINITY;
            let mut best_j = 0;
            for j in 0..n {
                let reduced_cost = matrix[i][j] - u[i] - v[j];
                if reduced_cost < min_reduced_cost && !col_assigned[j] {
                    min_reduced_cost = reduced_cost;
                    best_j = j;
                }
            }
            if !row_assigned[i] && !col_assigned[best_j] {
                row_assign[i] = best_j;
                col_assign[best_j] = i;
                row_assigned[i] = true;
                col_assigned[best_j] = true;
            }
        }

        // Compute subgradients
        let mut subgradient_u = vec![0.0; n];
        let mut subgradient_v = vec![0.0; n];
        for i in 0..n {
            if row_assign[i] == UNASSIGNED {
                subgradient_u[i] = -1.0; // Unassigned row
            } else {
                subgradient_u[i] = 1.0; // Assigned row
            }
        }
        for j in 0..n {
            let assigned = col_assign[j] != UNASSIGNED;
            subgradient_v[j] = if assigned { -1.0 } else { 1.0 };
        }

        // Check for convergence
        let unassigned_rows = row_assign.iter().filter(|&&col| col == UNASSIGNED).count();
        if unassigned_rows == 0 {
            break;
        }

        // Update dual variables
        let norm: f64 = (subgradient_u.iter().map(|x| x * x).sum::<f64>()
            + subgradient_v.iter().map(|x| x * x).sum::<f64>())
        .sqrt();
        if norm > 0.0 {
            let s = initial_step / (1.0 + iteration as f64 * 0.01); // Diminishing step size
            for i in 0..n {
                u[i] += s * subgradient_u[i] / norm;
                v[i] += s * subgradient_v[i] / norm;
            }
        }
    }

    // Calculate total cost
    let total_cost: f64 = row_assign
        .iter()
        .enumerate()
        .filter(|(_, &j)| j != UNASSIGNED)
        .map(|(i, &j)| matrix[i][j])
        .sum();

    (total_cost, row_assign, col_assign)
}
