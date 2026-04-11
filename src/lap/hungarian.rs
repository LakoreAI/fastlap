use crate::types::{LapSolution, UNASSIGNED};

/// Solves the Linear Assignment Problem using Hungarian algorithm (Kuhn-Munkres).
/// Note: This implementation uses a simplified augmenting path approach that may
/// fall into local minima / infinite loops if not bounded. Bounded by max_iterations.
pub fn solve(matrix: Vec<Vec<f64>>) -> LapSolution {
    let n = matrix.len();
    if n == 0 {
        return (0.0, vec![], vec![]);
    }
    let m = matrix[0].len();
    let mut cost = matrix.clone();

    // Row reduction
    for i in 0..n {
        let min_val = cost[i].iter().cloned().fold(f64::INFINITY, f64::min);
        for j in 0..m {
            cost[i][j] -= min_val;
        }
    }

    // Column reduction
    for j in 0..m {
        let min_val = (0..n).map(|i| cost[i][j]).fold(f64::INFINITY, f64::min);
        for i in 0..n {
            cost[i][j] -= min_val;
        }
    }

    // Cover zeros
    let mut row_covered = vec![false; n];
    let mut col_covered = vec![false; m];
    let mut row_assign = vec![UNASSIGNED; n];
    let mut col_assign = vec![UNASSIGNED; m];

    // Initial assignment
    for i in 0..n {
        for j in 0..m {
            if cost[i][j] == 0.0 && !row_covered[i] && !col_covered[j] {
                row_assign[i] = j;
                col_assign[j] = i;
                row_covered[i] = true;
                col_covered[j] = true;
                break;
            }
        }
    }

    let mut max_iters = n * n * n; // Prevent infinite loop in worst-case
    // Iterative augmentation
    while row_covered.iter().any(|&x| !x) && max_iters > 0 {
        max_iters -= 1;
        let mut zeros = vec![];
        for i in 0..n {
            if !row_covered[i] {
                for j in 0..m {
                    if cost[i][j] == 0.0 && !col_covered[j] {
                        zeros.push((i, j));
                    }
                }
            }
        }

        if zeros.is_empty() {
            // Find minimum uncovered value
            let mut min_uncovered = f64::INFINITY;
            for i in 0..n {
                if !row_covered[i] {
                    for j in 0..m {
                        if !col_covered[j] {
                            min_uncovered = min_uncovered.min(cost[i][j]);
                        }
                    }
                }
            }

            // Adjust matrix
            for i in 0..n {
                for j in 0..m {
                    if row_covered[i] && col_covered[j] {
                        cost[i][j] += min_uncovered;
                    } else if !row_covered[i] && !col_covered[j] {
                        cost[i][j] -= min_uncovered;
                    }
                }
            }

            // Retry assignment
            for i in 0..n {
                if !row_covered[i] {
                    for j in 0..m {
                        if cost[i][j] == 0.0 && !col_covered[j] {
                            row_assign[i] = j;
                            col_assign[j] = i;
                            row_covered[i] = true;
                            col_covered[j] = true;
                            break;
                        }
                    }
                }
            }
        } else {
            // Augment path (simplified)
            if let Some(&(i, j)) = zeros.first() {
                row_assign[i] = j;
                col_assign[j] = i;
                row_covered[i] = true;
                col_covered[j] = true;
            }
        }
    }

    let total_cost: f64 = row_assign
        .iter()
        .enumerate()
        .filter(|(_, &j)| j != UNASSIGNED)
        .map(|(i, &j)| matrix[i][j])
        .sum();

    (total_cost, row_assign, col_assign)
}
