pub fn solve(cost: Vec<Vec<f64>>) -> (f64, Vec<usize>, Vec<usize>) {
    let n = cost.len();
    if n == 0 {
        return (0.0, vec![], vec![]);
    }
    let m = cost[0].len();
    if n != m {
        return (0.0, vec![], vec![]);
    }

    // Initialize reduced cost matrix
    let mut reduced_cost = cost.clone();
    let mut row_assign = vec![usize::MAX; n]; // Row to column assignments
    let mut col_assign = vec![usize::MAX; n]; // Column to row assignments
    let mut row_covered = vec![false; n];
    let mut col_covered = vec![false; n];

    // Row reduction
    for i in 0..n {
        let min_val = reduced_cost[i]
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        for j in 0..m {
            reduced_cost[i][j] -= min_val;
        }
    }

    // Column reduction
    for j in 0..m {
        let min_val = (0..n)
            .map(|i| reduced_cost[i][j])
            .fold(f64::INFINITY, f64::min);
        for i in 0..n {
            reduced_cost[i][j] -= min_val;
        }
    }

    // Initial feasible solution (greedy assignment)
    for i in 0..n {
        for j in 0..m {
            if reduced_cost[i][j] == 0.0 && !row_covered[i] && !col_covered[j] {
                row_assign[i] = j;
                col_assign[j] = i;
                row_covered[i] = true;
                col_covered[j] = true;
                break;
            }
        }
    }

    // Simplex-like iteration
    let max_iterations = n * n;
    for _ in 0..max_iterations {
        // Check if optimal (all rows assigned)
        if row_assign.iter().all(|&j| j != usize::MAX) {
            break;
        }

        // Find an unassigned row and compute reduced costs
        let mut pivot_row = 0;
        let mut found_unassigned = false;
        for i in 0..n {
            if row_assign[i] == usize::MAX {
                pivot_row = i;
                found_unassigned = true;
                break;
            }
        }

        if !found_unassigned {
            break;
        }

        // Find pivot column (minimum reduced cost)
        let mut min_reduced_cost = f64::INFINITY;
        let mut pivot_col = 0;
        for j in 0..n {
            if !col_covered[j] && reduced_cost[pivot_row][j] < min_reduced_cost {
                min_reduced_cost = reduced_cost[pivot_row][j];
                pivot_col = j;
            }
        }

        // Update assignments
        if col_assign[pivot_col] != usize::MAX {
            let prev_row = col_assign[pivot_col];
            row_assign[prev_row] = usize::MAX;
            row_covered[prev_row] = false;
        }
        row_assign[pivot_row] = pivot_col;
        col_assign[pivot_col] = pivot_row;
        row_covered[pivot_row] = true;
        col_covered[pivot_col] = true;

        // Update reduced cost matrix
        let mut min_uncovered = f64::INFINITY;
        for i in 0..n {
            if !row_covered[i] {
                for j in 0..n {
                    if !col_covered[j] {
                        min_uncovered = min_uncovered.min(reduced_cost[i][j]);
                    }
                }
            }
        }

        if min_uncovered.is_finite() {
            for i in 0..n {
                for j in 0..n {
                    if row_covered[i] && col_covered[j] {
                        reduced_cost[i][j] += min_uncovered;
                    } else if !row_covered[i] && !col_covered[j] {
                        reduced_cost[i][j] -= min_uncovered;
                    }
                }
            }
        }
    }

    // Calculate total cost
    let total_cost: f64 = row_assign
        .iter()
        .enumerate()
        .filter(|(_, &j)| j != usize::MAX)
        .map(|(i, &j)| cost[i][j])
        .sum();

    (total_cost, row_assign, col_assign)
}
