pub fn lapjv(matrix: Vec<Vec<f64>>) -> (f64, Vec<usize>, Vec<usize>) {
    let n = matrix.len();
    if n == 0 {
        return (0.0, vec![], vec![]);
    }
    let m = matrix[0].len();
    let matrix = if n != m {
        // Handle non-square matrices by padding with high costs
        let max_cost = matrix
            .iter()
            .flatten()
            .fold(f64::INFINITY, |a, &b| a.max(b))
            + 1.0;
        let mut padded_matrix = vec![vec![max_cost; n.max(m)]; n.max(m)];
        for i in 0..n {
            for j in 0..m {
                padded_matrix[i][j] = matrix[i][j];
            }
        }
        padded_matrix
    } else {
        matrix
    };

    let n = matrix.len();
    let mut u = vec![0.0; n]; // Dual variables for rows
    let mut v = vec![0.0; n]; // Dual variables for columns
    let mut row_assign = vec![usize::MAX; n];
    let mut col_assign = vec![usize::MAX; n];

    // Greedy initialization
    for i in 0..n {
        if let Some((j_min, &min_val)) = matrix[i]
            .iter()
            .enumerate()
            .filter(|(j, _)| col_assign[*j] == usize::MAX)
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        {
            row_assign[i] = j_min;
            col_assign[j_min] = i;
            u[i] = min_val;
        }
    }

    // Augmenting path loop
    for i in 0..n {
        if row_assign[i] != usize::MAX {
            continue;
        }
        let mut min_slack = vec![f64::INFINITY; n];
        let mut prev = vec![usize::MAX; n];
        let mut visited = vec![false; n];
        let mut marked_row = i;

        #[allow(unused_assignments)]
        let mut marked_col = usize::MAX;

        loop {
            visited[marked_row] = true;
            for j in 0..n {
                if !visited[j] && col_assign[j] != usize::MAX {
                    let slack = matrix[marked_row][j] - u[marked_row] - v[j];
                    if slack < min_slack[j] {
                        min_slack[j] = slack;
                        prev[j] = marked_row;
                    }
                }
            }

            let (j, &delta) = min_slack
                .iter()
                .enumerate()
                .filter(|(j, _)| !visited[*j] && col_assign[*j] != usize::MAX)
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &f64::INFINITY));

            if delta == f64::INFINITY {
                // Find unassigned column
                let unassigned_j = (0..n).find(|&j| col_assign[j] == usize::MAX).unwrap();
                marked_col = unassigned_j;
                break;
            }

            for k in 0..n {
                if visited[k] {
                    u[k] += delta;
                    v[k] -= delta;
                } else {
                    min_slack[k] -= delta;
                }
            }

            marked_row = col_assign[j];
        }

        // Augment path
        while marked_col != usize::MAX {
            let i_prev = prev[marked_col];
            let j_prev = row_assign[i_prev];
            row_assign[i_prev] = marked_col;
            col_assign[marked_col] = i_prev;
            marked_col = j_prev;
        }
    }

    let total_cost: f64 = row_assign
        .iter()
        .enumerate()
        .filter(|(_, &j)| j != usize::MAX)
        .map(|(i, &j)| matrix[i][j])
        .sum();

    (total_cost, row_assign, col_assign)
}

pub fn hungarian(matrix: Vec<Vec<f64>>) -> (f64, Vec<usize>, Vec<usize>) {
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
    let mut row_assign = vec![usize::MAX; n];
    let mut col_assign = vec![usize::MAX; m];

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

    // Iterative augmentation
    while row_covered.iter().any(|&x| !x) {
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
            for &(i, j) in &zeros {
                row_assign[i] = j;
                col_assign[j] = i;
                row_covered[i] = true;
                col_covered[j] = true;
                break;
            }
        }
    }

    let total_cost: f64 = row_assign
        .iter()
        .enumerate()
        .filter(|(_, &j)| j != usize::MAX)
        .map(|(i, &j)| matrix[i][j])
        .sum();

    (total_cost, row_assign, col_assign)
}

pub fn lapmod(matrix: Vec<Vec<f64>>) -> (f64, Vec<usize>, Vec<usize>) {
    let n = matrix.len();
    if n == 0 {
        return (0.0, vec![], vec![]);
    }
    let m = matrix[0].len();

    // Handle non-square matrices by padding with INFINITY
    let dim = n.max(m);
    let padded_matrix = if n != m {
        let mut new_matrix = vec![vec![f64::INFINITY; dim]; dim];
        for i in 0..n {
            for j in 0..m {
                new_matrix[i][j] = matrix[i][j];
            }
        }
        new_matrix
    } else {
        matrix.clone()
    };

    let n = padded_matrix.len();
    let mut u = vec![0.0; n]; // Dual variables for rows
    let mut v = vec![0.0; n]; // Dual variables for columns
    let mut row_assign = vec![usize::MAX; n];
    let mut col_assign = vec![usize::MAX; n];

    // Greedy initialization: skip INFINITY costs
    for i in 0..n {
        if let Some((j_min, &min_val)) = padded_matrix[i]
            .iter()
            .enumerate()
            .filter(|(j, &cost)| cost != f64::INFINITY && col_assign[*j] == usize::MAX)
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        {
            row_assign[i] = j_min;
            col_assign[j_min] = i;
            u[i] = min_val;
        }
    }

    // Augmenting path loop
    for i in 0..n {
        if row_assign[i] != usize::MAX {
            continue;
        }
        let mut min_slack = vec![f64::INFINITY; n];
        let mut prev = vec![usize::MAX; n];
        let mut visited = vec![false; n];
        let mut marked_row = i;
        #[allow(unused_assignments)]
        let mut marked_col = usize::MAX;

        loop {
            visited[marked_row] = true;
            // Only consider finite costs
            for j in 0..n {
                let cost = padded_matrix[marked_row][j];
                if cost != f64::INFINITY && !visited[j] && col_assign[j] != usize::MAX {
                    let slack = cost - u[marked_row] - v[j];
                    if slack < min_slack[j] {
                        min_slack[j] = slack;
                        prev[j] = marked_row;
                    }
                }
            }

            let (j, &delta) = min_slack
                .iter()
                .enumerate()
                .filter(|(j, _)| !visited[*j] && col_assign[*j] != usize::MAX)
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &f64::INFINITY));

            if delta == f64::INFINITY {
                let unassigned_j = (0..n).find(|&j| col_assign[j] == usize::MAX).unwrap();
                marked_col = unassigned_j;
                break;
            }

            for k in 0..n {
                if visited[k] {
                    u[k] += delta;
                    v[k] -= delta;
                } else {
                    min_slack[k] -= delta;
                }
            }

            marked_row = col_assign[j];
        }

        // Augment path
        while marked_col != usize::MAX {
            let i_prev = prev[marked_col];
            let j_prev = row_assign[i_prev];
            row_assign[i_prev] = marked_col;
            col_assign[marked_col] = i_prev;
            marked_col = j_prev;
        }
    }

    // Compute total cost using original matrix
    let total_cost: f64 = row_assign
        .iter()
        .enumerate()
        .filter(|(i, &j)| j != usize::MAX && *i < n && j < m)
        .map(|(i, &j)| matrix[i][j])
        .sum();

    (total_cost, row_assign, col_assign)
}
