pub fn solve(matrix: Vec<Vec<f64>>) -> (f64, Vec<usize>, Vec<usize>) {
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
