use crate::lap::{auction, dantzig, hungarian, lapjv, lapmod, subgradient};
use crate::types::LapSolution;

pub fn supported_algorithms() -> &'static [&'static str] {
    &[
        "lapjv",
        "hungarian",
        "lapmod",
        "subgradient",
        "auction",
        "dantzig",
    ]
}

/// Dispatch to the named algorithm. Single source of truth for all entry points.
pub fn solve_with(matrix: Vec<Vec<f64>>, algorithm: &str) -> Result<LapSolution, String> {
    match algorithm {
        "lapjv" => Ok(lapjv::solve(matrix)),
        "hungarian" => Ok(hungarian::solve(matrix)),
        "lapmod" => Ok(lapmod::solve(matrix)),
        "subgradient" => Ok(subgradient::solve(matrix)),
        "auction" => Ok(auction::solve(matrix)),
        "dantzig" => Ok(dantzig::solve(matrix)),
        _ => Err(format!(
            "Unknown algorithm '{}'. Supported: {}",
            algorithm,
            supported_algorithms().join(", ")
        )),
    }
}

/// O(n³) shortest-augmenting-path (SAP) solver for a square n×n cost matrix.
///
/// This is the standard competitive-programming formulation of the
/// Hungarian / Jonker-Volgenant algorithm. It maintains dual variables u[i]
/// and v[j] to track reduced costs and finds augmenting paths in O(n²) per row,
/// giving O(n³) overall.
///
/// Requires: `cost` is an n×n slice with finite entries.
/// Returns: `(total_cost, row_assign, col_assign)` with all vectors of length n.
pub fn sap_solve(cost: &[Vec<f64>]) -> LapSolution {
    let n = cost.len();
    if n == 0 {
        return (0.0, vec![], vec![]);
    }

    // 1-indexed storage; p[j] = row matched to column j (0 = free column).
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; n + 1];
    let mut p = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];

        // Find the shortest augmenting path for row i.
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0;

            for j in 1..=n {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            // Shift duals by the shortest-path distance delta.
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }

            j0 = j1;
            if p[j0] == 0 {
                break; // Reached a free column; augmenting path is complete.
            }
        }

        // Flip the augmenting path.
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    // Convert 1-indexed p[] into 0-indexed row_assign / col_assign.
    let mut row_assign = vec![None; n];
    let mut col_assign = vec![None; n];
    for j in 1..=n {
        if p[j] != 0 {
            row_assign[p[j] - 1] = Some(j - 1);
            col_assign[j - 1] = Some(p[j] - 1);
        }
    }

    let total_cost: f64 = (0..n)
        .filter_map(|i| row_assign[i].map(|j| cost[i][j]))
        .sum();

    (total_cost, row_assign, col_assign)
}

/// Pad a (possibly non-square) cost matrix to dim×dim, filling added entries with `fill`.
pub fn pad_to_square(matrix: &[Vec<f64>], fill: f64) -> Vec<Vec<f64>> {
    let nrows = matrix.len();
    let ncols = if nrows > 0 { matrix[0].len() } else { 0 };
    let dim = nrows.max(ncols);
    if nrows == ncols {
        return matrix.to_vec();
    }
    let mut padded = vec![vec![fill; dim]; dim];
    for (i, row) in matrix.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            padded[i][j] = val;
        }
    }
    padded
}

/// Trim a SAP solution back to the original (nrows × ncols) dimensions.
///
/// Assignments that went to padded rows/columns are mapped to None.
/// The returned cost is recomputed from the original matrix.
pub fn trim_solution(
    matrix: &[Vec<f64>],
    row_assign: Vec<Option<usize>>,
    col_assign: Vec<Option<usize>>,
) -> LapSolution {
    let nrows = matrix.len();
    let ncols = if nrows > 0 { matrix[0].len() } else { 0 };

    let trimmed_row: Vec<Option<usize>> = (0..nrows)
        .map(|i| row_assign[i].filter(|&j| j < ncols))
        .collect();

    let trimmed_col: Vec<Option<usize>> = (0..ncols)
        .map(|j| col_assign[j].filter(|&i| i < nrows))
        .collect();

    let total_cost: f64 = (0..nrows)
        .filter_map(|i| trimmed_row[i].map(|j| matrix[i][j]))
        .sum();

    (total_cost, trimmed_row, trimmed_col)
}
