use crate::types::LapSolution;
use std::collections::VecDeque;

/// Maximum number of auction rounds before giving up (prevents infinite loops).
const MAX_ITERATIONS: usize = 1_000_000;

/// Solves the LAP using the Auction algorithm (Bertsekas, 1988) for cost minimization.
///
/// Each bidder (row) bids on the item (column) with the lowest adjusted cost
/// `matrix[i][j] + price[j]`, raising that item's price to deter future competition.
/// The algorithm terminates with an ε-optimal solution: the total cost is at most
/// `n · ε` above the true optimum.
pub fn solve(matrix: Vec<Vec<f64>>) -> LapSolution {
    let n = matrix.len();
    if n == 0 {
        return (0.0, vec![], vec![]);
    }
    let m = matrix[0].len();
    if n != m {
        // For rectangular matrices, fall back to SAP (pad + solve).
        return crate::utils::sap_solve(&crate::utils::pad_to_square(
            &matrix,
            matrix
                .iter()
                .flatten()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
                + 1.0,
        ));
    }

    // ε scales with the cost magnitude so the optimality gap stays negligible.
    let max_cost = matrix
        .iter()
        .flatten()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let epsilon = (max_cost.abs() * 1e-9).max(1e-12);

    let mut prices = vec![0.0f64; n];
    let mut row_assign = vec![None; n];
    let mut col_assign = vec![None; n];
    let mut unassigned: VecDeque<usize> = (0..n).collect();

    for _ in 0..MAX_ITERATIONS {
        let Some(bidder) = unassigned.pop_front() else {
            break;
        };

        // Minimization: best item is the one with the lowest (cost + price).
        let mut best_item = None;
        let mut best_val = f64::INFINITY;
        let mut second_best_val = f64::INFINITY;

        for item in 0..n {
            let val = matrix[bidder][item] + prices[item];
            if val < best_val {
                second_best_val = best_val;
                best_val = val;
                best_item = Some(item);
            } else if val < second_best_val {
                second_best_val = val;
            }
        }

        let best_item = match best_item {
            Some(item) => item,
            None => unreachable!("n >= 1 guarantees at least one item"),
        };

        // Raise the price of the best item so it becomes less attractive to others.
        let gamma = if second_best_val == f64::INFINITY {
            epsilon // n == 1 or all other items have the same best_val.
        } else {
            second_best_val - best_val + epsilon
        };
        prices[best_item] += gamma;

        // Displace the previous holder of best_item, if any.
        if let Some(prev) = col_assign[best_item] {
            unassigned.push_back(prev);
            row_assign[prev] = None;
        }

        row_assign[bidder] = Some(best_item);
        col_assign[best_item] = Some(bidder);
    }

    let total_cost: f64 = row_assign
        .iter()
        .enumerate()
        .filter_map(|(bidder, item)| item.map(|item| matrix[bidder][item]))
        .sum();

    (total_cost, row_assign, col_assign)
}
