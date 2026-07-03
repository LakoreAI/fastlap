use crate::types::{LapSolution, UNASSIGNED};
use std::collections::VecDeque;

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
        return (0.0, vec![], vec![]);
    }

    // ε scales with the cost magnitude so the optimality gap stays negligible.
    let max_cost = matrix
        .iter()
        .flatten()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let epsilon = (max_cost.abs() * 1e-9).max(1e-12);

    let mut prices = vec![0.0f64; n];
    let mut row_assign = vec![UNASSIGNED; n];
    let mut col_assign = vec![UNASSIGNED; n];
    let mut unassigned: VecDeque<usize> = (0..n).collect();

    while let Some(bidder) = unassigned.pop_front() {
        // Minimization: best item is the one with the lowest (cost + price).
        let mut best_item = 0;
        let mut best_val = f64::INFINITY;
        let mut second_best_val = f64::INFINITY;

        for item in 0..n {
            let val = matrix[bidder][item] + prices[item];
            if val < best_val {
                second_best_val = best_val;
                best_val = val;
                best_item = item;
            } else if val < second_best_val {
                second_best_val = val;
            }
        }

        // Raise the price of the best item so it becomes less attractive to others.
        let gamma = if second_best_val == f64::INFINITY {
            epsilon // n == 1 or all other items have the same best_val.
        } else {
            second_best_val - best_val + epsilon
        };
        prices[best_item] += gamma;

        // Displace the previous holder of best_item, if any.
        if col_assign[best_item] != UNASSIGNED {
            let prev = col_assign[best_item];
            unassigned.push_back(prev);
            row_assign[prev] = UNASSIGNED;
        }

        row_assign[bidder] = best_item;
        col_assign[best_item] = bidder;
    }

    let total_cost: f64 = row_assign
        .iter()
        .enumerate()
        .filter(|(_, &item)| item != UNASSIGNED)
        .map(|(bidder, &item)| matrix[bidder][item])
        .sum();

    (total_cost, row_assign, col_assign)
}
