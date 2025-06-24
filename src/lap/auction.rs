pub fn solve(values: Vec<Vec<f64>>) -> (f64, Vec<usize>, Vec<usize>) {
    let n = values.len();
    if n == 0 {
        return (0.0, vec![], vec![]);
    }
    let m = values[0].len();
    if n != m {
        return (0.0, vec![], vec![]);
    }

    let epsilon = 0.01; // Bidding increment
    let mut prices = vec![0.0; n]; // Item prices
    let mut row_assign = vec![usize::MAX; n]; // Bidder (row) to item (column)
    let mut col_assign = vec![usize::MAX; n]; // Item (column) to bidder (row)
    let mut unassigned = (0..n).collect::<Vec<usize>>(); // Unassigned bidders

    while !unassigned.is_empty() {
        let bidder = unassigned[0];
        let mut best_item = 0;
        let mut best_value = f64::NEG_INFINITY;
        let mut second_best_value = f64::NEG_INFINITY;

        // Find best and second-best items for bidder
        for item in 0..n {
            let value = values[bidder][item] - prices[item];
            if value > best_value {
                second_best_value = best_value;
                best_value = value;
                best_item = item;
            } else if value > second_best_value {
                second_best_value = value;
            }
        }

        // Compute bid
        let bid = best_value - second_best_value + epsilon;
        prices[best_item] += bid;

        // Update assignments
        if col_assign[best_item] != usize::MAX {
            let prev_bidder = col_assign[best_item];
            unassigned.push(prev_bidder);
            row_assign[prev_bidder] = usize::MAX;
        }

        row_assign[bidder] = best_item;
        col_assign[best_item] = bidder;
        unassigned.remove(0);
    }

    // Calculate total value
    let total_value: f64 = row_assign
        .iter()
        .enumerate()
        .filter(|(_, &item)| item != usize::MAX)
        .map(|(bidder, &item)| values[bidder][item])
        .sum();

    (total_value, row_assign, col_assign)
}
