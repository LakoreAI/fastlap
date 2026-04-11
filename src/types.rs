/// A standard type for the result of a Linear Assignment Problem algorithm.
/// Returns a tuple containing:
/// 1. The total optimal cost (f64).
/// 2. The assignment mapping from rows to columns (`Vec<usize>`).
///    `row_assign[i]` gives the column assigned to row `i`.
/// 3. The assignment mapping from columns to rows (`Vec<usize>`).
///    `col_assign[j]` gives the row assigned to column `j`.
pub type LapSolution = (f64, Vec<usize>, Vec<usize>);

/// A generic sentinel constant indicating an unassigned row or column.
pub const UNASSIGNED: usize = usize::MAX;
