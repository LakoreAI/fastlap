/// A standard type for the result of a Linear Assignment Problem algorithm.
/// Returns a tuple containing:
/// 1. The total optimal cost (f64).
/// 2. The assignment mapping from rows to columns (`Vec<Option<usize>>`).
///    `row_assign[i]` gives the column assigned to row `i`, or `None` if unassigned.
/// 3. The assignment mapping from columns to rows (`Vec<Option<usize>>`).
///    `col_assign[j]` gives the row assigned to column `j`, or `None` if unassigned.
pub type LapSolution = (f64, Vec<Option<usize>>, Vec<Option<usize>>);
