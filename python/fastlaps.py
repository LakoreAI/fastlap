import time
import fastlap
import numpy as np
from scipy.optimize import linear_sum_assignment
import lap


if __name__ == "__main__":
    # Quick example for a 3x3 matrix
    matrix = np.random.rand(4, 5)
    for algo in ["lapjv", "hungarian"]:
        print(f"\nAlgorithm: {algo}")
        start = time.time()
        fastlap_cost, fastlap_rows, fastlap_cols = fastlap.solve_lap(matrix, algo)
        fastlap_end = time.time()
        print(f"fastlap.{algo}: Time={fastlap_end - start:.8f}s")
        print(
            f"fastlap.{algo}: Cost={fastlap_cost}, Rows={fastlap_rows}, Cols={fastlap_cols}"
        )
        if algo == "hungarian":
            start = time.time()
            scipy_rows, scipy_cols = linear_sum_assignment(matrix)
            scipy_cost = matrix[scipy_rows, scipy_cols].sum()
            scipy_end = time.time()
            print(
                f"scipy.optimize.linear_sum_assignment: Time={scipy_end - start:.8f}s"
            )
            print(
                f"scipy.optimize.linear_sum_assignment: Cost={scipy_cost}, Rows={list(scipy_rows)}, Cols={list(scipy_cols)}"
            )
        if algo == "lapjv":
            start = time.time()
            lap_cost, lap_rows, lap_cols = lap.lapjv(matrix, extend_cost=True)
            lap_end = time.time()
            print(f"lap.lapjv: Time={lap_end - start:.8f}s")
            print(f"lap.lapjv: Cost={lap_cost}, Rows={lap_rows}, Cols={lap_cols}")

"""
First release:

Algorithm: lapjv
fastlap.lapjv: Time=0.00017548s
fastlap.lapjv: Cost=inf, Rows=[2, 0, 1, 4, 3], Cols=[1, 2, 0, 4, 3]
lap.lapjv: Time=0.00013208s
lap.lapjv: Cost=0.6229432588732741, Rows=[2 0 1 4], Cols=[ 1  2  0 -1  3]

Algorithm: hungarian
fastlap.hungarian: Time=0.00000453s
fastlap.hungarian: Cost=0.7465856501551806, Rows=[2, 0, 1, 3], Cols=[1, 2, 0, 3, 18446744073709551615]
scipy.optimize.linear_sum_assignment: Time=0.00001287s
scipy.optimize.linear_sum_assignment: Cost=0.6229432588732741, Rows=[0, 1, 2, 3], Cols=[2, 0, 1, 4]
"""
