# **fastlap**

**Python’s LAP (Linear Assignment Problem) solver — written in Rust for performance.**

`fastlap` delivers a blazing-fast implementation of popular assignment algorithms, including:

* **Jonker–Volgenant (LAPJV)**
* **Hungarian (a.k.a. Munkres)**
* **LAPMOD**

Built with [Rust](https://www.rust-lang.org/) and exposed to Python via [PyO3](https://pyo3.rs), `fastlap` combines performance and interoperability in a single lightweight package.


## 📖 Algorithms

* **LAPJV** — Efficient dual-based shortest augmenting path algorithm
  *(Jonker & Volgenant, 1987)*
* **Hungarian Algorithm** — Classic method using row/column reduction and assignment phases
* **LAPMOD** — A modified variant for better performance under specific conditions


## 🚀 Usage

```python
import fastlap

# Example cost matrix
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Solve the LAP using LAPJV algorithm
cost, row_assign, col_assign = fastlap.solve_lap(matrix, method="lapjv")

print("Total cost:", cost)
print("Row assignments:", row_assign)
print("Column assignments:", col_assign)
```


## 📄 Citation

If you use `fastlap` in your research or project, please cite the following:

```
@misc{fastlap2025,
  author       = {Le Duc Minh},
  title        = {fastlap: A Python LAP solver powered by Rust},
  year         = {2025},
  howpublished = {\url{https://github.com/8Opt/fastlap}},
  note         = {Python-Rust LAP solver implementing LAPJV, Hungarian, and LAPMOD}
}
```


## 📃 License

**MIT License** © 2025 — use it freely in commercial or open-source projects.
