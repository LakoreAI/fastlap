
# fastlap

**Python’s LAP (Linear Assignment Problem) solver — written in Rust for performance.**

`fastlap` provides a blazing-fast implementation of common assignment algorithms such as:

- Jonker-Volgenant (LAPJV)
- Hungarian (a.k.a. Munkres)

Built with [Rust](https://www.rust-lang.org/) and exposed to Python via [PyO3](https://pyo3.rs), this library offers performance and interoperability in one package.


## ✨ Features

- ✅ Fast and memory-safe implementation in Rust
- ✅ Python bindings via PyO3
- ✅ Supports both `lapjv` and `hungarian` algorithms
- ✅ Can be used in native Rust projects or as a Python package


## 📖 Algorithms

* **LAPJV** – Efficient dual-based shortest augmenting path algorithm (Jonker & Volgenant, 1987)
* **Hungarian** – Classic method with row/column reduction and assignment phases

## Roadmap

- [ ] Release first version
- [ ] Add more algorithms
- [ ] Add more features
- [ ] Add more examples
- [ ] Add more tests
- [ ] Add more benchmarks


## 📚 References

* Jonker, R., & Volgenant, A. (1987). *A shortest augmenting path algorithm for dense and sparse linear assignment problems*. Computing, 38(4), 325–340.
* Munkres, J. (1957). *Algorithms for the Assignment and Transportation Problems*. Journal of the Society for Industrial and Applied Mathematics.


## 📃 License

MIT License © 2025

