
<div style="text-align: center;">
  <img src="https://raw.githubusercontent.com/8Opt/fastlap/main/docs/static/fastlap.png" alt="fastlap logo" width="50%"/>
</div>


<div style="text-align: center;">
<h2>fastlap<br>High-Performance Linear Assignment Problem Solver</h2>
</div>

<div style="text-align: center;">
<img src="https://img.shields.io/badge/in%20progress-8A2BE2" alt="in progress">
<img src="https://img.shields.io/badge/Python-3.9–3.13-blue" alt="Python 3.9–3.13">  
<img src="https://img.shields.io/badge/Rust-1.80.0-blue" alt="Rust 1.80.0">  
<img src="https://img.shields.io/pypi/v/fastlap" alt="PyPI version">  
</div>


fastlap is a high-performance Python library for solving Linear Assignment Problems (LAP), implemented in Rust for optimal speed and efficiency. Leveraging the PyO3 framework, fastlap seamlessly integrates Rust's performance with Python's ease of use, delivering a lightweight and robust solution for assignment optimization tasks.

## ✨ Features

- High Performance: Built in Rust for superior computational speed.
- Multiple Algorithms: Supports state-of-the-art LAP algorithms, including LAPJV, Hungarian, LAPMOD, Dantzig’s, Auction, and Subgradient.
- Python Integration: User-friendly Python interface via PyO3.
- Lightweight: Minimal dependencies for easy integration into projects.

## 📖 Supported Algorithms

- LAPJV — Efficient dual-based shortest augmenting path algorithm (Jonker & Volgenant, 1987)
- Hungarian Algorithm — Classic method using row/column reduction and assignment phases (Kuhn, 1955)
- Dantzig’s Algorithm — Simplex-based method for solving linear assignment problems (Dantzig, 1963)
- Auction Algorithm — Iterative bidding approach for optimal assignment (Bertsekas, 1988)
- Subgradient Algorithm — Optimization method using subgradient updates for assignment problems (Held & Karp, 1971)

## 🚀 Getting Started

fastlap is available on PyPI and can be installed with `pip install fastlap`.

### Installation

Install fastlap directly from PyPI:

```bash
pip install fastlap
```

For development or building from source:

```bash
# 1. Clone the project
git clone https://github.com/8Opt/fastlap.git
cd fastlap

# 2. Install dependencies
pip install maturin
# or `uv sync`

# 3. Build and install
maturin develop
```

### Example Usage

```python

import fastlap

# Define a sample cost matrix
cost_matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

##  Solve the LAP using the LAPJV algorithm
total_cost, row_assignments, col_assignments = fastlap.solve_lap(cost_matrix, algorithm="lapjv")

print("Total Cost:", total_cost)
print("Row Assignments:", row_assignments)
print("Column Assignments:", col_assignments)


```


## 📄 Citation

If you use fastlap in your research or projects, please cite it as follows:

```

@software{fastlap2025,
  author       = {Le Duc Minh},
  title        = {fastlap: A High-Performance Python LAP Solver Powered by Rust},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/8Opt/fastlap},
  note         = {Python-Rust implementation of LAPJV, Hungarian, LAPMOD, Dantzig’s, Auction, and Subgradient algorithms}
}

```


## 📃 License
fastlap is licensed under the MIT License © 2025.

## 🛠️ Contributing
Contributions are welcome! Please see our Contributing Guidelines for more details on how to get involved.

## 📧 Contact
For questions or support, please open an issue on the GitHub repository or contact the maintainers directly.
