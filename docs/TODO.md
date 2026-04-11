# fastlap — Project TODO

> Acting as project manager: this list focuses on making fastlap genuinely useful to real users —
> discoverable on PyPI, trustworthy via correctness, and easy to contribute to.
> Items are ordered by impact. Each item is self-contained and actionable.

---

## 🔴 P0 — Blockers (ship nothing until these are done)

### Fix algorithm correctness
- [ ] **Hungarian**: The simplified augmenting-path implementation produces suboptimal solutions on matrices ≥ 3×3. Replace with a correct Kuhn-Munkres implementation (proper alternating-path search with backtracking).
- [ ] **LAPJV/LAPMOD**: Verify the augmenting-path logic produces the global optimum, not just a locally feasible assignment.
- [ ] **Auction**: `best_item` is initialised to `0` before the loop — if no item beats `-INFINITY` (impossible, but fragile), it silently assigns item 0. Guard this.
- [ ] **Add a correctness benchmark**: After fixing, assert that every algorithm produces identical cost to `scipy.optimize.linear_sum_assignment` for 1 000 random square matrices of sizes 2–50. Gate CI on this.

### PyPI publishing pipeline
- [ ] **Test the CI publish workflow end-to-end** against TestPyPI before the next release (`scripts/publish.sh` points at the real index).
- [ ] **Add `PYPI_API_TOKEN` to the GitHub repo secrets** so the automated publish job in `CI.yml` can actually run.
- [ ] **Bumb `Cargo.toml` + `pyproject.toml` versions in sync** before every publish. Currently they can drift.

---

## 🟠 P1 — High Impact (next two weeks)

### Broaden platform/Python support
- [ ] **Add `aarch64-unknown-linux-gnu` target** to `CI.yml` build matrix (AWS Graviton, Raspberry Pi, Docker on Apple Silicon).
- [ ] **Test against Python 3.11, 3.12, 3.13** — the `CI.yml` only specifies `'3.x'`. Pin a matrix of `[3.9, 3.10, 3.11, 3.12, 3.13]` to catch ABI breaks early.
- [ ] **Update the Python badge in README.md** — currently says `3.8–3.10`, which is stale.

### Sparse matrix support
- [ ] **Wire sparse input through `solve_lap`** — `matrix.rs` already has `extract_sparse_matrix` but `solve_lap` in `lib.rs` still only receives a dense `PyArray2`. Expose a second `solve_lap_sparse(cost_matrix, algorithm)` entry point, or unify under `PyAny` (already done in the source — just needs a test).
- [ ] **Add a test for sparse input** in `tests/test_correctness.py` using `scipy.sparse.csr_matrix`.

### Separate CI from CD
- [ ] **Add a `ci.yml` that runs on every PR** (`cargo check`, `clippy`, `pytest`) — the only current workflow is the publish trigger. Developers get no feedback on PRs.
- [ ] **Add `cargo audit`** to CI to catch known CVEs in the dependency tree.

---

## 🟡 P2 — Quality of Life (next month)

### Test coverage
- [ ] **Add non-square matrix tests** — no test currently exercises rectangular inputs (`n × m` where `n ≠ m`).
- [ ] **Add edge-case tests**: empty matrix, 1×1 matrix, matrix with duplicate costs, matrix with `inf`/`nan` entries.
- [ ] **Add performance regression test** — `test_performance.py` exists but only prints, never asserts. Assert that `fastlap.lapjv` is ≤ 2× slower than `lap.lapjv` on a 100×100 matrix.

### Documentation
- [ ] **Write a proper `CONTRIBUTING.md`** — README says "see Contributing Guidelines" but the file does not exist.
- [ ] **Add docstrings to `solve_lap` in `lib.rs`** so PyO3 exposes `help(fastlap.solve_lap)` correctly.
- [ ] **Replace lorem-ipsum placeholder text in `docs/content/journal.md`** — it's a Zola template leftover (remove or replace with an actual dev journal).
- [ ] **Add algorithm comparison table to README**: columns = algorithm, time complexity, optimality guarantee, square-only.
- [ ] **Add a visualization example**: Create a script using `matplotlib` or `seaborn` that plots a cost matrix as a heatmap and overlays the optimal assignment (circles/x-marks) to help users verify results visually.
- [ ] **Replace `examples/__init__.py`** (currently blank) with a re-export so `from fastlap.examples import run_example` works.
- [ ] **Update `CHANGELOG.md`** — the current entries are undated or dated 2025 but are blank.

### Developer experience
- [ ] **Add `cargo fmt --check`** to `scripts/test_rust.sh` so formatting is enforced.
- [ ] **Add `scripts/test_all.sh`** that chains `test_rust.sh` and `test_python.sh`.
- [ ] **Add a `Makefile`** (or `justfile`) as a top-level alternative to remembering script names.
- [ ] **Add `.cargo/config.toml`** with `[build] rustflags = ["-D", "warnings"]` so warnings are errors in local builds without needing to remember the clippy flag.

---

## 🟢 P3 — Growth (next quarter)

### Feature expansion
- [ ] **Expose `get_supported_algorithms()` in docs/README** — users don't know it exists.
- [ ] **Add async/parallel solve option** — use Rayon to solve multiple independent matrices in parallel and expose a `solve_lap_batch(matrices, algorithm)` function.
- [ ] **Add optional `weights` parameter** to re-scale costs before solving (common in tracking pipelines).
- [ ] **Publish to conda-forge** — many scientific Python users are on conda, not pip.

### Visibility
- [ ] **Write a blog post / notebook comparing fastlap to `lapjv`, `scipy`, `lapsolver`** with benchmarks on real-world tracking data sizes (100–10 000 objects). Publish to the Zola docs site.
- [ ] **Add a `doi` to the citation** via Zenodo so the package is formally citable in papers.
- [ ] **Submit to Awesome-Rust and Awesome-Python-Scientific** lists for discoverability.
