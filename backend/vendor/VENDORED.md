# Vendored third-party code

Nothing here is pip-installed. Files are copied from upstream and modified only
as noted. Do not import from `vendor/` directly anywhere in the project —
import from `backend/stats_bootstrap.py` instead.

---

## arch_bootstrap (from `bashtage/arch`)

- **Source repo:** https://github.com/bashtage/arch
- **Commit SHA:** `b4781d51eb8c371b2a8ea79cd444e1cc3a34eb9d`
- **Upstream commit date:** 2026-09-20
- **Vendored on:** 2026-09-21
- **License:** NCSA (see `arch_bootstrap/LICENSE.md`, copied verbatim)

### Files copied

| Upstream path | Vendored path |
|---|---|
| `arch/_typing.py` | `arch_bootstrap/_typing.py` |
| `arch/compat/numba.py` | `arch_bootstrap/compat/numba.py` |
| `arch/compat/pandas.py` | `arch_bootstrap/compat/pandas.py` |
| `arch/utility/array.py` | `arch_bootstrap/utility/array.py` |
| `arch/utility/exceptions.py` | `arch_bootstrap/utility/exceptions.py` |
| `arch/bootstrap/base.py` | `arch_bootstrap/bootstrap/base.py` |
| `arch/bootstrap/multiple_comparison.py` | `arch_bootstrap/bootstrap/multiple_comparison.py` |
| `arch/bootstrap/_samplers_python.py` | `arch_bootstrap/bootstrap/_samplers_python.py` |
| `LICENSE.md` | `arch_bootstrap/LICENSE.md` |

### Modifications

1. **Imports rewritten.** All `from arch.<...>` imports in executable code were
   converted to package-relative imports (`from .._typing import ...`,
   `from ..utility.array import ...`, `from .base import ...`, etc.). Relative
   form was chosen over `from vendor.arch_bootstrap.<...>` so the package works
   regardless of whether `backend/` or the repo root is on `sys.path`.
   Docstring doctest examples were rewritten to
   `from vendor.arch_bootstrap.bootstrap import ...`, which is the real import
   path for a caller with `backend/` on `sys.path`.

2. **Upstream `__init__.py` files replaced, not copied.** Upstream's
   `arch/__init__.py`, `arch/compat/__init__.py`, `arch/utility/__init__.py`
   and `arch/bootstrap/__init__.py` import the whole arch package (mean models,
   volatility models, unit-root tests), which would pull in far more than the
   bootstrap code and require dependencies not present here.
   - `vendor/__init__.py`, `arch_bootstrap/__init__.py`,
     `arch_bootstrap/compat/__init__.py`, `arch_bootstrap/utility/__init__.py`
     are **empty**.
   - `arch_bootstrap/bootstrap/__init__.py` re-exports only:
     `CircularBlockBootstrap`, `IIDBootstrap`, `MovingBlockBootstrap`,
     `StationaryBootstrap`, `optimal_block_length`, `MCS`, `SPA`, `StepM`.

3. **No source logic changed.** Only import statements were edited.

### Known behaviour

- `arch/bootstrap/_samplers.py` (the compiled Cython sampler) was **not**
  copied. `base.py` tries to import it and falls back to the pure-Python
  `_samplers_python` on ImportError. That fallback is the intended path here.
- numba is not installed, so `StationaryBootstrap` emits a `PerformanceWarning`
  on every call. Expected and harmless; `stats_bootstrap.py` suppresses it
  within its own wrappers only.
