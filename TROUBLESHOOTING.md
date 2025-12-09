# Troubleshooting PySR Crash on macOS

## Issue

When importing `pysr` (specifically `PySRRegressor`), the Python process crashes immediately with exit code 137 (SIGKILL). This often happens on macOS due to conflicts between Python's and Julia's signal handling mechanisms.

## Solution

The issue is resolved by disabling `juliacall`'s signal handling before importing `pysr`. This is done by setting the `PYTHON_JULIACALL_HANDLE_SIGNALS` environment variable to `"no"`.

### Code Fix

In your Python script (e.g., `main.py`), add the following lines **before** importing `pysr`:

```python
import os
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "no"
import pysr
```

## Notes

- You may see a warning: `UserWarning: PYTHON_JULIACALL_HANDLE_SIGNALS environment variable is set to something other than 'yes' or ''. You will experience segfaults if running with multithreading.`
- This warning is expected. While disabling signal handling might cause issues with multithreading in some edge cases, it is necessary to prevent the immediate crash on startup in this environment.
