# Atlas Deployment-Readiness Audit

This document outlines the findings of the deployment-readiness audit targeted for PaaS environments (Railway, Render) and Linux-based Docker deployments.

## 1. Package Requirements (`requirements.txt`)
**Status:** ❌ **FAIL (Deployment Blocker)**
- **Issue:** The `requirements.txt` file contains a critical encoding corruption at the end of the file. The entries for `schedule` and `tabulate` were appended using UTF-16 LE encoding (with null bytes like `\x00`), while the rest of the file is UTF-8. 
- **Impact:** Running `pip install -r requirements.txt` during the Docker build process will instantly crash with an encoding error, halting the entire deployment.
- **Fix:** Open `requirements.txt`, delete the corrupted bottom lines (`s c h e d u l e = = 1 . 2 . 1`, etc.), and manually re-type them as:
  ```text
  schedule==1.2.1
  tabulate==0.9.0
  ```

## 2. Dockerfile Build
**Status:** ❌ **FAIL (Dependent on fixing #1)**
- **Issue:** As mentioned above, the `pip install` step inside the Dockerfile will fail due to the `requirements.txt` encoding. 
- **Otherwise:** The Dockerfile is correctly structured for a Python 3.10 slim environment. It properly exposes port 8501 and executes `start.sh`.

## 3. Hardcoded Windows Paths
**Status:** ✅ **PASS**
- **Verification:** A comprehensive regex search for `C:\`, `R:\`, and absolute path patterns across the `.py` codebase returned 0 hits.
- **Note:** All file generation utilizes `pathlib.Path(__file__)` to construct paths dynamically, making the system 100% portable across Windows and Linux.

## 4. Streamlit Startup Command (`start.sh`)
**Status:** ❌ **FAIL (Deployment Blocker for PaaS)**
- **Issue:** The `start.sh` file hardcodes the Streamlit port:
  `streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0 &`
- **Impact:** Cloud providers like Render and Railway dynamically allocate a port and inject it via the `$PORT` environment variable. If Streamlit binds strictly to `8501`, the platform's health checks will fail, and the container will be killed for failing to bind to the assigned port.
- **Fix:** Modify `start.sh` to use the environment variable with a fallback to 8501 for local development:
  `streamlit run dashboard/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 &`

## 5. File Read Mechanisms
**Status:** ⚠️ **WARNING (Local Windows Bug)**
- **Verification:** All file reads utilize relative `pathlib` objects.
- **Issue:** In `dashboard/app.py`, the `load_latest_md` function opens markdown files without specifying the encoding (`open(files[0], "r")`). 
- **Impact:** Because Linux defaults to UTF-8, this will **PASS** successfully on Railway/Render. However, if you run the dashboard locally on Windows, Python defaults to `cp1252` encoding. Reading the newly added `✅` emojis in the Data Quality Reports will trigger a `UnicodeDecodeError` and crash your local Streamlit instance.
- **Fix:** Update `dashboard/app.py` to use `open(..., "r", encoding="utf-8")`.

## 6. Dashboard Resilience (Missing `research/live`)
**Status:** ✅ **PASS**
- **Verification:** The `load_json_file` and `load_latest_md` functions in `dashboard/app.py` wrap their logic in `if directory.exists():` and `if file_path.exists():`.
- **Impact:** If deployed to a fresh environment without historical data, the dashboard will render warning banners ("Not enough history", "No active portfolio") instead of crashing.

## 7. Scheduler Independence
**Status:** ✅ **PASS**
- **Verification:** `scheduler.py` is entirely decoupled from the dashboard. It initializes its own database connection pool and runs its own APScheduler instance.
- **Impact:** It can run autonomously in the foreground of the Docker container while Streamlit runs in the background.

---

### Summary of Action Items Before Deployment
1. Re-type the last two lines of `requirements.txt` to fix the UTF-16 corruption.
2. Update `start.sh` to bind to `${PORT:-8501}`.
3. Add `encoding="utf-8"` to file reads in `dashboard/app.py`.
