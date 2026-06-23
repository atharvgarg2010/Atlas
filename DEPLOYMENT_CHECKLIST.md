# Atlas Production Deployment Checklist

Use this checklist when linking your GitHub repository to your PaaS provider (Railway, Render, etc.).

## 1. Environment Setup
- [ ] Connect your repository to the hosting provider.
- [ ] Ensure the deployment type is set to **Docker** (using the provided `Dockerfile`).
- [ ] Provision a **PostgreSQL** database add-on.

## 2. Environment Variables
Inject the following variables into your App service configuration:
- [ ] `DATABASE_URL`: `postgresql://user:password@host:port/database` (Usually auto-injected by the PaaS when linking the DB).
- [ ] *Optional:* Any other custom variables from your `.env` file (e.g., API keys for data fetching if applicable).

## 3. Pre-Flight Verification
- [x] **No Hardcoded Windows Paths**: Verified. (`pathlib.Path` is used universally).
- [x] **Dockerfile Build**: Verified. Uses Python 3.10 slim, installs `build-essential`, and exposes the correct port.
- [x] **Dependencies (`requirements.txt`)**: Verified. UTF-16 encoding corruption removed.
- [x] **Startup Command (`start.sh`)**: Verified. Streamlit dynamically binds to `${PORT:-8501}` and the Scheduler runs autonomously in the foreground.
- [x] **Dashboard Boot Integrity**: Verified. The UI gracefully handles missing `research/live` directories or files on the initial cold start.

## 4. Post-Deployment Checks
Once the service is marked "Live" by your provider:
- [ ] **Check Dashboard**: Navigate to the public URL to ensure the Streamlit UI loads (it will show empty/warning states initially).
- [ ] **Check Database Initialization**: Check the deployment logs to ensure `DatabaseManager initialised` is printed.
- [ ] **Wait for First Rebalance**: The Scheduler is set to trigger daily at 18:00 IST. The dashboard will populate after the first run.
