#!/bin/bash
# scheduler_setup.sh
# ==================
# Run once on GCP VM to install all cron jobs.
#
# Usage:
#   gcloud compute ssh data-collector --zone=asia-south1-c
#   cd ~/quant-fund
#   bash src/automation/scheduler_setup.sh
#
# What it installs:
#   6:15 PM IST  — wait for pipeline, then run rolling optimizer
#   8:45 AM IST  — pre-market param health check
#   6:00 AM IST  — daily git pull (keeps VM code up to date)
#
# IST = UTC+5:30
# 6:15 PM IST = 12:45 UTC
# 8:45 AM IST = 03:15 UTC
# 6:00 AM IST = 00:30 UTC

set -e

REPO_DIR="/home/ubuntu/quant-fund"
VENV_DIR="/home/ubuntu/quant-fund/venv"
LOG_DIR="/home/ubuntu/quant-fund/logs"
PYTHON="$VENV_DIR/bin/python"

echo "========================================"
echo "  Quant Fund Scheduler Setup"
echo "========================================"
echo ""

# Verify paths
if [ ! -d "$REPO_DIR" ]; then
    echo "ERROR: Repo not found at $REPO_DIR"
    exit 1
fi

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Python venv not found at $PYTHON"
    echo "Run: python -m venv $VENV_DIR && $VENV_DIR/bin/pip install -r requirements.txt"
    exit 1
fi

mkdir -p "$LOG_DIR"

echo "Installing cron jobs..."
echo ""

# Build cron entries
# Note: cron uses UTC times
CRON_JOBS="
# ── Quant Fund Automation ────────────────────────────────────────────────────

# 6:00 AM IST (00:30 UTC) — git pull to keep VM code up to date
30 0 * * 1-5 cd $REPO_DIR && git pull origin main >> $LOG_DIR/git_pull.log 2>&1

# 6:15 PM IST (12:45 UTC) — nightly rolling param optimizer (after pipeline)
# Waits 15 min after pipeline scheduler (which runs at 6:00 PM IST / 12:30 UTC)
45 12 * * 1-5 cd $REPO_DIR && $PYTHON -m src.automation.rolling_optimizer --model all >> $LOG_DIR/rolling_optimizer.log 2>&1

# 8:45 AM IST (03:15 UTC) — pre-market param health check
15 3 * * 1-5 cd $REPO_DIR && $PYTHON -m src.automation.param_health_check >> $LOG_DIR/param_health_check.log 2>&1

# ─────────────────────────────────────────────────────────────────────────────
"

# Install cron jobs (preserve existing non-quant-fund entries)
# 1. Get current crontab (ignore error if empty)
CURRENT_CRON=$(crontab -l 2>/dev/null || true)

# 2. Remove existing quant-fund block if present
CLEANED_CRON=$(echo "$CURRENT_CRON" | sed '/# ── Quant Fund Automation/,/^$/d')

# 3. Add new block
NEW_CRON="${CLEANED_CRON}${CRON_JOBS}"

# 4. Install
echo "$NEW_CRON" | crontab -

echo "Cron jobs installed. Current crontab:"
echo ""
crontab -l
echo ""

# Verify python can import the modules
echo "Verifying imports..."
cd "$REPO_DIR"
$PYTHON -c "from src.automation.rolling_optimizer import run_optimization; print('  rolling_optimizer: OK')"
$PYTHON -c "from src.backtest.param_transfer import transfer_params; print('  param_transfer: OK')"
$PYTHON -c "from src.backtest.param_loader import get_symbol_params; print('  param_loader: OK')"

echo ""
echo "========================================"
echo "  Setup complete."
echo ""
echo "  Schedule (IST):"
echo "    6:00 AM  — git pull"
echo "    6:15 PM  — rolling param optimizer"
echo "    8:45 AM  — param health check"
echo ""
echo "  Logs:"
echo "    $LOG_DIR/rolling_optimizer.log"
echo "    $LOG_DIR/param_health_check.log"
echo "    $LOG_DIR/git_pull.log"
echo ""
echo "  Test run (dry run, no saves):"
echo "    $PYTHON -m src.automation.rolling_optimizer --dry-run"
echo "========================================"