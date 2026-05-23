# 🚑 Auto DJ Recovery Plan

In the event of a system failure, process hang, or failed transition rendering, follow these sequential steps to restore operations.

## 1. Confirm Failure
- **Check Dashboard**: Observe the "Incident Recovery Log" in the Web Console for red ERROR entries.
- **Inspect Session Logs**: Check the `.log` files in the repository root or the redirected output from `auto_dj.py` for specific tracebacks.
- **Verify Status**: If the "System State" pill remains stuck in "Warping" or "Mixing" for more than 5 minutes without progress, a process may have hung.

## 2. Terminate Stuck Processes
- **Kill GUI/Engine**: `kill $(lsof -t -i :8000)` to stop the FastAPI server and its background tasks.
- **Clean Orphaned Workers**: `pgrep -af python3` to find orphaned `ProcessPoolExecutor` workers and terminate them using `kill -9 <PID>`.
- **Force Port Release**: If the dashboard fails to restart, ensure the port is free: `fuser -k 8000/tcp`.

## 3. Clear Transient State
- **Reset Nodes**: Use the "↺" button in the Cluster Monitor UI to clear failed task counts for specific nodes.
- **Delete Artifacts**: Remove any partially rendered `.wav` files in the output directory to prevent checksum or file-access conflicts.

## 4. Cold Restart
- Restart the engine: `python3 auto_dj.py --gui`.
- Re-initiate the session via the "🚀 INITIATE SESSION" button.

---
*Stay resilient. The party must go on.*
