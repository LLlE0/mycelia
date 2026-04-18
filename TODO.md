# Distributed ML Diploma Project TODO

## Current Task: Fix Training Tasks Not Created/Assigned

### Breakdown of Approved Plan
✅ **Step 1**: Create TODO.md - DONE  
✅ **Step 2**: Enable WS polling in peer/proc.py - DONE  
✅ **Step 3**: Relaxed trigger in coordinator/server.go batch_progress - DONE  
✅ **Step 4**: Fixed WS poll handler - now supports role=="PROC" (GetPendingTasksByRole("PROC")) - DONE

**Test**:
- Rebuild coordinator: `cd coordinator && go build -o bin/coordinator`
- Run nodes, create job → PREP processes → batches ready → "starting training round" log → PROC polls → "Assigned task TRAIN to PROC" → executes.

Progress: 4/5 complete. PROC now receives train tasks from PendingTasks!

**Final verification**: Check logs/DB after PREP completes.



