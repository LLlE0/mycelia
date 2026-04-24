# Distributed ML Diploma Project TODO

## Current Task: Reassign failed PREP tasks to other PREP nodes

### Breakdown of Approved Plan
- [ ] **Step 1**: Add `Retries`/`MaxRetries` fields to `PendingTask` in `coordinator/models.go`
- [ ] **Step 2**: Add `GetAssignedPendingTasks` DB helper in `coordinator/database.go`
- [ ] **Step 3**: Update `task_error` handler in `coordinator/server.go` to reset failed tasks to `pending` (with retry limit)
- [ ] **Step 4**: Update `handlePrepNodeDisconnect` in `coordinator/server.go` to reset assigned tasks on disconnect
- [ ] **Step 5**: Remove dead (unreachable) second `task_completed` block in WebSocket handler
- [ ] **Step 6**: Set `MaxRetries: 3` when creating PREP `PendingTask` in `startTrainingJob`
- [ ] **Step 7**: Build coordinator and verify compilation

