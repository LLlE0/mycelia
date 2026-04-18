package coordinator

import "html/template"

var htmlTemplate = template.Must(template.New("index").Parse(`<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>P2P Network Coordinator</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 20px; background-color: #f8f9fa; }
        .container { max-width: 1400px; }
        .card { margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status-online { color: #28a745; }
        .status-offline { color: #dc3545; }
        .nav-tabs { margin-bottom: 20px; }
        .coordinator-badge { background-color: #ffc107; color: #000; }
        .prep-badge { background-color: #0d6efd; color: #fff; }
    </style>
</head>
<body>
    <div class="container">
        <div class="row mb-4">
            <div class="col">
                <h1>P2P Network Coordinator</h1>
                <div class="alert alert-info">
                    <strong>Coordinator:</strong> {{.Coordinator.Name}} | 
                    <strong>Address:</strong> {{.Coordinator.Address}}:{{.Coordinator.Port}} | 
                    <strong>Role:</strong> {{.Coordinator.Role}}
                </div>
            </div>
        </div>
        
        <!-- Coordinator Console -->
        <div class="card">
            <div class="card-header bg-warning">
                <h5 class="mb-0">Coordinator Console</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-8">
                        <div class="input-group">
                            <span class="input-group-text">Command</span>
                            <select class="form-select" id="cmdType" style="max-width: 100px;">
                                <option value="send">send</option>
                                <option value="ping">ping</option>
                            </select>
                            <input type="text" class="form-control" id="cmdTarget" placeholder="Target name">
                            <input type="text" class="form-control" id="cmdContent" placeholder="Message content (for send)">
                            <button class="btn btn-primary" onclick="executeCommand()">Execute</button>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="input-group">
                            <input type="text" class="form-control" id="consoleOutput" placeholder="Command output..." readonly>
                        </div>
                    </div>
                </div>
                <div class="mt-2">
                    <small class="text-muted">
                        Commands: "send <name> <message>" or "ping <name>" (from coordinator)
                    </small>
                </div>
            </div>
        </div>
        
        <ul class="nav nav-tabs" id="myTab" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="participants-tab" data-bs-toggle="tab" data-bs-target="#participants" type="button">Network Members</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="messages-tab" data-bs-toggle="tab" data-bs-target="#messages" type="button">Messages</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="training-tab" data-bs-toggle="tab" data-bs-target="#training" type="button">Training Tasks</button>
            </li>
        </ul>
        
        <div class="tab-content" id="myTabContent">
            <div class="tab-pane fade show active" id="participants" role="tabpanel">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">Network Members</h5>
                    </div>
                    <div class="card-body">
                        <table class="table table-hover">
                            <thead>
                                <tr>
                                    <th>Name</th>
                                    <th>Role</th>
                                    <th>CC</th>
                                    <th>Address</th>
                                    <th>Port</th>
                                    <th>Status</th>
                                    <th>System Info</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody id="members-list">
                                {{range .Members}}
                                <tr>
                                    <td>
                                        {{.Name}}
                                        {{if eq .Role "COORD"}}
                                        <span class="badge coordinator-badge">COORD</span>
                                        {{else}}
                                        <span class="badge prep-badge">PREP</span>
                                        {{end}}
                                    </td>
                                    <td>{{.Role}}</td>
                                    <td>
                                        {{if .System}}{{index .System "compute_credit"}}{{else}}-{{end}}
                                    </td>
                                    <td>{{.Address}}</td>
                                    <td>{{.Port}}</td>
                                    <td><span class="badge {{if eq .Status "online"}}bg-success{{else}}bg-danger{{end}}">{{.Status}}</span></td>
                                    <td>
                                        {{if .System}}
                                        <small>
                                            CPU: {{index .System "cpu"}}% | 
                                            RAM: {{index .System "memory"}}% | 
                                            Disk: {{index .System "disk"}}%
                                        </small>
                                        {{else}}
                                        <small class="text-muted">N/A</small>
                                        {{end}}
                                    </td>
                                    <td>
                                        <button class="btn btn-sm btn-outline-primary" onclick="ping('{{.Name}}')">Ping</button>
                                        <button class="btn btn-sm btn-outline-success" onclick="prepareSend('{{.Name}}')">Send</button>
                                        {{if ne .Role "COORD"}}
                                        <button class="btn btn-sm btn-outline-danger" onclick="deleteParticipant('{{.Name}}')">Delete</button>
                                        {{end}}
                                    </td>
                                </tr>
                                {{end}}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            
            <div class="tab-pane fade" id="messages" role="tabpanel">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Message History</h5>
                    </div>
                    <div class="card-body">
                        <table class="table table-hover">
                            <thead>
                                <tr>
                                    <th>From</th>
                                    <th>To</th>
                                    <th>Content</th>
                                    <th>Timestamp</th>
                                </tr>
                            </thead>
                            <tbody id="messages-list">
                                {{range .Messages}}
                                <tr>
                                    <td>{{.From}}</td>
                                    <td>{{.To}}</td>
                                    <td>{{.Content}}</td>
                                    <td>{{.Timestamp.Format "2006-01-02 15:04:05"}}</td>
                                </tr>
                                {{end}}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            
            <div class="tab-pane fade" id="training" role="tabpanel">
                <!-- Create Training Job Form -->
                <div class="card mb-4">
                    <div class="card-header bg-primary text-white">
                        <h5 class="mb-0">Create New Training Job</h5>
                    </div>
                    <div class="card-body">
                        <form id="trainingForm" onsubmit="return createTrainingJob(event)">
                            <div class="row mb-3">
                                <div class="col-md-6">
                                    <label class="form-label">Job Name</label>
                                    <input type="text" class="form-control" id="jobName" required placeholder="e.g., sentiment-analysis-model">
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label">Model Type</label>
                                    <select class="form-select" id="modelType">
                                        <option value="bert">BERT</option>
                                        <option value="lstm">LSTM</option>
                                        <option value="cnn">CNN</option>
                                        <option value="gpt">GPT</option>
                                        <option value="transformer">Transformer</option>
                                        <option value="linear">Linear Regression</option>
                                    </select>
                                </div>
                            </div>
                            <div class="row mb-3">
                                <div class="col-md-8">
                                    <label class="form-label">Dataset URL</label>
                                    <input type="url" class="form-control" id="datasetUrl" required placeholder="https://example.com/dataset.csv" value="https://huggingface.co/datasets/lucynwang/text_classifier_bert">
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label">Dataset Size</label>
                                    <input type="number" class="form-control" id="datasetSize" value="19500" min="1" placeholder="Total records" onchange="calculateTotalBatches()">
                                </div>
                            </div>
                            <div class="row mb-3">
                                <div class="col-md-2">
                                    <label class="form-label">Batch Size</label>
                                    <input type="number" class="form-control" id="batchSize" value="32" min="1" onchange="calculateTotalBatches()">
                                </div>
                                <div class="col-md-2">
                                    <label class="form-label">Total Batches</label>
                                    <input type="number" class="form-control" id="totalBatches" min="1" readonly title="Calculated automatically">
                                </div>
                                <div class="col-md-2">
                                    <label class="form-label">Epochs</label>
                                    <input type="number" class="form-control" id="epochs" value="10" min="1">
                                </div>
                                <div class="col-md-3">
                                    <label class="form-label">Learning Rate</label>
                                    <input type="number" class="form-control" id="learningRate" value="0.02" step="0.0001">
                                </div>
                                <div class="col-md-3">
                                    <label class="form-label">Max Rounds</label>
                                    <input type="number" class="form-control" id="maxRounds" value="100" min="1">
                                </div>
                            </div>
                            <div class="row mb-3">
                                <div class="col-md-12">
                                    <label class="form-label">Description</label>
                                    <textarea class="form-control" id="jobDescription" rows="2" placeholder="Optional description..."></textarea>
                                </div>
                            </div>
                            <button type="submit" class="btn btn-primary">Create Training Job</button>
                        </form>
                    </div>
                </div>
                
                <!-- Training Jobs List -->
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">Training Jobs</h5>
                        <button class="btn btn-sm btn-outline-secondary" onclick="loadTrainingJobs()">Refresh</button>
                    </div>
                    <div class="card-body">
                        <table class="table table-hover">
                            <thead>
                                <tr>
                                    <th>Name</th>
                                    <th>Model</th>
                                    <th>Dataset</th>
                                    <th>Progress</th>
                                    <th>Status</th>
                                    <th>Created</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody id="training-jobs-list">
                                <tr><td colspan="7" class="text-center text-muted">Loading...</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function prepareSend(name) {
            document.getElementById('cmdType').value = 'send';
            document.getElementById('cmdTarget').value = name;
            document.getElementById('cmdContent').focus();
        }
        
        async function executeCommand() {
            const cmdType = document.getElementById('cmdType').value;
            const target = document.getElementById('cmdTarget').value;
            const content = document.getElementById('cmdContent').value;
            
            if (!target) {
                alert('Please specify target');
                return;
            }
            
            const res = await fetch('/api/command', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    type: cmdType,
                    target: target,
                    content: content
                })
            });
            
            const data = await res.json();
            document.getElementById('consoleOutput').value = JSON.stringify(data);
            
            if (res.ok) {
                setTimeout(() => location.reload(), 500);
            }
        }
        
        async function deleteParticipant(name) {
            if (!confirm('Delete ' + name + '?')) return;
            
            const res = await fetch('/api/participant/' + name, {method: 'DELETE'});
            if (res.ok) {
                location.reload();
            }
        }
        
        async function ping(name) {
            const res = await fetch('/api/ping/' + name);
            const data = await res.json();
            alert(name + ' is ' + data.status);
        }
        
        // Handle Enter key in command input
        document.getElementById('cmdContent').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                executeCommand();
            }
        });
        
        // Load training jobs on page load
        document.addEventListener('DOMContentLoaded', function() {
            loadTrainingJobs();
            calculateTotalBatches(); // Calculate initial value
        });

        function calculateTotalBatches() {
            const datasetSize = parseInt(document.getElementById('datasetSize').value) || 0;
            const batchSize = parseInt(document.getElementById('batchSize').value) || 1;
            const totalBatches = Math.ceil(datasetSize / batchSize);
            document.getElementById('totalBatches').value = totalBatches;
        }
        
        async function createTrainingJob(e) {
            e.preventDefault();
            
            const jobData = {
                name: document.getElementById('jobName').value,
                model_type: document.getElementById('modelType').value,
                dataset_url: document.getElementById('datasetUrl').value,
                dataset_size: parseInt(document.getElementById('datasetSize').value),
                batch_size: parseInt(document.getElementById('batchSize').value),
                total_batches: parseInt(document.getElementById('totalBatches').value),
                epochs: parseInt(document.getElementById('epochs').value),
                learning_rate: parseFloat(document.getElementById('learningRate').value),
                max_rounds: parseInt(document.getElementById('maxRounds').value),
                description: document.getElementById('jobDescription').value,
                dataset_type: 'csv',
                threshold: 0.01,
                status: 'pending'
            };
            
            try {
                const res = await fetch('/api/training', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(jobData)
                });
                
                if (res.ok) {
                    alert('Training job created successfully!');
                    document.getElementById('trainingForm').reset();
                    loadTrainingJobs();
                } else {
                    const err = await res.json();
                    alert('Error: ' + (err.error || 'Failed to create job'));
                }
            } catch (err) {
                alert('Error: ' + err.message);
            }
            
            return false;
        }
        
        async function loadTrainingJobs() {
            const tbody = document.getElementById('training-jobs-list');
            if (!tbody) return;
            
            try {
                const res = await fetch('/api/training');
                const jobs = await res.json();
                
                if (!jobs || jobs.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No training jobs</td></tr>';
                    return;
                }
                
                let html = '';
                for (const job of jobs) {
                    const progress = Math.round(job.progress || 0);
                    const statusBadge = getStatusBadge(job.status);
                    const jobId = job._id || job.id;
                    const name = job.name || 'N/A';
                    const modelType = job.model_type || 'N/A';
                    const datasetUrl = job.dataset_url ? job.dataset_url.substring(0, 30) + '...' : 'N/A';
                    const status = job.status || 'pending';
                    const createdAt = job.created_at ? new Date(job.created_at).toLocaleDateString() : 'N/A';
                    
                    let actions = '';
                    if (status === 'pending') {
                        actions = '<button class="btn btn-sm btn-success" onclick="startTrainingJob(' + jobId + ')">Start</button>';
                    } else if (status === 'training') {
                        actions = '<button class="btn btn-sm btn-danger" onclick="stopTrainingJob(' + jobId + ')">Stop</button>';
                    }
                    actions += '<button class="btn btn-sm btn-outline-primary" onclick="viewTrainingJob(' + jobId + ')">View</button>';
                    
                    html += '<tr>';
                    html += '<td><strong>' + name + '</strong></td>';
                    html += '<td>' + modelType + '</td>';
                    html += '<td><small>' + datasetUrl + '</small></td>';
                    html += '<td><div class="progress" style="height: 20px;"><div class="progress-bar" role="progressbar" style="width: ' + progress + '%">' + progress + '%</div></div></td>';
                    html += '<td><span class="badge ' + statusBadge + '">' + status + '</span></td>';
                    html += '<td><small>' + createdAt + '</small></td>';
                    html += '<td>' + actions + '</td>';
                    html += '</tr>';
                }
                tbody.innerHTML = html;
            } catch (err) {
                tbody.innerHTML = '<tr><td colspan="7" class="text-danger">Error loading jobs: ' + err.message + '</td></tr>';
            }
        }
        
        function getStatusBadge(status) {
            switch(status) {
                case 'pending': return 'bg-secondary';
                case 'preprocessing': return 'bg-info';
                case 'training': return 'bg-primary';
                case 'completed': return 'bg-success';
                case 'failed': return 'bg-danger';
                default: return 'bg-secondary';
            }
        }
        
        async function startTrainingJob(jobId) {
            try {
                const res = await fetch('/api/training/' + jobId + '/start', {method: 'PUT'});
                if (res.ok) {
                    alert('Training started!');
                    loadTrainingJobs();
                } else {
                    alert('Failed to start training');
                }
            } catch (err) {
                alert('Error: ' + err.message);
            }
        }
        
        async function stopTrainingJob(jobId) {
            try {
                const res = await fetch('/api/training/' + jobId + '/stop', {method: 'PUT'});
                if (res.ok) {
                    alert('Training stopped');
                    loadTrainingJobs();
                } else {
                    alert('Failed to stop training');
                }
            } catch (err) {
                alert('Error: ' + err.message);
            }
        }
        
        async function viewTrainingJob(jobId) {
            try {
                const res = await fetch('/api/training/' + jobId);
                const job = await res.json();
                
                let details = 'Job: ' + (job.name || 'N/A') + '\n';
                details += 'Model: ' + (job.model_type || 'N/A') + '\n';
                details += 'Dataset: ' + (job.dataset_url || 'N/A') + '\n';
                details += 'Progress: ' + Math.round(job.progress || 0) + '%\n';
                details += 'Status: ' + (job.status || 'pending') + '\n';
                details += 'Current Loss: ' + (job.current_loss ? job.current_loss.toFixed(4) : 'N/A') + '\n';
                details += 'Current Round: ' + (job.current_round || 0) + '/' + (job.max_rounds || 0);
                
                alert(details);
            } catch (err) {
                alert('Error: ' + err.message);
            }
        }
    </script>
</body>
</html>`))
