"""
PROC (Processing) node functionality for P2P Network with Federated Learning
Handles model training using preprocessed batches from PREP nodes
"""

import json
import logging
import time
import threading
import requests
import uuid
import network

logger = logging.getLogger(__name__)


class ProcNode:
    """PROC node handles model training"""

    def __init__(self, participant):
        self.participant = participant  # Reference to main Participant instance
        self.polling_active = False
        self.polling_thread = None
        self.active_tasks = {}  # task_id -> thread
        self.poll_interval = 5  # Poll every 5 seconds

    def handle_task_assigned(self, data):
        task_id = data.get("id")
        task_data = data.get("data", {})
        logger.info(f"PROC task: received assigned task {task_id}")
        # Start task in separate thread
        task_thread = threading.Thread(
            target=self._process_task,
            args=(task_id, task_data),
            daemon=True
        )
        self.active_tasks[task_id] = task_thread
        task_thread.start()

    def _process_task(self, task_id, task_data):
        try:
            logger.info(f"Processing PROC task {task_id}")
            self.do_federated_train_task(task_data)
            self._report_task_completed(task_id)
        except Exception as e:
            logger.error(f"PROC task {task_id} failed: {e}")
            self._report_task_error(task_id, str(e))
        finally:
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]

    def _report_task_completed(self, task_id):
        if self.participant.ws and self.participant.connected:
            self.participant.ws.send(json.dumps({
                "type": "task_completed",
                "from": self.participant.name,
                "task_id": task_id,
                "status": "completed"
            }))

    def _report_task_error(self, task_id, error_msg):
        if self.participant.ws and self.participant.connected:
            self.participant.ws.send(json.dumps({
                "type": "task_error",
                "from": self.participant.name,
                "task_id": task_id,
                "error": error_msg
            }))

    def start_polling(self):
        """Start WS polling for training tasks (like PREP node)"""
        if self.polling_active:
            logger.info("PROC polling already active")
            return

        self.polling_active = True
        self.polling_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.polling_thread.start()
        logger.info("Started WS polling for training tasks")

    def stop_polling(self):
        """Stop polling for training tasks"""
        self.polling_active = False
        if self.polling_thread and self.polling_thread.is_alive():
            self.polling_thread.join(timeout=5)
        logger.info("Stopped polling for training tasks")

    def _poll_loop(self):
        """Main WS polling loop for training tasks (sends 'poll' message)"""
        while self.polling_active and self.participant.running:
            try:
                if self.participant.ws and self.participant.connected:
                    self.participant.ws.send(json.dumps({
                        "type": "poll",
                        "from": self.participant.name,
                        "role": "PROC"
                    }))
                    logger.debug("Sent WS poll message")
                else:
                    logger.warning("WS not connected, skipping poll")
            except Exception as e:
                logger.warning(f"Poll error: {e}")

            time.sleep(self.poll_interval)

    def _poll_for_task(self):
        """Poll coordinator for available training tasks"""
        if self.participant.training_active:
            return  # Already training, don't poll for new tasks

        try:
            resp = requests.get(
                f"{self.participant.coordinator_url}/api/poll/tasks",
                params={"name": self.participant.name, "role": "PROC"},
                timeout=10
            )

            if resp.status_code == 200:
                result = resp.json()
                tasks = result.get("tasks", [])
                if tasks:
                    task = tasks[0]
                    task_data = task.get("data", {})
                    # Add job_id from task to data (poll response has it at top level)
                    task_data["job_id"] = task.get("job_id")
                    task_data["job_name"] = task.get("job_name")

                    logger.info(f"Received training task from poll: {task.get('task_type')} for job {task.get('job_id')}")

                    # Dispatch to task queue
                    task_id = str(uuid.uuid4())
                    self.participant.task_queue.put({"task_id": task_id, "type": "task_train", "data": task_data})
                else:
                    logger.debug("No pending training tasks")
            else:
                logger.warning(f"Poll failed with status {resp.status_code}")

        except Exception as e:
            logger.warning(f"Poll request error: {e}")

    def do_federated_train_task(self, data):
        """PROC: Train model from scratch using federated learning"""
               
        job_id = data.get("job_id")
        round_num = data.get("round", 1)
        batches = data.get("batches", [])
        model_type = data.get("model_type", "bert")
        model_name = data.get("model_name", "bert-base-uncased")
        epochs = data.get("epochs", 3)
        learning_rate = data.get("learning_rate", 0.001)
        batch_size = data.get("batch_size", 32)
        threshold = data.get("threshold", 0.1)
        is_first_round = data.get("is_first_round", True)
        
        self.participant.current_job_id = job_id
        self.participant.current_round = round_num
        self.participant.training_active = True
        
        batch_nums = [b.get('batch_number', 0) for b in batches]
        batch_range = f"{min(batch_nums)}-{max(batch_nums)}" if batch_nums else "none"
        logger.info(f"PROC: Starting streaming training for job {job_id}, round {round_num}")
        logger.info(f"  Model: {model_type}/{model_name}")
        logger.info(f"  Expected Batches: {batch_range}, Count: {len(batches)}")
        logger.info(f"  First round: {is_first_round}")
        logger.info(f"  Training will start immediately with available batches and continue indefinitely")
        
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            from transformers import AutoModelForSequenceClassification
            
            # Initialize model (from scratch or load weights)
            if is_first_round:
                logger.info("  Initializing model from scratch")
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=2  # Binary classification by default
                )
                self.participant.local_model = model
            else:
                # Would load global weights from coordinator
                logger.info("  Would load global weights for non-first round")
                # For demo, continue with current model
                if not hasattr(self.participant, 'local_model'):
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_name, num_labels=2
                    )
                    self.participant.local_model = model
            
            model = self.participant.local_model
            model.train()
            
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            # Prepare training data from batches
            # First, check which batches we need and request them if not available locally
            all_sequences = []
            all_masks = []
            all_labels = []
            incorporated_batches = set()

            # Check if we have batch source info
            if hasattr(self.participant, 'batch_sources') and job_id in self.participant.batch_sources:
                batch_sources = self.participant.batch_sources[job_id]

                # Request missing batches from their PREP nodes
                for batch_info in batches:
                    batch_num = batch_info.get('batch_number', 0)
                    batch_key = f"{job_id}_batch_{batch_num}"

                    # Check if we already have this batch locally
                    if batch_key not in getattr(self.participant, 'local_batches', {}):
                        # Request from the PREP node that has it
                        # batch_sources format: {batch_num: {name: "prep-node", ip: "192.168.1.1", port: 11130}}
                        if str(batch_num) in batch_sources:
                            prep_info = batch_sources[str(batch_num)]
                            prep_node = prep_info.get('name', '')
                            prep_ip = prep_info.get('ip', '')
                            prep_port = prep_info.get('port', 11130)

                            # Use direct HTTP request instead of coordinator
                            logger.info(f"Requesting batch {batch_num} directly from {prep_node} ({prep_ip}:{prep_port})")
                            batch_data = network.request_batch_direct(prep_ip, prep_port, job_id, batch_num, timeout=10)
                            if batch_data:
                                batch_key = f"{job_id}_batch_{batch_num}"
                                if not hasattr(self.participant, 'local_batches'):
                                    self.participant.local_batches = {}
                                self.participant.local_batches[batch_key] = batch_data
                                logger.info(f"Received batch {batch_num} directly from {prep_node}")
                            else:
                                # Fallback to coordinator request
                                self.participant._request_batch_from_peer(job_id, batch_num, prep_node, prep_ip)
                                logger.warning(f"Direct request failed, using coordinator")

            # Collect initial available batches for this job (start immediately with what's available)
            local_batches = getattr(self.participant, 'local_batches', {})
            for batch_key, batch_data in local_batches.items():
                if batch_key.startswith(f"{job_id}_batch_") and batch_key not in incorporated_batches:
                    incorporated_batches.add(batch_key)
                    all_sequences.extend(batch_data["input_ids"])
                    all_masks.extend(batch_data["attention_mask"])
                    # Generate dummy labels for training (one per sequence)
                    all_labels.extend([0] * len(batch_data["input_ids"]))
                    batch_num = batch_key.split('_')[-1]
                    logger.info(f"Initial batch {batch_num} available for streaming training")

            # If no initial batches, wait for first batch to start streaming training
            while not all_sequences and self.participant.training_active:
                logger.info("Waiting for first batch to start streaming training")
                time.sleep(1)
                # Check for new batches
                for batch_key, batch_data in local_batches.items():
                    if batch_key.startswith(f"{job_id}_batch_") and batch_key not in incorporated_batches:
                        incorporated_batches.add(batch_key)
                        all_sequences.extend(batch_data["input_ids"])
                        all_masks.extend(batch_data["attention_mask"])
                        all_labels.extend([0] * len(batch_data["input_ids"]))
                        batch_num = batch_key.split('_')[-1]
                        logger.info(f"Added initial batch {batch_num} to training")
                        break  # Start with first batch

            if not all_sequences:
                logger.error("Training stopped before any batch data available")
                self.participant._send_training_update("error", {"error": "Training stopped before batch data available"})
                return

            # Function to rebuild dataloader from current data
            def rebuild_dataloader():
                if not all_sequences:
                    return None
                # Pad sequences to same length
                max_len = max(len(seq) for seq in all_sequences)
                padded_sequences = []
                padded_masks = []
                for seq, mask in zip(all_sequences, all_masks):
                    # Pad with 0s
                    padded_seq = seq + [0] * (max_len - len(seq))
                    padded_msk = mask + [0] * (max_len - len(mask))
                    padded_sequences.append(padded_seq)
                    padded_masks.append(padded_msk)

                # Create tensors (limit for demo, remove for production)
                num_samples = min(len(padded_sequences), batch_size * 10)
                input_ids = torch.tensor(padded_sequences[:num_samples])
                attention_mask = torch.tensor(padded_masks[:num_samples])
                labels = torch.tensor(all_labels[:num_samples])

                dataset = TensorDataset(input_ids, attention_mask, labels)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                return dataloader

            dataloader = rebuild_dataloader()

            # Streaming training loop - indefinite continuation
            total_loss = 0
            epoch = 0
            while self.participant.training_active:
                # Check for new batches and incorporate them (any batch for this job)
                new_batches_added = False
                local_batches = getattr(self.participant, 'local_batches', {})
                for batch_key, batch_data in local_batches.items():
                    if batch_key.startswith(f"{job_id}_batch_") and batch_key not in incorporated_batches:
                        incorporated_batches.add(batch_key)
                        all_sequences.extend(batch_data["input_ids"])
                        all_masks.extend(batch_data["attention_mask"])
                        all_labels.extend([0] * len(batch_data["input_ids"]))
                        batch_num = batch_key.split('_')[-1]
                        new_batches_added = True
                        logger.info(f"Streaming: Added batch {batch_num} to ongoing training")

                if new_batches_added:
                    dataloader = rebuild_dataloader()
                    logger.info(f"Streaming: Rebuilt dataset with {len(all_sequences)} total samples")

                if dataloader and len(dataloader) > 0:
                    epoch_loss = 0
                    for batch in dataloader:
                        batch_input_ids, batch_attention, batch_labels = batch

                        optimizer.zero_grad()
                        outputs = model(
                            input_ids=batch_input_ids,
                            attention_mask=batch_attention,
                            labels=batch_labels
                        )
                        loss = outputs.loss
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()

                    avg_loss = epoch_loss / len(dataloader)
                    total_loss += avg_loss
                    epoch += 1
                    logger.info(f"  Streaming Epoch {epoch} - Loss: {avg_loss:.4f}, Samples: {len(all_sequences)}")
                else:
                    # No data available, wait
                    time.sleep(1)

            # Training stopped, send final update
            if epoch > 0:
                avg_loss = total_loss / epoch
            else:
                avg_loss = 0.0

            # Calculate accuracy (simplified)
            accuracy = max(0.5, 1.0 - avg_loss)  # Simplified metric

            logger.info(f"PROC: Streaming training stopped for round {round_num} - Epochs: {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

            # Get model weights (delta)
            weights = self._extract_model_weights(model)

            # Send update to coordinator
            self.participant._send_model_update(job_id, round_num, weights, avg_loss, accuracy)
            
        except Exception as e:
            logger.error(f"PROC training error: {e}")
            import traceback
            traceback.print_exc()
            self.participant._send_training_update("error", {"error": str(e)})
    
    def _extract_model_weights(self, model):
        """Extract model weights as dictionary"""
        weights = {}
        for name, param in model.named_parameters():
            # For efficiency, could send only updated params
            # For now, send all as float32
            weights[name] = param.detach().cpu().numpy().tolist()
        return weights
    
    def handle_batch_data(self, data):
        """Handle received batch data from PREP node"""
        job_id = data.get("job_id")
        batch_num = data.get("batch_number")
        batch_data = data.get("batch_data", {})
        from_peer = data.get("from")
        logger.info(f"Received batch {batch_num} from {from_peer}")
        
        # Store the batch locally
        batch_key = f"{job_id}_batch_{batch_num}"
        if not hasattr(self.participant, 'local_batches'):
            self.participant.local_batches = {}
        self.participant.local_batches[batch_key] = batch_data
    
    def handle_batch_sources(self, data):
        """Store batch source info (which PREP node has which batch)"""
        job_id = data.get("job_id")
        batch_sources = data.get("batch_sources", {})
        if not hasattr(self.participant, 'batch_sources'):
            self.participant.batch_sources = {}
        self.participant.batch_sources[job_id] = batch_sources
        logger.info(f"Received batch sources for job {job_id}: {len(batch_sources)} batches")
    
   