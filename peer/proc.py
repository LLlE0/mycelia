"""
PROC (Processing) node functionality for P2P Network with Federated Learning
Handles distributed model training using preprocessed batches from PREP nodes
Supports parallel training across multiple PROC nodes
"""

import json
import logging
import time
import threading
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class ProcNode:
    """PROC node handles distributed model training"""

    def __init__(self, participant):
        self.participant = participant
        self.polling_active = False
        self.polling_thread = None
        self.active_tasks = {}
        self.poll_interval = 5
        self.local_model = None
        self.training_active = False

    def handle_task_assigned(self, data):
        task_id = data.get("id")
        task_data = data.get("data", {})
        logger.info(f"PROC task: received assigned task {task_id}")
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
        if self.polling_active:
            return
        self.polling_active = True
        self.polling_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.polling_thread.start()
        logger.info("Started WS polling for training tasks")

    def stop_polling(self):
        self.polling_active = False
        if self.polling_thread and self.polling_thread.is_alive():
            self.polling_thread.join(timeout=5)
        logger.info("Stopped polling for training tasks")

    def _poll_loop(self):
        while self.polling_active and self.participant.running:
            try:
                if self.participant.ws and self.participant.connected:
                    self.participant.ws.send(json.dumps({
                        "type": "poll",
                        "from": self.participant.name,
                        "role": "PROC"
                    }))
            except Exception as e:
                logger.warning(f"Poll error: {e}")
            time.sleep(self.poll_interval)

    def do_federated_train_task(self, data):
        """PROC: Training task received via polling from coordinator"""
        
        job_id = data.get("job_id")
        round_num = data.get("round", 1)
        batches = data.get("batches", [])
        model_name = data.get("model_name", "bert-base-uncased")
        epochs = data.get("epochs", 3)
        learning_rate = data.get("learning_rate", 0.001)
        batch_size = data.get("batch_size", 32)
        is_first_round = data.get("is_first_round", True)
        
        self.participant.current_job_id = job_id
        self.participant.current_round = round_num
        self.training_active = True
        
        logger.info(f"PROC: Starting training for job {job_id}, round {round_num}")
        logger.info(f"  Model: {model_name}, Epochs: {epochs}, LR: {learning_rate}")
        logger.info(f"  Batches assigned: {len(batches)}")
        
        try:
            # Initialize or load model
            if is_first_round:
                logger.info("  Initializing model from scratch")
                self.local_model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=2
                )
            else:
                logger.info("  Loading global weights from coordinator")
                if self.local_model is None:
                    self.local_model = AutoModelForSequenceClassification.from_pretrained(
                        model_name, num_labels=2
                    )
            
            model = self.local_model
            model.train()
            
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            # Wait for batch sources to be available
            if not hasattr(self.participant, 'batch_sources') or job_id not in self.participant.batch_sources:
                logger.info(f"Waiting for batch sources for job {job_id}...")
                wait_count = 0
                while (not hasattr(self.participant, 'batch_sources') or job_id not in self.participant.batch_sources) and wait_count < 60:
                    time.sleep(1)
                    wait_count += 1
            
            if not hasattr(self.participant, 'batch_sources') or job_id not in self.participant.batch_sources:
                logger.error(f"No batch sources available for job {job_id} after waiting")
                return

            batch_sources = self.participant.batch_sources[job_id]
            logger.info(f"Found batch sources for job {job_id}: {len(batch_sources)} mappings")
            
            # Ensure all required batches are downloaded before training
            # Store only metadata, not full data in memory
            batch_metadata = []
            
            for batch_info in batches:
                batch_num = batch_info.get('batch_number', 0)
                batch_key = f"{job_id}_batch_{batch_num}"
                
                # Check if already downloaded
                if batch_key not in getattr(self.participant, 'local_batches', {}):
                    if str(batch_num) in batch_sources:
                        prep_info = batch_sources[str(batch_num)]
                        prep_name = prep_info.get('name', '')
                        prep_ip = prep_info.get('ip', '')
                        prep_port = prep_info.get('port', 11130)
                        
                        logger.info(f"Requesting batch {batch_num} from {prep_name} ({prep_ip}:{prep_port})")
                        from network import request_batch_direct
                        batch_data = request_batch_direct(prep_ip, prep_port, job_id, batch_num, timeout=30)
                        
                        if batch_data:
                            if not hasattr(self.participant, 'local_batches'):
                                self.participant.local_batches = {}
                            self.participant.local_batches[batch_key] = batch_data
                            logger.info(f"Received batch {batch_num} successfully")
                        else:
                            logger.error(f"Failed to download batch {batch_num}")
                    else:
                        logger.warning(f"Batch {batch_num} not found in batch_sources")
                else:
                    logger.debug(f"Batch {batch_num} already in memory")
                
                batch_metadata.append(batch_num)

            # Verify we have data
            local_batches = getattr(self.participant, 'local_batches', {})
            available_count = sum(1 for bn in batch_metadata if f"{job_id}_batch_{bn}" in local_batches)
            
            if available_count == 0:
                logger.error("No data available for training after downloading")
                return

            logger.info(f"Starting training with {available_count} batches loaded")
            
            # Helper function to get a single batch tensor
            def get_batch_tensor(batch_num):
                batch_key = f"{job_id}_batch_{batch_num}"
                if batch_key not in local_batches:
                    return None
                
                batch_data = local_batches[batch_key]
                input_ids = batch_data.get("input_ids", [])
                attention_mask = batch_data.get("attention_mask", [])
                
                if not input_ids:
                    return None
                
                # Pad to max length
                model_max_length = 512
                padded_seqs = []
                padded_masks = []
                
                for seq, mask in zip(input_ids, attention_mask):
                    if len(seq) < model_max_length:
                        seq = seq + [0] * (model_max_length - len(seq))
                        mask = mask + [0] * (model_max_length - len(mask))
                    else:
                        seq = seq[:model_max_length]
                        mask = mask[:model_max_length]
                    padded_seqs.append(seq)
                    padded_masks.append(mask)
                
                ids_tensor = torch.tensor(padded_seqs, dtype=torch.long)
                mask_tensor = torch.tensor(padded_masks, dtype=torch.long)
                # Labels are assumed 0 for this example, adjust if needed
                labels_tensor = torch.zeros(len(ids_tensor), dtype=torch.long)
                
                return ids_tensor, mask_tensor, labels_tensor

            # Create a custom dataset that loads batches on demand to save memory
            class StreamingDataset(torch.utils.data.Dataset):
                def __init__(self, batch_nums, get_batch_fn):
                    self.batch_nums = batch_nums
                    self.get_batch_fn = get_batch_fn
                    self._cache = {} # Cache individual samples if needed, but better to cache whole batches
                    
                    # Pre-load all batches into a single large tensor structure? 
                    # No, that causes OOM. Instead, we keep them separate and index logically.
                    # Actually, for simplicity and speed with small number of batches (32), 
                    # we can concatenate them once here IF they fit. 
                    # But since we just OOMed, let's try to be smarter.
                    # We will just store the list of tensors and index into them.
                    
                    self.all_ids = []
                    self.all_masks = []
                    self.all_labels = []
                    
                    logger.info("Concatenating batches for dataset...")
                    for bn in self.batch_nums:
                        res = self.get_batch_fn(bn)
                        if res:
                            ids, masks, labels = res
                            self.all_ids.append(ids)
                            self.all_masks.append(masks)
                            self.all_labels.append(labels)
                    
                    if self.all_ids:
                        self.total_len = sum(t.shape[0] for t in self.all_ids)
                        logger.info(f"Total samples in dataset: {self.total_len}")
                    else:
                        self.total_len = 0

                def __len__(self):
                    return self.total_len

                def __getitem__(self, idx):
                    # Find which batch this index belongs to
                    current_idx = 0
                    for i, ids_tensor in enumerate(self.all_ids):
                        batch_len = ids_tensor.shape[0]
                        if idx < current_idx + batch_len:
                            local_idx = idx - current_idx
                            return (
                                self.all_ids[i][local_idx],
                                self.all_masks[i][local_idx],
                                self.all_labels[i][local_idx]
                            )
                        current_idx += batch_len
                    raise IndexError("Index out of range")

            # Build dataset
            dataset = StreamingDataset(batch_metadata, get_batch_tensor)
            
            if len(dataset) == 0:
                logger.error("Dataset is empty")
                return

            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            
            logger.info(f"Starting training loop: {len(dataloader)} steps per epoch")
            
            # Training loop
            total_loss = 0
            for epoch in range(epochs):
                if not self.training_active:
                    break
                    
                epoch_loss = 0
                step_count = 0
                
                for batch in dataloader:
                    batch_input_ids, batch_attention, batch_labels = batch
                    
                    # Move to GPU if available
                    if torch.cuda.is_available():
                        batch_input_ids = batch_input_ids.to('cuda')
                        batch_attention = batch_attention.to('cuda')
                        batch_labels = batch_labels.to('cuda')
                        model.to('cuda')

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
                    step_count += 1
                    
                    # Optional: Log progress every 10 steps
                    if step_count % 10 == 0:
                        logger.debug(f"Epoch {epoch+1}, Step {step_count}, Loss: {loss.item():.4f}")
                
                if step_count > 0:
                    avg_loss = epoch_loss / step_count
                    total_loss += avg_loss
                    logger.info(f"  Epoch {epoch+1}/{epochs} completed - Loss: {avg_loss:.4f}")
                else:
                    logger.warning(f"  Epoch {epoch+1} had no steps")
            
            if epochs > 0:
                avg_loss = total_loss / epochs
            else:
                avg_loss = 0.0
            
            logger.info(f"Training complete - Final loss: {avg_loss:.4f}")
            
            # Extract weights for aggregation
            weights = self._extract_model_weights(model)
            
            # Send weights to coordinator for aggregation
            self.participant._send_model_update(job_id, round_num, weights, avg_loss)
            
            # Optional: Clear memory after training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Optionally clear local batches if no longer needed immediately
            # self.participant.local_batches = {} 
            
        except Exception as e:
            logger.error(f"PROC training error: {e}")
            import traceback
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _extract_model_weights(self, model):
        """Extract model weights as dictionary"""
        weights = {}
        for name, param in model.named_parameters():
            weights[name] = param.detach().cpu().numpy().tolist()
        return weights

    def handle_batch_data(self, data):
        job_id = data.get("job_id")
        batch_num = data.get("batch_number")
        batch_data = data.get("batch_data", {})
        from_peer = data.get("from")
        logger.info(f"Received batch {batch_num} from {from_peer}")
        
        batch_key = f"{job_id}_batch_{batch_num}"
        if not hasattr(self.participant, 'local_batches'):
            self.participant.local_batches = {}
        self.participant.local_batches[batch_key] = batch_data

    def handle_batch_sources(self, data):
        job_id = data.get("job_id")
        batch_sources = data.get("batch_sources", {})
        if not hasattr(self.participant, 'batch_sources'):
            self.participant.batch_sources = {}
        self.participant.batch_sources[job_id] = batch_sources
        logger.info(f"Received batch sources for job {job_id}: {len(batch_sources)} batches")
