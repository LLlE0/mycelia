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
                # Load global weights from previous round
                logger.info("  Loading global weights from coordinator")
                if self.local_model is None:
                    self.local_model = AutoModelForSequenceClassification.from_pretrained(
                        model_name, num_labels=2
                    )
            
            model = self.local_model
            model.train()
            
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            # Collect all assigned batches
            all_sequences = []
            all_masks = []
            all_labels = []
            incorporated_batches = set()
            
            # Request and collect batches from PREP nodes
            if not hasattr(self.participant, 'batch_sources') or job_id not in self.participant.batch_sources:
                logger.warning(f"No batch sources available for job {job_id}, waiting...")
                wait_count = 0
                while (not hasattr(self.participant, 'batch_sources') or job_id not in self.participant.batch_sources) and wait_count < 30:
                    time.sleep(1)
                    wait_count += 1
            
            if hasattr(self.participant, 'batch_sources') and job_id in self.participant.batch_sources:
                batch_sources = self.participant.batch_sources[job_id]
                logger.info(f"Found batch sources for job {job_id}: {len(batch_sources)} mappings")
                
                for batch_info in batches:
                    batch_num = batch_info.get('batch_number', 0)
                    batch_key = f"{job_id}_batch_{batch_num}"
                    
                    if batch_key not in getattr(self.participant, 'local_batches', {}):
                        if str(batch_num) in batch_sources:
                            prep_info = batch_sources[str(batch_num)]
                            prep_name = prep_info.get('name', '')
                            prep_ip = prep_info.get('ip', '')
                            prep_port = prep_info.get('port', 11130)
                            
                            logger.info(f"Requesting batch {batch_num} from {prep_name} ({prep_ip}:{prep_port})")
                            from network import request_batch_direct
                            batch_data = request_batch_direct(prep_ip, prep_port, job_id, batch_num, timeout=15)
                            
                            if batch_data:
                                if not hasattr(self.participant, 'local_batches'):
                                    self.participant.local_batches = {}
                                self.participant.local_batches[batch_key] = batch_data
                                logger.info(f"Received batch {batch_num} successfully")
                        else:
                            logger.warning(f"Batch {batch_num} not found in batch_sources")
            else:
                logger.error(f"Still no batch sources for job {job_id} after waiting")
            
            # Collect local batches
            local_batches = getattr(self.participant, 'local_batches', {})
            for batch_key, batch_data in local_batches.items():
                if batch_key.startswith(f"{job_id}_batch_") and batch_key not in incorporated_batches:
                    batch_num = int(batch_key.split('_')[-1])
                    incorporated_batches.add(batch_key)
                    # Use input_ids for BERT training (not embeddings)
                    all_sequences.extend(batch_data["input_ids"])
                    all_masks.extend(batch_data["attention_mask"])
                    all_labels.extend([0] * len(batch_data.get("input_ids", [])))
                    logger.info(f"Added batch {batch_num} to training data")
            
            if not all_sequences:
                logger.warning("No batches available, waiting...")
                wait_count = 0
                while not all_sequences and self.training_active and wait_count < 60:
                    time.sleep(1)
                    wait_count += 1
                    local_batches = getattr(self.participant, 'local_batches', {})
                    for batch_key, batch_data in local_batches.items():
                        if batch_key.startswith(f"{job_id}_batch_") and batch_key not in incorporated_batches:
                            incorporated_batches.add(batch_key)
                            all_sequences.extend(batch_data["input_ids"])
                            all_masks.extend(batch_data["attention_mask"])
                            all_labels.extend([0] * len(batch_data.get("input_ids", [])))
                            logger.info(f"Added batch after waiting")
                            break
            
            if not all_sequences:
                logger.error("No data available for training")
                return
            
            # Prepare dataset - pad sequences to model's expected max_length (512 for BERT)
            # The error occurs when sequence length doesn't match model's embedding size
            # We need to ensure consistent padding to 512 tokens
            model_max_length = 512  # BERT default max_length
            padded_sequences = []
            padded_masks = []
            for seq, mask in zip(all_sequences, all_masks):
                if len(seq) < model_max_length:
                    seq = seq + [0] * (model_max_length - len(seq))
                    mask = mask + [0] * (model_max_length - len(mask))
                else:
                    seq = seq[:model_max_length]
                    mask = mask[:model_max_length]
                padded_sequences.append(seq)
                padded_masks.append(mask)
            
            # Limit samples for efficiency - increased from 10 to 100 batches
            num_samples = min(len(padded_sequences), batch_size * 100)
            input_ids = torch.tensor(padded_sequences[:num_samples], dtype=torch.long)
            attention_mask = torch.tensor(padded_masks[:num_samples], dtype=torch.long)
            labels = torch.tensor(all_labels[:num_samples], dtype=torch.long)
            
            dataset = TensorDataset(input_ids, attention_mask, labels)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            logger.info(f"Starting training with {len(dataset)} samples, {len(dataloader)} batches")
            
            # Training loop
            total_loss = 0
            for epoch in range(epochs):
                if not self.training_active:
                    break
                    
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
                logger.info(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            
            if epochs > 0:
                avg_loss = total_loss / epochs
            else:
                avg_loss = 0.0
            
            logger.info(f"Training complete - Final loss: {avg_loss:.4f}")
            
            # Extract weights for aggregation
            weights = self._extract_model_weights(model)
            
            # Send weights to coordinator for aggregation
            self.participant._send_model_update(job_id, round_num, weights, avg_loss)
            
        except Exception as e:
            logger.error(f"PROC training error: {e}")
            import traceback
            traceback.print_exc()

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
