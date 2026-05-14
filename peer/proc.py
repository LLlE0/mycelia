"""
PROC (Processing) node functionality for P2P Network with Federated Learning
Handles distributed model training using preprocessed batches from PREP nodes
Supports parallel training across multiple PROC nodes
"""

import json
import logging
import time
import threading
import gc
import queue
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
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
        self._task_queue = queue.Queue()
        self._worker_thread = None
        self._worker_running = True
        self._busy_task_id = None
        self._start_worker()

    def _start_worker(self):
        if self._worker_thread and self._worker_thread.is_alive():
            return

        def worker():
            logger.info("PROC worker started (single-thread)")
            while self._worker_running and self.participant.running:
                try:
                    item = self._task_queue.get(timeout=1)
                except queue.Empty:
                    continue
                if item is None:
                    break

                task_id, task_data = item
                try:
                    self.active_tasks[task_id] = "running"
                    self._busy_task_id = task_id
                    self._process_task(task_id, task_data)
                finally:
                    self._busy_task_id = None
                    self._task_queue.task_done()
            logger.info("PROC worker stopped")

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()

    def stop_worker(self):
        self._worker_running = False
        try:
            self._task_queue.put_nowait(None)
        except Exception:
            pass
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)

    def handle_task_assigned(self, data):
        task_id = data.get("id")
        task_data = data.get("data", {})
        logger.info(f"PROC task: received assigned task {task_id}")
        if self._busy_task_id is not None or (not self._task_queue.empty()):
            logger.warning(f"PROC is busy (current={self._busy_task_id}), rejecting task {task_id}")
            self._report_task_error(task_id, "busy")
            return

        self.active_tasks[task_id] = "queued"
        self._task_queue.put((task_id, task_data))

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
                if self._busy_task_id is not None or (not self._task_queue.empty()):
                    time.sleep(self.poll_interval)
                    continue
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
        
        learning_rate = float(data.get("learning_rate", 2e-5))
        if learning_rate > 1e-3:
            logger.warning(f"Learning rate {learning_rate} is too high for BERT, capping to 2e-5")
            learning_rate = 2e-5
            
        batch_size = data.get("batch_size", 32)
        is_first_round = data.get("is_first_round", True)
        
        self.participant.current_job_id = job_id
        self.participant.current_round = round_num
        self.training_active = True
        
        logger.info(f"PROC: Starting training for job {job_id}, round {round_num}")
        logger.info(f"  Model: {model_name}, Epochs: {epochs}, LR: {learning_rate}, Batch Size: {batch_size}")
        logger.info(f"  Batches assigned: {len(batches)}")
        
        try:
            if is_first_round:
                logger.info("  Initializing model from scratch")
                self.local_model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, num_labels=151
                )
            else:
                logger.info("  Loading global weights from coordinator")
                if self.local_model is None:
                    self.local_model = AutoModelForSequenceClassification.from_pretrained(
                        model_name, num_labels=151
                    )
            
            model = self.local_model
            model.train()
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            all_sequences = []
            all_masks = []
            all_labels = []
            incorporated_batches = set()
            
            if not hasattr(self.participant, 'batch_sources') or job_id not in self.participant.batch_sources:
                logger.warning(f"No batch sources available for job {job_id}, waiting...")
                wait_count = 0
                while (not hasattr(self.participant, 'batch_sources') or job_id not in self.participant.batch_sources) and wait_count < 30:
                    time.sleep(1)
                    wait_count += 1
            
            max_batches_to_load = len(batches)
            
            if hasattr(self.participant, 'batch_sources') and job_id in self.participant.batch_sources:
                batch_sources = self.participant.batch_sources[job_id]
                logger.info(f"Found batch sources for job {job_id}: {len(batch_sources)} mappings")
                
                batches_requested = 0
                for batch_info in batches:
                    if batches_requested >= max_batches_to_load:
                        logger.info(f"Requested maximum {max_batches_to_load} batches to conserve memory")
                        break
                        
                    batch_num = batch_info.get('batch_number', 0)
                    batch_key = f"{job_id}_batch_{batch_num}"
                    
                    local_batches = getattr(self.participant, 'local_batches', {})
                    if batch_key not in local_batches:
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
                                batches_requested += 1
                        else:
                            logger.warning(f"Batch {batch_num} not found in batch_sources")
                    else:
                        batches_requested += 1
            
            local_batches = getattr(self.participant, 'local_batches', {})
            batches_loaded = 0
            for batch_info in batches:
                if isinstance(batch_info, dict):
                    batch_num = batch_info.get("batch_number", 0)
                    if isinstance(batch_num, str) and batch_num.isdigit():
                        batch_num = int(batch_num)
                    elif isinstance(batch_num, float):
                        batch_num = int(batch_num)
                else:
                    batch_num = int(batch_info)

                batch_key = f"{job_id}_batch_{batch_num}"
                if batch_key not in local_batches:
                    logger.warning(f"Missing local batch {batch_num} after fetch; skipping")
                    continue

                batch_data = local_batches[batch_key]
                if batch_key in incorporated_batches:
                    continue
                incorporated_batches.add(batch_key)

                seqs = batch_data.get("input_ids", [])
                masks = batch_data.get("attention_mask", [])
                labels_from_batch = batch_data.get("labels", None)

                all_sequences.extend(seqs)
                all_masks.extend(masks)

                if labels_from_batch is not None:
                    all_labels.extend(labels_from_batch)
                    logger.info(f"Added batch {batch_num} to training data ({len(seqs)} samples) with labels")
                else:
                    all_labels.extend([0] * len(seqs))
                    logger.warning(f"Added batch {batch_num} without labels - using default label 0")

                batches_loaded += 1
            
            if not all_sequences:
                logger.error("No data available for training after loading batches")
                return

            logger.info(f"Prepared {len(all_sequences)} samples for training")
            
            unique_labels = list(set(all_labels))
            logger.info(f"Unique labels in dataset: {unique_labels}")
            if len(unique_labels) == 1:
                logger.warning(f"All labels are {unique_labels[0]}! Model won't learn meaningful patterns.")
            
            class FederatedDataset(torch.utils.data.Dataset):
                def __init__(self, sequences, masks, labels):
                    self.sequences = sequences
                    self.masks = masks
                    self.labels = labels

                def __len__(self):
                    return len(self.sequences)

                def __getitem__(self, idx):
                    return {
                        'input_ids': torch.tensor(self.sequences[idx], dtype=torch.long),
                        'attention_mask': torch.tensor(self.masks[idx], dtype=torch.long),
                        'labels': torch.tensor(self.labels[idx], dtype=torch.long)
                    }

            dataset = FederatedDataset(all_sequences, all_masks, all_labels)
            def collate_fn(items):
                input_ids_list = [x["input_ids"] for x in items]
                attn_list = [x["attention_mask"] for x in items]
                labels_list = [x["labels"] for x in items]

                input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
                attention_mask = pad_sequence(attn_list, batch_first=True, padding_value=0)
                labels = torch.stack(labels_list, dim=0)

                return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=collate_fn,
            )

            logger.info(f"Starting training loop: {len(dataloader)} steps per epoch, batch_size={batch_size}")
            gc.collect()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                torch.cuda.empty_cache()
            model.to(device)
            
            del all_sequences, all_masks, all_labels, dataset
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            total_loss = 0.0
            final_loss = 0.0

            for epoch in range(epochs):
                if not self.training_active:
                    break
                
                model.train()
                epoch_loss = 0.0
                steps_count = 0
                
                for batch_idx, batch in enumerate(dataloader):
                    if not self.training_active:
                        break

                    input_ids = batch['input_ids'].to(device, non_blocking=(device == "cuda"))
                    attention_mask = batch['attention_mask'].to(device, non_blocking=(device == "cuda"))
                    labels = batch['labels'].to(device, non_blocking=(device == "cuda"))

                    optimizer.zero_grad(set_to_none=True)  
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    
                    if not torch.isfinite(loss):
                        logger.warning(f"Non-finite loss detected: {loss.item()}, skipping batch")
                        del input_ids, attention_mask, labels, outputs, loss
                        continue
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()

                    loss_value = loss.item()
                    epoch_loss += loss_value
                    steps_count += 1
                    
                    if epoch == 0 and batch_idx == 0:
                        with torch.no_grad():
                            preds = torch.argmax(outputs.logits, dim=-1)
                            logger.info(f"Debug - First batch: labels={labels[:5].cpu().tolist()}, preds={preds[:5].cpu().tolist()}, loss={loss_value:.4f}")
                    
                    del input_ids, attention_mask, labels, outputs, loss
                    if torch.cuda.is_available() and batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                    
                    if steps_count % 10 == 0:
                        logger.info(f"  Epoch {epoch+1}/{epochs}, Step {steps_count}/{len(dataloader)}: Loss {loss_value:.4f}")

                if steps_count > 0:
                    avg_epoch_loss = epoch_loss / steps_count
                    total_loss += avg_epoch_loss
                    logger.info(f"  Epoch {epoch+1}/{epochs} completed - Avg Loss: {avg_epoch_loss:.4f}")
                    
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    logger.warning(f"  Epoch {epoch+1} had no steps!")

            if epochs > 0 and total_loss > 0:
                final_loss = total_loss / epochs
            else:
                final_loss = epoch_loss / steps_count if steps_count > 0 else 0.0

            logger.info(f"Training complete - Final Loss: {final_loss:.4f}")

            if torch.cuda.is_available():
                model.cpu()
                torch.cuda.empty_cache()
            
            weights = self._extract_model_weights(model)
            self.participant._send_model_update(job_id, round_num, weights, final_loss)
            
            if hasattr(self.participant, 'local_batches'):
                keys_to_delete = [k for k in self.participant.local_batches.keys() if k.startswith(f"{job_id}_batch_")]
                for key in keys_to_delete:
                    del self.participant.local_batches[key]
                logger.info(f"Cleaned up {len(keys_to_delete)} batches for job {job_id}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
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
