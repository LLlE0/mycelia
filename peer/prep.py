"""
PREP (Preprocessing) node functionality for P2P Network with Federated Learning
Handles data preprocessing: loading datasets, tokenizing, and storing batches locally
"""

import json
import logging
import pandas as pd
import warnings
import threading
import time
import requests

logger = logging.getLogger(__name__)


class PrepNode:
    """PREP node handles dataset preprocessing and batch storage"""
    
    def __init__(self, participant):
        self.participant = participant  # Reference to main Participant instance
        self.polling_active = False
        self.polling_thread = None
        self.active_tasks = {}  # task_id -> thread
        self.max_parallel_tasks = 3  # Allow up to 3 parallel preprocessing tasks
        self.poll_interval = 5  # Poll every 5 seconds

    def start_polling(self):
        """Start polling for preprocessing tasks"""
        if self.polling_active:
            logger.info("Polling already active")
            return

        self.polling_active = True
        self.polling_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.polling_thread.start()
        logger.info("Started polling for preprocessing tasks")

    def stop_polling(self):
        """Stop polling for preprocessing tasks"""
        self.polling_active = False
        if self.polling_thread and self.polling_thread.is_alive():
            self.polling_thread.join(timeout=5)
        logger.info("Stopped polling for preprocessing tasks")

    def _poll_loop(self):
        """Main polling loop for preprocessing tasks"""
        while self.polling_active and self.participant.running:
            try:
                self._poll_for_task()
            except Exception as e:
                logger.warning(f"Poll error: {e}")

            time.sleep(self.poll_interval)

    def _process_task(self, task_id, task_data):
        """Process a single preprocessing task"""
        try:
            logger.info(f"Processing task {task_id}")
            self.do_preprocess_task(task_data)

            # Mark task as completed
            self._report_task_completed(task_id)

        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            self._report_task_error(task_id, str(e))
        finally:
            # Remove from active tasks
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]

    def _report_task_completed(self, task_id):
        """Report task completion to coordinator"""
        if self.participant.ws and self.participant.connected:
            try:
                self.participant.ws.send(json.dumps({
                    "type": "task_completed",
                    "from": self.participant.name,
                    "task_id": task_id,
                    "status": "completed"
                }))
                logger.info(f"Reported task {task_id} completion")
            except Exception as e:
                logger.warning(f"Failed to report task completion: {e}")

    def _report_task_error(self, task_id, error_msg):
        """Report task error to coordinator"""
        if self.participant.ws and self.participant.connected:
            try:
                self.participant.ws.send(json.dumps({
                    "type": "task_error",
                    "from": self.participant.name,
                    "task_id": task_id,
                    "error": error_msg
                }))
                logger.error(f"Reported task {task_id} error: {error_msg}")
            except Exception as e:
                logger.warning(f"Failed to report task error: {e}")

    def do_preprocess_task(self, data):
        """PREP: Preprocess dataset for federated learning - split into batches"""
        # Initialize ML environment if not already done

        job_id = data.get("job_id")
        dataset_url = data.get("dataset_url", "https://huggingface.co/datasets/lucynwang/text_classifier_bert")
        dataset_type = data.get("dataset_type", "csv")
        batch_size = data.get("batch_size", 32)
        start_batch = data.get("start_batch", 0)
        end_batch = data.get("end_batch", 610)
        start_batch_num = start_batch
        model_type = data.get("model_type", "bert")
        task_id = data.get("task_id")

        start_offset = data.get("start_offset", 0)
        end_offset = data.get("end_offset", 1000)
        logger.info(f"DEBUG: Received data: start_batch={start_batch}, end_batch={end_batch}, batch_size={batch_size}")
        logger.info(f"DEBUG: Offsets: start_offset={start_offset}, end_offset={end_offset}")

        self.participant.current_job_id = job_id
        self.participant.training_active = True

        logger.info(f"PREP: Starting preprocessing task {task_id} for job {job_id}")
        logger.info(f"  Dataset: {dataset_url}, Type: {dataset_type}")
        logger.info(f"  Records: {start_offset}-{end_offset}, Batch size: {batch_size}")
        logger.info(f"  Model: {model_type}")

        try:
            
            total_records_needed = end_offset - start_offset
            total_batches = total_records_needed // batch_size

            logger.info(f"  Loading from offset={start_offset}, length={total_records_needed}")
            dataset = self._load_dataset(dataset_url, dataset_type, offset=start_offset, length=total_records_needed)
            total_records = len(dataset)
            
            logger.info(f"  Loaded {total_records} records")
            
            logger.info("  Loading tokenizer...")
            tokenizer = self._get_tokenizer(model_type)
            
            texts = dataset['text'].tolist() if 'text' in dataset.columns else dataset.iloc[:, 0].tolist()
            logger.info(f"  Tokenizing {len(texts)} records...")
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            logger.info("Tokenization complete")
            
            for i in range(total_batches):
                if not self.participant.training_active:
                    logger.info("Preprocessing stopped")
                    break

                batch_num = start_batch_num + i
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(texts))
                
                if start_idx >= len(texts):
                    break
                
                # Extract batch from pre-tokenized data
                # Handle both tensor and list types from tokenizer
                input_ids_data = encoded["input_ids"]
                attention_mask_data = encoded["attention_mask"]

                if hasattr(input_ids_data, 'tolist'):
                    # It's a tensor, convert to list
                    batch_input_ids = input_ids_data[start_idx:end_idx].tolist()
                    batch_attention_mask = attention_mask_data[start_idx:end_idx].tolist()
                else:
                    # It's already a list
                    batch_input_ids = input_ids_data[start_idx:end_idx]
                    batch_attention_mask = attention_mask_data[start_idx:end_idx]
                
                # Store batch locally
                batch_key = f"{job_id}_batch_{batch_num}"
                stored_data = {
                    "input_ids": batch_input_ids.tolist() if hasattr(batch_input_ids, 'tolist') else batch_input_ids,
                    "attention_mask": batch_attention_mask.tolist() if hasattr(batch_attention_mask, 'tolist') else batch_attention_mask,
                }
                self._store_batch_locally(batch_key, stored_data)
                
                # Report progress to coordinator
                progress = (i + 1) / total_batches * 100
                self.participant._report_batch_progress(job_id, batch_num, "ready", progress, end_idx - start_idx)

                logger.info(f"  Batch {batch_num} ready ({progress:.1f}%)")
            
            logger.info(f"PREP: Completed preprocessing for job {job_id}")
            
        except Exception as e:
            logger.error(f"PREP error: {e}")
            # Report error
            if self.participant.ws and self.participant.connected:
                self.participant.ws.send(json.dumps({
                    "type": "preprocess_error",
                    "from": self.participant.name,
                    "job_id": job_id,
                    "error": str(e)
                }))
    
    def _load_dataset(self, url, dataset_type, offset=0, length=None):
        """Load dataset from URL or local file"""
        
        if "huggingface.co/datasets" in url:
            return self._load_huggingface_dataset(url, dataset_type, offset=offset, length=length)
        
        if url.startswith("http://") or url.startswith("https://"):
            logger.info(f"Downloading dataset from {url}...")
            try:
                import requests
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Determine filename from URL
                filename = url.split('/')[-1]
                if '?' in filename:
                    filename = filename.split('?')[0]
                
                # Save to temp file
                temp_path = f"/tmp/{filename}"
                with open(temp_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Downloaded to {temp_path}")
                return self._load_local_file(temp_path, dataset_type)
                
            except Exception as e:
                logger.warning(f"Failed to download dataset: {e}, using sample data")
        
        # Try to load from local file
        import os
        if os.path.exists(url):
            return self._load_local_file(url, dataset_type)
        
        # Generate sample data for testing
        logger.info("Using sample data for testing")
        return self._generate_sample_data()
    
    def _load_huggingface_rows(self, dataset_path, offset=0, length=100):
        rows_url = f"https://datasets-server.huggingface.co/rows?dataset={dataset_path}&config=default&split=train&offset={offset}&length={length}"
        logger.info(f"Loading HF rows API: {rows_url}")
        
        try:
            resp = requests.get(rows_url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            rows = data.get("rows", [])
            logger.info(f"Loaded {len(rows)} rows from HF rows API (requested length={length})")
            
            if not rows:
                raise ValueError("No rows returned from HF API")
            
            # Flatten row data
            row_data = []
            for row in rows:
                row_data.append(row["row"])
            
            return pd.DataFrame(row_data)
        except Exception as e:
            logger.error(f"HF rows API failed: {e}")
            logger.info("Using sample data")
            return self._generate_sample_data()
    
    def _load_huggingface_dataset(self, url, dataset_type, offset=0, length=100):
        if "huggingface.co/datasets/" in url:
            dataset_path = url.split("huggingface.co/datasets/")[1].split("?")[0]
        else:
            dataset_path = url
        
        return self._load_huggingface_rows(dataset_path, offset, length)
    
    def _load_local_file(self, path, dataset_type):
        """Load local dataset file"""
        
        if dataset_type == "csv":
            df = pd.read_csv(path)
        elif dataset_type == "json":
            df = pd.read_json(path)
        elif dataset_type == "txt":
            with open(path, 'r') as f:
                lines = f.readlines()
            df = pd.DataFrame({"text": [l.strip() for l in lines]})
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        return df
    
    def _generate_sample_data(self):
        """Generate sample text data for testing"""
        
        sample_texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing deals with text data.",
            "Computer vision enables machines to see and understand images.",
            "Reinforcement learning trains agents through rewards and penalties.",
            "Data science combines statistics and programming to extract insights.",
            "Neural networks are inspired by the human brain structure.",
            "Backpropagation is used to train neural networks.",
            "Transformers are state-of-the-art models for NLP tasks.",
            "BERT is a pretrained language representation model.",
        ]
        
        # Repeat to create more data
        texts = sample_texts * 500  # 5s000 samples
        return pd.DataFrame({"text": texts})
    
    def _tokenize_batch(self, batch_data, model_type):
        """Tokenize a batch of text data (legacy - use _get_tokenizer for bulk)"""
        tokenizer = self._get_tokenizer(model_type)
        
        # Extract text column
        texts = batch_data['text'].tolist() if 'text' in batch_data.columns else batch_data.iloc[:, 0].tolist()
        
        # Tokenize
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        return encoded
    
    def _get_tokenizer(self, model_type):
        """Get tokenizer - cached to avoid multiple HuggingFace API calls"""
        # Use cache to avoid reloading tokenizer
        cache_key = f"tokenizer_{model_type}"
        if not hasattr(self.participant, '_tokenizer_cache'):
            self.participant._tokenizer_cache = {}
        
        if cache_key in self.participant._tokenizer_cache:
            return self.participant._tokenizer_cache[cache_key]
        
        # Suppress HuggingFace API warnings
        warnings.filterwarnings('ignore')
        
        import torch
        from transformers import AutoTokenizer
        
        tokenizer_kwargs = {
            'local_files_only': False,
            'use_fast': True,
        }
        
        if model_type == "bert":
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', **tokenizer_kwargs)
        elif model_type == "roberta":
            tokenizer = AutoTokenizer.from_pretrained('roberta-base', **tokenizer_kwargs)
        elif model_type == "distilbert":
            tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', **tokenizer_kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', **tokenizer_kwargs)
        
        # Cache the tokenizer
        self.participant._tokenizer_cache[cache_key] = tokenizer
        return tokenizer
    
    def _store_batch_locally(self, key, batch_data):
        """Store preprocessed batch locally"""
        # In production, would save to disk or cloud storage
        # For now, store in memory
        if not hasattr(self.participant, 'local_batches'):
            self.participant.local_batches = {}
        
        # Convert tensors to lists for storage (memory efficient)
        stored_data = {
            "input_ids": batch_data["input_ids"].tolist() if hasattr(batch_data["input_ids"], 'tolist') else batch_data["input_ids"],
            "attention_mask": batch_data["attention_mask"].tolist() if hasattr(batch_data["attention_mask"], 'tolist') else batch_data["attention_mask"],
        }
        
        self.participant.local_batches[key] = stored_data
    
    def handle_task_assigned(self, data):
        task_id = data.get("id")
        task_data = data.get("data")
        logger.info(f"Received assigned task {task_id}")
        # Start task in a separate thread
        task_thread = threading.Thread(
            target=self._process_task,
            args=(task_id, task_data),
            daemon=True
        )
        self.active_tasks[task_id] = task_thread
        task_thread.start()

    def handle_send_batch(self, data):
        """Handle batch request from PROC node - send batch data"""
        job_id = data.get("job_id")
        batch_num = data.get("batch_number")
        request_from = data.get("request_from")
        logger.info(f"Sending batch {batch_num} to {request_from}")
        
        # Retrieve batch from local storage
        batch_key = f"{job_id}_batch_{batch_num}"
        if hasattr(self.participant, 'local_batches') and batch_key in self.participant.local_batches:
            batch_data = self.participant.local_batches[batch_key]
            # Send batch data to the requesting PROC node via coordinator
            if self.participant.ws and self.participant.connected:
                self.participant.ws.send(json.dumps({
                    "type":          "batch_data",
                    "job_id":        job_id,
                    "batch_number":  batch_num,
                    "batch_data":    batch_data,
                    "from":          self.participant.name,
                }))
                logger.info(f"Sent batch {batch_num} to {request_from}")
        else:
            logger.warning(f"Batch {batch_num} not found in local storage")
    
    def do_prep_task(self, data):
        """PREP: Data preprocessing tasks (legacy)"""
        logger.info("PREP: Processing data preprocessing task")
        
        # Get task parameters
        dataset_path = data.get("dataset_path", "./data/input")
        output_path = data.get("output_path", "./data/preprocessed")
        preprocessing_type = data.get("preprocessing_type", "tokenize")
        
        result = {
            "type": "prep_done",
            "from": self.participant.name,
            "status": "completed",
            "preprocessing_type": preprocessing_type,
            "dataset_path": dataset_path,
            "output_path": output_path,
            "records_processed": 0
        }
        
        try:
            result["records_processed"] = self._preprocess_data(
                dataset_path, output_path, preprocessing_type
            )
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            result["status"] = "error"
            result["error"] = str(e)
        
        if self.participant.ws and self.participant.connected:
            self.participant.ws.send(json.dumps(result))
    
    def _preprocess_data(self, input_path, output_path, prep_type):
        """Perform actual data preprocessing"""
        logger.info(f"Preprocessing: {prep_type} from {input_path}")
        # This would be implemented based on specific preprocessing needs
        return 0
