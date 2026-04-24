"""
PREP (Preprocessing) node functionality for P2P Network with Federated Learning
Handles data preprocessing: loading datasets, tokenizing, encoding, and storing batches locally
"""

import json
import logging
import pandas as pd
import warnings
import threading
import time
import requests
import torch

logger = logging.getLogger(__name__)


class PrepNode:
    """PREP node handles dataset preprocessing and batch storage"""
    
    # Available encoders
    ENCODERS = {
        "bert": "bert-base-uncased",
        "roberta": "roberta-base",
        "distilbert": "distilbert-base-uncased",
        "sentence-transformer": "paraphrase-multilingual-MiniLM-L12-v2",
    }
    
    # 🔹 Расширенный список возможных имён колонок с метками
    LABEL_COLUMNS = ['label', 'labels', 'target', 'targets', 'class', 'classes', 
                     'y', 'category', 'sentiment', 'score', 'rating', 'outcome']
    
    def __init__(self, participant):
        self.participant = participant
        self.polling_active = False
        self.polling_thread = None
        self.active_tasks = {}
        self.poll_interval = 5
        self._encoder_cache = {}

    def start_polling(self):
        if self.polling_active:
            return
        self.polling_active = True
        self.polling_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.polling_thread.start()
        logger.info("Started polling for preprocessing tasks")

    def stop_polling(self):
        self.polling_active = False
        if self.polling_thread and self.polling_thread.is_alive():
            self.polling_thread.join(timeout=5)
        logger.info("Stopped polling for preprocessing tasks")

    def _poll_loop(self):
        while self.polling_active and self.participant.running:
            try:
                self._poll_for_task()
            except Exception as e:
                logger.warning(f"Poll error: {e}")
            time.sleep(self.poll_interval)

    def _poll_for_task(self):
        """Poll coordinator for preprocessing tasks"""
        if not self.participant.ws or not self.participant.connected:
            return
        
        try:
            self.participant.ws.send(json.dumps({
                "type": "poll",
                "from": self.participant.name,
                "role": "PREP"
            }))
        except Exception as e:
            logger.warning(f"Poll send error: {e}")

    def _process_task(self, task_id, task_data):
        try:
            logger.info(f"Processing task {task_id}")
            self.do_preprocess_task(task_data)
            self._report_task_completed(task_id)
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            self._report_task_error(task_id, str(e))
        finally:
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]

    def _report_task_completed(self, task_id):
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
        """PREP: Preprocess dataset - load, tokenize, encode, and store batches"""
        job_id = data.get("job_id")
        dataset_url = data.get("dataset_url", "")
        dataset_type = data.get("dataset_type", "csv")
        batch_size = data.get("batch_size", 32)
        start_offset = data.get("start_offset", 0)
        end_offset = data.get("end_offset", 1000)
        model_name = data.get("model_name", "bert-base-uncased")
        encoder_name = data.get("encoder_name", "bert")
        task_id = data.get("task_id")

        self.participant.current_job_id = job_id
        self.participant.training_active = True

        logger.info(f"PREP: Starting task {task_id} for job {job_id}")
        logger.info(f"  Dataset: {dataset_url}, Model: {model_name}, Encoder: {encoder_name}")
        logger.info(f"  Records: {start_offset}-{end_offset}, Batch size: {batch_size}")

        try:
            total_records_needed = end_offset - start_offset
            total_batches = (total_records_needed + batch_size - 1) // batch_size

            logger.info(f"  Loading {total_records_needed} records from offset={start_offset}")
            dataset = self._load_dataset(dataset_url, dataset_type, offset=start_offset, length=total_records_needed)
            total_records = len(dataset)
            logger.info(f"  Loaded {total_records} records")
            
            # 🔹 ОТЛАДКА: Выводим все колонки датасета
            logger.info(f"  Dataset columns: {list(dataset.columns)}")
            if len(dataset) > 0:
                logger.info(f"  Dataset sample (first row): {dataset.iloc[0].to_dict()}")
            
            # 🔹 Поиск колонки с метками
            label_column = self._find_label_column(dataset)
            logger.info(f"  Label column found: {label_column}")
            
            logger.info("  Loading tokenizer...")
            tokenizer = self._get_tokenizer(model_name)
            
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
            
            logger.info(f"  Loading encoder: {encoder_name}")
            encoder = self._get_encoder(encoder_name)
            encoder.eval()
            
            for i in range(total_batches):
                if not self.participant.training_active:
                    logger.info("Preprocessing stopped")
                    break

                batch_num = i
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(texts))
                
                if start_idx >= len(texts):
                    break
                
                input_ids_data = encoded["input_ids"]
                attention_mask_data = encoded["attention_mask"]

                if hasattr(input_ids_data, 'tolist'):
                    batch_input_ids = input_ids_data[start_idx:end_idx]
                    batch_attention_mask = attention_mask_data[start_idx:end_idx]
                else:
                    batch_input_ids = torch.tensor(input_ids_data[start_idx:end_idx])
                    batch_attention_mask = torch.tensor(attention_mask_data[start_idx:end_idx])
                
                logger.info(f"  Encoding batch {batch_num}...")
                with torch.no_grad():
                    embeddings = encoder(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask
                    )
                    if hasattr(embeddings, 'pooler_output') and embeddings.pooler_output is not None:
                        batch_embeddings = embeddings.pooler_output
                    elif hasattr(embeddings, 'last_hidden_state'):
                        batch_embeddings = embeddings.last_hidden_state.mean(dim=1)
                    else:
                        batch_embeddings = embeddings[0]
                
                batch_key = f"{job_id}_batch_{batch_num}"
                
                # 🔹 Извлечение меток через универсальный метод
                batch_labels = self._extract_labels(dataset, label_column, start_idx, end_idx)
                
                # 🔹 Отладка: первые метки батча
                if i == 0:
                    logger.info(f"  First batch labels sample: {batch_labels[:10]}")
                    unique = list(set(batch_labels))
                    logger.info(f"  Unique labels in batch: {unique}")
                
                stored_data = {
                    "embeddings": batch_embeddings.cpu().tolist(),
                    "attention_mask": batch_attention_mask.cpu().tolist(),
                    "input_ids": batch_input_ids.cpu().tolist(),
                    "labels": batch_labels,
                }
                self._store_batch_locally(batch_key, stored_data)
                
                progress = (i + 1) / total_batches * 100
                self.participant._report_batch_progress(job_id, batch_num, "ready", progress, end_idx - start_idx)
                logger.info(f"  Batch {batch_num} ready ({progress:.1f}%)")
            
            logger.info(f"PREP: Completed preprocessing for job {job_id}")
            
            # 🔹 После завершения — сразу поллим координатор на новые задачи!
            self._poll_for_task()
            
        except Exception as e:
            logger.error(f"PREP error: {e}")
            import traceback
            traceback.print_exc()
            if self.participant.ws and self.participant.connected:
                self.participant.ws.send(json.dumps({
                    "type": "preprocess_error",
                    "from": self.participant.name,
                    "job_id": job_id,
                    "error": str(e)
                }))
    
    def _find_label_column(self, dataset):
        """Находит колонку с метками в датасете"""
        # 🔹 Сначала проверяем известные имена — берём без фильтра по уникальным значениям!
        for col in self.LABEL_COLUMNS:
            if col in dataset.columns:
                unique_vals = dataset[col].dropna().unique()
                logger.info(f"  Found label column '{col}' with {len(unique_vals)} unique values: {sorted(unique_vals)[:10]}...")
                return col
        
        # 🔹 Эвристика для неизвестных имён: ищем числовые колонки с дискретными значениями
        numeric_cols = dataset.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        for col in numeric_cols:
            if col.lower() not in ['id', 'index', 'idx', 'count', 'len', 'row_idx']:
                unique_vals = dataset[col].dropna().unique()
                if 1 < len(unique_vals) <= 500:  # Увеличили лимит для гибкости
                    logger.info(f"  Guessed label column: {col} (unique values: {len(unique_vals)}, sample: {unique_vals[:10]})")
                    return col
        
        return None
    
    def _extract_labels(self, dataset, label_column, start_idx, end_idx):
        """Извлекает метки из датасета"""
        if label_column and label_column in dataset.columns:
            labels = dataset[label_column].iloc[start_idx:end_idx].tolist()
            return [int(l) if isinstance(l, (int, float, str)) and str(l).isdigit() else l for l in labels]
        else:
            logger.warning(f"  No label column found! Using default label 0")
            return [0] * (end_idx - start_idx)
    
    def _load_dataset(self, url, dataset_type, offset=0, length=None):
        if "huggingface.co/datasets" in url:
            return self._load_huggingface_dataset(url, dataset_type, offset=offset, length=length)
        
        if url.startswith("http://") or url.startswith("https://"):
            logger.info(f"Downloading dataset from {url}...")
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                filename = url.split('/')[-1].split('?')[0]
                temp_path = f"/tmp/{filename}"
                with open(temp_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Downloaded to {temp_path}")
                return self._load_local_file(temp_path, dataset_type)
            except Exception as e:
                logger.warning(f"Failed to download dataset: {e}, using sample data")
        
        import os
        if os.path.exists(url):
            return self._load_local_file(url, dataset_type)
        
        logger.info("Using sample data for testing")
        return self._generate_sample_data()
    
    def _load_huggingface_rows(self, dataset_path, offset=0, length=100):
        """
        Загружает строки из HuggingFace Datasets Server API с пагинацией.
        🔹 Защита от 429: задержки между запросами + автоматический повтор.
        """
        MAX_PER_REQUEST = 100
        REQUEST_DELAY = 1.5        # Задержка между успешными запросами (сек)
        RATE_LIMIT_WAIT = 10.0     # Задержка при получении 429 (сек)
        MAX_RETRIES = 2            # Максимум повторов при ошибке сети

        all_rows = []
        remaining = length
        current_offset = offset

        logger.info(f"Loading HF  offset={offset}, length={length} (paginated, max {MAX_PER_REQUEST}/req)")

        while remaining > 0:
            batch_size = min(remaining, MAX_PER_REQUEST)
            rows_url = (
                f"https://datasets-server.huggingface.co/rows"
                f"?dataset={dataset_path}"
                f"&config=default"
                f"&split=train"
                f"&offset={current_offset}"
                f"&length={batch_size}"
            )

            success = False
            for attempt in range(MAX_RETRIES + 1):
                try:
                    resp = requests.get(rows_url, timeout=30)
                    
                    # 🔹 Обработка 429 Too Many Requests
                    if resp.status_code == 429:
                        wait = RATE_LIMIT_WAIT * (attempt + 1)
                        logger.warning(f"⏳ Rate limit (429). Waiting {wait}s before retry {attempt+1}/{MAX_RETRIES+1}...")
                        time.sleep(wait)
                        continue
                    
                    resp.raise_for_status()
                    data = resp.json()
                    rows = data.get("rows", [])
                    success = True
                    break
                    
                except requests.RequestException as e:
                    logger.error(f"Network error at offset {current_offset}: {e}")
                    if attempt < MAX_RETRIES:
                        wait = 2 ** attempt
                        logger.warning(f"Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        raise

            if not success:
                logger.error(f"Failed to load batch at offset {current_offset} after {MAX_RETRIES} retries")
                break

            if not rows:
                logger.warning(f"No more rows at offset {current_offset}, stopping pagination")
                break

            # Извлекаем только данные строки
            for row in rows:
                all_rows.append(row["row"])

            loaded = len(rows)
            remaining -= loaded
            current_offset += loaded
            logger.debug(f"  Loaded {loaded} rows (offset {current_offset - loaded}), remaining: {remaining}")

            # 🔹 Задержка перед следующим запросом, чтобы не получить 429
            if remaining > 0:
                time.sleep(REQUEST_DELAY)

        if not all_rows:
            raise ValueError(f"No data loaded from HF dataset '{dataset_path}' at offset {offset}")

        logger.info(f"Total loaded: {len(all_rows)} rows from HF API")
        return pd.DataFrame(all_rows)
    
    def _load_huggingface_dataset(self, url, dataset_type, offset=0, length=100):
        if "huggingface.co/datasets/" in url:
            dataset_path = url.split("huggingface.co/datasets/")[1].split("?")[0]
        else:
            dataset_path = url
        return self._load_huggingface_rows(dataset_path, offset, length)
    
    def _load_local_file(self, path, dataset_type):
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
        sample_texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing deals with text data.",
            "Computer vision enables machines to see and understand images.",
            "Reinforcement learning trains agents through rewards and penalties.",
        ]
        texts = sample_texts * 500
        labels = [i % 2 for i in range(len(texts))]
        return pd.DataFrame({"text": texts, "label": labels})
    
    def _get_tokenizer(self, model_name):
        cache_key = f"tokenizer_{model_name}"
        if not hasattr(self.participant, '_tokenizer_cache'):
            self.participant._tokenizer_cache = {}
        
        if cache_key in self.participant._tokenizer_cache:
            return self.participant._tokenizer_cache[cache_key]
        
        warnings.filterwarnings('ignore')
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.participant._tokenizer_cache[cache_key] = tokenizer
        return tokenizer
    
    def _get_encoder(self, encoder_name):
        """Get encoder model - cached to avoid reloading"""
        if encoder_name in self._encoder_cache:
            return self._encoder_cache[encoder_name]
        
        warnings.filterwarnings('ignore')
        from transformers import AutoModel
        
        if encoder_name in self.ENCODERS:
            model_name = self.ENCODERS[encoder_name]
        else:
            model_name = encoder_name
        
        logger.info(f"Loading encoder model: {model_name}")
        encoder = AutoModel.from_pretrained(model_name)
        encoder.eval()
        
        self._encoder_cache[encoder_name] = encoder
        return encoder
    
    def _store_batch_locally(self, key, batch_data):
        if not hasattr(self.participant, 'local_batches'):
            self.participant.local_batches = {}
        self.participant.local_batches[key] = batch_data

    def handle_task_assigned(self, data):
        task_id = data.get("id")
        task_data = data.get("data")
        logger.info(f"Received assigned task {task_id}")
        task_thread = threading.Thread(
            target=self._process_task,
            args=(task_id, task_data),
            daemon=True
        )
        self.active_tasks[task_id] = task_thread
        task_thread.start()

    def handle_send_batch(self, data):
        job_id = data.get("job_id")
        batch_num = data.get("batch_number")
        request_from = data.get("request_from")
        logger.info(f"Sending batch {batch_num} to {request_from}")

        batch_key = f"{job_id}_batch_{batch_num}"
        if hasattr(self.participant, 'local_batches') and batch_key in self.participant.local_batches:
            batch_data = self.participant.local_batches[batch_key]
            if self.participant.ws and self.participant.connected:
                self.participant.ws.send(json.dumps({
                    "type": "batch_data",
                    "job_id": job_id,
                    "batch_number": batch_num,
                    "batch_data": batch_data,
                    "from": self.participant.name,
                }))
                logger.info(f"Sent batch {batch_num} to {request_from}")
        else:
            logger.warning(f"Batch {batch_num} not found in local storage")
