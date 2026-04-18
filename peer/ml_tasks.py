# ML Tasks Module for P2P Network
# This module provides task handlers for various ML operations

import logging
import os

logger = logging.getLogger(__name__)


def task_keybert_train(participant, data):
    """
    Task: Train KeyBERT model / Extract keywords
    
    Parameters:
    - texts: list of text documents
    - model_name: sentence-transformer model name
    - num_keywords: number of keywords to extract per document
    - min_df: minimum document frequency
    """
    try:
        from keybert import KeyBERT
        from sentence_transformers import SentenceTransformer
        
        texts = data.get("texts", [])
        model_name = data.get("model_name", "paraphrase-multilingual-MiniLM-L12-v2")
        num_keywords = data.get("num_keywords", 10)
        min_df = data.get("min_df", 1)
        
        logger.info(f"Loading KeyBERT model: {model_name}")
        
        # Load model
        sentence_model = SentenceTransformer(model_name)
        kw_model = KeyBERT(model=sentence_model)
        
        # Extract keywords
        results = []
        for i, text in enumerate(texts):
            try:
                keywords = kw_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=(1, 3),
                    stop_words='english',
                    top_n=num_keywords,
                    min_df=min_df
                )
                results.append({
                    "doc_id": i,
                    "text": text[:200] + "..." if len(text) > 200 else text,
                    "keywords": keywords
                })
                logger.info(f"Processed document {i+1}/{len(texts)}")
            except Exception as e:
                logger.warning(f"Error processing doc {i}: {e}")
        
        return {
            "status": "completed",
            "from": participant.name,
            "task": "keybert_train",
            "results": results,
            "num_documents": len(texts)
        }
        
    except ImportError as e:
        logger.error(f"Missing ML library: {e}")
        return {
            "status": "error",
            "from": participant.name,
            "error": f"Missing library: {e}"
        }
    except Exception as e:
        logger.error(f"KeyBERT task error: {e}")
        return {
            "status": "error",
            "from": participant.name,
            "error": str(e)
        }


# Global tokenizer cache
_tokenizer_cache = {}


def _get_tokenizer(model_name="bert-base-uncased"):
    """Get tokenizer from cache or load and cache it"""
    if model_name not in _tokenizer_cache:
        from transformers import AutoTokenizer
        logger.info(f"Loading tokenizer: {model_name}")
        _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer_cache[model_name]


def task_data_preprocess(participant, data):
    """
    Task: Preprocess data for training
    
    Parameters:
    - dataset_path: path to input dataset
    - output_path: path to save processed data
    - preprocessing_type: type of preprocessing (tokenize, normalize, augment)
    """
    try:
        import pandas as pd
        import numpy as np
        
        dataset_path = data.get("dataset_path", "./data/input")
        output_path = data.get("output_path", "./data/output")
        preprocessing_type = data.get("preprocessing_type", "tokenize")
        
        logger.info(f"Preprocessing: {preprocessing_type}")
        
        os.makedirs(output_path, exist_ok=True)
        
        # Load data (assuming CSV format)
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
        else:
            # Create sample data for demonstration
            df = pd.DataFrame({
                "text": data.get("sample_texts", [
                    "Machine learning is a subset of artificial intelligence.",
                    "Deep learning uses neural networks with multiple layers.",
                    "Natural language processing deals with text data.",
                    "Computer vision enables machines to see and understand images.",
                    "Reinforcement learning trains agents through rewards."
                ])
            })
        
        processed_count = 0
        
        if preprocessing_type == "tokenize":
            import torch
            tokenizer = _get_tokenizer('bert-base-uncased')
            
            def tokenize_text(text):
                return tokenizer.encode(
                    text, 
                    padding='max_length', 
                    truncation=True, 
                    max_length=512
                )
                
                df['tokens'] = df['text'].apply(tokenize_text)
                processed_count = len(df)
                            
        elif preprocessing_type == "normalize":
            # Simple normalization
            df['text_normalized'] = df['text'].str.lower().str.strip()
            processed_count = len(df)
            
        elif preprocessing_type == "augment":
            # Simple augmentation (add noise)
            import random
            def augment_text(text):
                words = text.split()
                if len(words) > 3:
                    # Randomly duplicate a word
                    idx = random.randint(0, len(words)-1)
                    words.insert(idx, words[idx])
                return ' '.join(words)
            
            df['text_augmented'] = df['text'].apply(augment_text)
            processed_count = len(df)
        
        # Save processed data
        output_file = os.path.join(output_path, "processed.parquet")
        df.to_parquet(output_file)
        
        return {
            "status": "completed",
            "from": participant.name,
            "task": "data_preprocess",
            "preprocessing_type": preprocessing_type,
            "records_processed": processed_count,
            "output_file": output_file
        }
        
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return {
            "status": "error",
            "from": participant.name,
            "error": str(e)
        }


def task_model_train(participant, data):
    """
    Task: Train a neural network model
    
    Parameters:
    - model_type: type of model (bert, lstm, cnn)
    - dataset_path: path to training data
    - epochs: number of training epochs
    - batch_size: batch size for training
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        
        model_type = data.get("model_type", "bert")
        epochs = data.get("epochs", 10)
        batch_size = data.get("batch_size", 32)
        
        logger.info(f"Training {model_type} model for {epochs} epochs")
        
        # For demonstration, use simple synthetic data
        # In production, load from dataset_path
        num_samples = 1000
        input_dim = 768  # BERT hidden size
        num_classes = 10
        
        # Create synthetic data
        X = torch.randn(num_samples, input_dim)
        y = torch.randint(0, num_classes, (num_samples,))
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Simple classifier
        model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        total_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            total_loss += avg_loss
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        avg_total_loss = total_loss / epochs
        
        return {
            "status": "completed",
            "from": participant.name,
            "task": "model_train",
            "model_type": model_type,
            "epochs": epochs,
            "final_loss": avg_total_loss,
            "samples_processed": num_samples
        }
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        return {
            "status": "error",
            "from": participant.name,
            "error": str(e)
        }


def task_aggregate_weights(participant, data):
    """
    Task: Aggregate model weights from multiple clients (Federated Learning)
    
    Parameters:
    - client_weights: list of weight dictionaries from clients
    - aggregation_strategy: strategy for aggregation (fedavg, fedprox, etc.)
    """
    try:
        import torch
        
        client_weights = data.get("client_weights", [])
        strategy = data.get("aggregation_strategy", "fedavg")
        
        logger.info(f"Aggregating weights from {len(client_weights)} clients")
        logger.info(f"Strategy: {strategy}")
        
        if not client_weights:
            return {
                "status": "error",
                "from": participant.name,
                "error": "No client weights provided"
            }
        
        aggregated = None
        num_clients = len(client_weights)
        
        if strategy == "fedavg":
            # Federated Averaging
            for weights in client_weights:
                if aggregated is None:
                    aggregated = {}
                    for key, value in weights.items():
                        if isinstance(value, torch.Tensor):
                            aggregated[key] = value.clone()
                        else:
                            aggregated[key] = value
                else:
                    for key in aggregated.keys():
                        if key in weights:
                            if isinstance(aggregated[key], torch.Tensor):
                                aggregated[key] += weights[key]
            
            # Average
            if aggregated:
                for key in aggregated.keys():
                    if isinstance(aggregated[key], torch.Tensor):
                        aggregated[key] = aggregated[key] / num_clients
        
        # Convert tensors to lists for JSON serialization
        result_weights = {}
        for key, value in aggregated.items():
            if isinstance(value, torch.Tensor):
                result_weights[key] = value.cpu().detach().numpy().tolist()
            else:
                result_weights[key] = value
        
        return {
            "status": "completed",
            "from": participant.name,
            "task": "aggregate_weights",
            "strategy": strategy,
            "num_clients": num_clients,
            "aggregated_keys": list(aggregated.keys()) if aggregated else []
        }
        
    except Exception as e:
        logger.error(f"Aggregation error: {e}")
        return {
            "status": "error",
            "from": participant.name,
            "error": str(e)
        }


def task_validate_model(participant, data):
    """
    Task: Validate a trained model
    
    Parameters:
    - model_path: path to model weights
    - dataset_path: path to validation data
    - metrics: list of metrics to compute
    """
    try:
        import torch
        
        model_path = data.get("model_path")
        metrics = data.get("metrics", ["accuracy", "f1"])
        
        logger.info(f"Validating model: {model_path}")
        
        # For demonstration, return mock results
        results = {
            "accuracy": 0.85,
            "f1_score": 0.82,
            "precision": 0.84,
            "recall": 0.83
        }
        
        return {
            "status": "completed",
            "from": participant.name,
            "task": "validate_model",
            "metrics": results,
            "loss_threshold": data.get("loss_threshold", 0.1)
        }
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return {
            "status": "error",
            "from": participant.name,
            "error": str(e)
        }


# Task registry - maps task names to handler functions
TASK_HANDLERS = {
    "keybert_train": task_keybert_train,
    "data_preprocess": task_data_preprocess,
    "model_train": task_model_train,
    "aggregate_weights": task_aggregate_weights,
    "validate_model": task_validate_model,
}


def execute_task(participant, task_name, data):
    """Execute a registered task"""
    if task_name in TASK_HANDLERS:
        return TASK_HANDLERS[task_name](participant, data)
    else:
        logger.warning(f"Unknown task: {task_name}")
        return {
            "status": "error",
            "from": participant.name,
            "error": f"Unknown task: {task_name}"
        }