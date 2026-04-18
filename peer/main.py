# #!/usr/bin/env python3
# """
# Main entry point for P2P Network participant
# Coordinates PREP and PROC nodes based on role assigned by coordinator
# """

# import os
# import sys
# import json
# import socket
# import logging
# import platform
# import getpass
# import requests
# import websocket
# import threading
# import time
# import psutil
# import subprocess
# import signal

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# # Import role-specific handlers
# from prep import PrepNode
# from proc import ProcNode


# class Participant:
#     def __init__(self, coordinator_url: str, role: str = None):
#         self.coordinator_url = coordinator_url.rstrip('/')
#         self.role = role  # PREP or PROC - set by coordinator
#         self.data_dir = "./data"
#         self.name = self._generate_name()
#         self.ws = None
#         self.connected = False
#         self.running = True
#         self.training_active = False  # For federated training
        
#         # Current training job info
#         self.current_job_id = None
#         self.current_round = 0
        
#         # ML modules - loaded dynamically
#         self.ml_module = None
        
#         # Direct peer connections (for P2P messaging)
#         self.direct_peers = {}  # peer_name -> websocket
#         self.peer_ports = {}    # peer_name -> port for direct connections
        
#         # System info (without ML libs)
#         self.system_info = self._get_system_info()
        
#         # Initialize role-specific handlers
#         self.prep_node = PrepNode(self)
#         self.proc_node = ProcNode(self)
        
#     def _generate_name(self) -> str:
#         hostname = socket.gethostname()
#         os_name = platform.system().lower()
#         username = getpass.getuser()
#         return f"{hostname}-{os_name}-{username}"
    
#     def _get_system_info(self) -> dict:
#         info = {
#             "hostname": socket.gethostname(),
#             "os": platform.system(),
#             "arch": platform.machine(),
#             "cpu": self._get_cpu(),
#             "cpu_cores": psutil.cpu_count(),
#             "memory": self._get_memory(),
#             "disk": self._get_disk(),
#             "gpu": self._get_gpu(),
#         }
        
#         # Also add GPU info at top level for easier access
#         try:
#             gpu_info = json.loads(info["gpu"])
#             info["vram_gb"] = gpu_info.get("vram_gb", 0)
#             info["cuda_available"] = gpu_info.get("cuda_available", False)
#         except:
#             info["vram_gb"] = 0
#             info["cuda_available"] = False
        
#         return info
    
#     def _get_cpu(self) -> float:
#         try:
#             if platform.system() == "Linux":
#                 result = subprocess.run(
#                     ["bash", "-c", "top -bn1 | grep 'Cpu(s)' | sed 's/.*, *\\([0-9.]*\\)%.*.*/\\1/' | awk '{print 100 - $1}'"],
#                     capture_output=True, text=True, timeout=5
#                 )
#                 return float(result.stdout.strip()) if result.stdout.strip() else 0
#         except:
#             pass
#         return 0
    
#     def _get_memory(self) -> float:
#         try:
#             mem = psutil.virtual_memory()
#             return mem.percent
#         except:
#             return 0
    
#     def _get_disk(self) -> float:
#         try:
#             return psutil.disk_usage('/').percent
#         except:
#             return 0
    
#     def _get_gpu(self) -> str:
#         """Detect GPU using nvidia-smi first (for CUDA), then lspci as fallback"""
#         gpu_info = {"name": "No GPU", "vram_gb": 0, "cuda_available": False}
        
#         # Try nvidia-smi first - provides accurate GPU info including VRAM
#         try:
#             result = subprocess.run(
#                 ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
#                 capture_output=True, text=True, timeout=5
#             )
#             if result.returncode == 0 and result.stdout.strip():
#                 lines = result.stdout.strip().split('\n')
#                 if lines:
#                     parts = lines[0].split(',')
#                     gpu_name = parts[0].strip()
#                     vram_mb = int(parts[1].strip()) if len(parts) > 1 else 0
#                     gpu_info["name"] = gpu_name
#                     gpu_info["vram_gb"] = vram_mb / 1024.0
#                     gpu_info["cuda_available"] = True
#                     return json.dumps(gpu_info)
#         except Exception as e:
#                     return 0
        
#         # Fallback to lspci
#         try:
#             result = subprocess.run(
#                 ["bash", "-c", "lspci 2>/dev/null | grep -i vga"],
#                 capture_output=True, text=True, timeout=5
#             )
#             if result.stdout:
#                 gpu = result.stdout.strip()
#                 logger.info(f"GPU detected (lspci): {gpu}")
#                 gpu_info["name"] = gpu
#                 return json.dumps(gpu_info)
#         except:
#             pass
        
#         return json.dumps(gpu_info)
    
#     def _get_local_ip(self) -> str:
#         try:
#             s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#             s.connect(("8.8.8.8", 80))
#             ip = s.getsockname()[0]
#             s.close()
#             return ip
#         except:
#             return "127.0.0.1"
    
#     def _init_ml_environment(self):
#         """Load ML libraries dynamically when assigned PROC role"""
#         logger.info("PROC role assigned - initializing ML environment...")
        
#         try:
#             import torch
#             logger.info(f"Torch version: {torch.__version__}")
            
#             # Update system info with GPU details
#             if torch.cuda.is_available():
#                 self.system_info["gpu"] = torch.cuda.get_device_name(0)
#                 self.system_info["cuda"] = True
#                 self.system_info["cuda_version"] = torch.version.cuda
#             else:
#                 self.system_info["gpu"] = "CPU only"
#                 self.system_info["cuda"] = False
            
#             self._init_ml_module()
            
#             return True
            
#         except ImportError as e:
#             logger.warning(f"ML libraries not installed: {e}")
#             logger.info("Installing ML libraries...")
            
#             # Install required packages
#             packages = [
#                 "torch>=2.0.0",
#                 "transformers>=4.30.0", 
#                 "sentence-transformers>=2.2.0",
#                 "keybert>=0.7.0",
#                 "scikit-learn>=1.3.0",
#                 "numpy>=1.24.0",
#                 "pandas>=2.0.0"
#             ]
            
#             for pkg in packages:
#                 try:
#                     result = subprocess.run(
#                         [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
#                         capture_output=True, text=True, timeout=300
#                     )
#                     if result.returncode != 0:
#                         logger.warning(f"Failed to install {pkg}: {result.stderr}")
#                 except Exception as e:
#                     logger.warning(f"Error installing {pkg}: {e}")
            
#             # Try importing again
#             try:
#                 import torch
#                 if torch.cuda.is_available():
#                     self.system_info["gpu"] = torch.cuda.get_device_name(0)
#                     self.system_info["cuda"] = True
#                 self._init_ml_module()
#                 return True
#             except ImportError:
#                 logger.error("Failed to load ML libraries after installation")
#                 return False
    
#     def _init_ml_module(self):
#         """Initialize ML module after torch is loaded"""
#         try:
#             # Import our ML task handlers
#             from . import ml_tasks
#             self.ml_module = ml_tasks
#             logger.info("ML module initialized")
#         except ImportError:
#             logger.info("ML module not available - using built-in handlers")
    
#     def register(self) -> bool:
#         data = {
#             "name": self.name,
#             "address": self._get_local_ip(),
#             "port": 0,
#             "status": "online",
#             "system": self.system_info
#         }
        
#         try:
#             resp = requests.post(
#                 f"{self.coordinator_url}/api/register",
#                 json=data,
#                 timeout=10
#             )
#             if resp.status_code == 200:
#                 result = resp.json()
#                 if 'participant' in result:
#                     self.role = result['participant'].get('role', 'PREP')
#                 logger.info(f"Registered: {self.name}")
#                 logger.info(f"Role: {self.role}")
                
#                 # Initialize ML environment if assigned PROC role
#                 if self.role == "PROC":
#                     self._init_ml_environment()
                
#                 return True
#             return False
#         except Exception as e:
#             logger.error(f"Registration error: {e}")
#             return False
    
#     def connect_websocket(self):
#         ws_url = self.coordinator_url.replace("http", "ws") + f"/ws?name={self.name}"
#         logger.info(f"Connecting to {ws_url}")
        
#         while self.running:
#             try:
#                 self.ws = websocket.WebSocketApp(
#                     ws_url,
#                     on_open=self._on_ws_open,
#                     on_message=self._on_ws_message,
#                     on_error=self._on_ws_error,
#                     on_close=self._on_ws_close
#                 )
#                 self.ws.run_forever(ping_interval=30, ping_timeout=10)
#             except websocket.WebSocketBadStatusException as e:
#                 # Connection rejected (e.g., server not running)
#                 logger.error(f"WebSocket connection rejected: {e.status_code}")
#             except Exception as e:
#                 logger.error(f"WebSocket error: {e}")
            
#             if self.running:
#                 logger.info("Reconnecting in 5 seconds...")
#                 time.sleep(5)
    
#     def _on_ws_open(self, ws):
#         self.connected = True
#         logger.info("Connected to coordinator")
        
#         # Removed: using only WebSocket poll, not HTTP poll
#         # self._poll_for_tasks()
    
#     def _poll_for_tasks(self):
#         """Poll coordinator for pending tasks"""
#         if not self.role:
#             logger.warning("No role assigned, skipping poll")
#             return
        
#         try:
#             resp = requests.get(
#                 f"{self.coordinator_url}/api/poll/tasks",
#                 params={"name": self.name, "role": self.role},
#                 timeout=10
#             )
#             if resp.status_code == 200:
#                 result = resp.json()
#                 tasks = result.get("tasks", [])
#                 if tasks:
#                     task = tasks[0]
#                     task_data = task.get("data", {})
#                     # Add job_id from task to data (poll response has it at top level)
#                     task_data["job_id"] = task.get("job_id")
#                     task_data["job_name"] = task.get("job_name")
                    
#                     logger.info(f"Received task from poll: {task.get('task_type')} for job {task.get('job_id')}")
                    
#                     # Process the task
#                     if task.get("task_type") == "preprocess":
#                         self.prep_node.do_preprocess_task(task_data)
#                     elif task.get("task_type") == "train":
#                         self.proc_node.do_federated_train_task(task_data)
#                 else:
#                     logger.debug("No pending tasks in poll")
#             else:
#                 logger.warning(f"Poll failed with status {resp.status_code}")
#         except Exception as e:
#             logger.warning(f"Poll error: {e}")
    
#     def _on_ws_message(self, ws, message):
#         try:
#             data = json.loads(message)
#             msg_type = data.get("type")
            
#             if msg_type == "role_assigned":
#                 new_role = data.get("role")
#                 if new_role == "PROC" and self.role != "PROC":
#                     logger.info("Assigned PROC role - initializing ML environment")
#                     self._init_ml_environment()
#                 self.role = new_role
#                 logger.info(f"Role: {self.role}")
            
#             elif msg_type == "ping":
#                 ws.send(json.dumps({"type": "pong", "from": self.name}))
            
#             elif msg_type == "poll_ack":
#                 logger.debug("Received poll_ack from coordinator")
            
#             elif msg_type == "poll_response":
#                 has_tasks = data.get("has_tasks", False)
#                 task_count = data.get("task_count", 0)
#                 if has_tasks:
#                     logger.info(f"Coordinator has {task_count} pending tasks - requesting task details")
#                     # Request task details from coordinator
#                     self.ws.send(json.dumps({
#                         "type": "get_task",
#                         "from": self.name,
#                         "role": self.role
#                     }))
#                 else:
#                     logger.debug("No pending tasks")
            
#             elif msg_type == "task_details":
#                 # Received task details from coordinator
#                 task = data.get("task", {})
#                 if task:
#                     task_type = task.get("task_type", "unknown")
#                     logger.info(f"Received task: {task_type}")
#                 else:
#                     logger.debug("No tasks available")
            
#             elif msg_type == "message":
#                 msg_from = data.get("from", "unknown")
#                 content = data.get("content", "")
#                 print(f"\n[MESSAGE] From: {msg_from} | Content: {content}\n> ")
            
#             elif msg_type == "task_prep":
#                 logger.info("PREP task: data preprocessing")
#                 self.prep_node.do_prep_task(data)

#             elif msg_type == "task_preprocess":
#                 logger.info("PREP task: preprocess dataset for federated learning")
#                 self.prep_node.do_preprocess_task(data)

#             elif msg_type == "send_batch":
#                 # Handle batch request from PROC node - send batch data
#                 self.prep_node.handle_send_batch(data)

#             elif msg_type == "task_train":
#                 logger.info("PROC task: train model from scratch (federated)")
#                 self.proc_node.do_federated_train_task(data)

#             elif msg_type == "batch_sources":
#                 # Store batch source info (which PREP node has which batch)
#                 self.proc_node.handle_batch_sources(data)

#             elif msg_type == "batch_data":
#                 # Receive batch data from PREP node
#                 self.proc_node.handle_batch_data(data)

#             elif msg_type == "training_stop":
#                 job_id = data.get("job_id")
#                 logger.info(f"Training stop requested for job {job_id}")
#                 self.training_active = False

#             elif msg_type == "training_complete":
#                 job_id = data.get("job_id")
#                 final_loss = data.get("final_loss", 0)
#                 logger.info(f"Training complete! Job {job_id}, final loss: {final_loss}")
#                 self.training_active = False
            
#             elif msg_type == "task_keybert":
#                 logger.info("PROC task: KeyBERT training")
#                 self.proc_node.do_keybert_task(data)
            
#             elif msg_type == "task_custom":
#                 # Generic task handler - uses task_name to dispatch
#                 task_name = data.get("task_name", "unknown")
#                 logger.info(f"Custom task: {task_name}")
#                 self._do_custom_task(data)
            
#             elif msg_type == "p2p_message":
#                 # P2P message from another peer (via coordinator)
#                 msg_from = data.get("from", "unknown")
#                 content = data.get("content", "")
#                 print(f"\n[P2P] From: {msg_from} | Content: {content}\n> ")
            
#             elif msg_type == "peer_info":
#                 # Response to get_peer_info request
#                 peer_name = data.get("peer_name")
#                 peer_ip = data.get("ip")
#                 peer_port = data.get("port")
#                 if peer_port:
#                     self.peer_ports[peer_name] = {"ip": peer_ip, "port": peer_port}
#                     print(f"Peer info for {peer_name}: {peer_ip}:{peer_port}")
#                 else:
#                     print(f"Peer {peer_name} not available for direct connection")
            
#             elif msg_type == "participants_list":
#                 # Updated participant list
#                 participants = data.get("participants", [])
#                 print(f"\n{'='*50}")
#                 print("PARTICIPANTS:")
#                 print(f"{'='*50}")
#                 for p in participants:
#                     status = "✓ online" if p.get("online") else "✗ offline"
#                     role = p.get("role", "?")
#                     print(f"  {p.get('name', '?')} | {role} | {status}")
#                 print(f"{'='*50}\n")
            
#             elif msg_type == "delivered":
#                 # Message delivery confirmation
#                 content = data.get("content", "")
#                 print(f"[Message delivered] {content[:50]}...")
            
#             elif msg_type == "error":
#                 # Error message
#                 error_msg = data.get("message", "Unknown error")
#                 print(f"\n[ERROR] {error_msg}\n> ")
            
#             else:
#                 logger.info(f"Received: {msg_type}")
                
#         except Exception as e:
#             logger.error(f"Message error: {e}")
    
#     def _on_ws_error(self, ws, error):
#         logger.error(f"WS error: {error}")
    
#     def _on_ws_close(self, ws, code, msg):
#         self.connected = False
#         logger.info(f"Disconnected: {code}")
    
#     def _report_batch_progress(self, job_id, batch_num, status, progress, record_count):
#         """Report batch progress to coordinator via WebSocket only"""
#         # Report via WebSocket (more reliable)
#         if self.ws and self.connected:
#             try:
#                 self.ws.send(json.dumps({
#                     "type": "batch_progress",
#                     "from": self.name,
#                     "job_id": job_id,
#                     "batch_number": batch_num,
#                     "status": status,
#                     "progress": progress,
#                     "record_count": record_count
#                 }))
#             except Exception as e:
#                 logger.warning(f"WS batch progress failed: {e}")

#     def _request_batch_from_peer(self, job_id, batch_number, peer_name, peer_ip=None):
#         """Request a specific batch from a PREP peer node"""
#         if self.ws and self.connected:
#             try:
#                 self.ws.send(json.dumps({
#                     "type": "request_batch",
#                     "from": self.name,
#                     "job_id": job_id,
#                     "batch_number": batch_number,
#                     "target_peer": peer_name,
#                     "target_peer_ip": peer_ip,  # IP for direct P2P connection
#                 }))
#                 logger.info(f"Requested batch {batch_number} from {peer_name} ({peer_ip})")
#             except Exception as e:
#                 logger.error(f"Failed to request batch: {e}")
    
#     def _send_model_update(self, job_id, round_num, weights, loss, accuracy):
#         """Send model weights delta to coordinator"""
#         try:
#             requests.post(
#                 f"{self.coordinator_url}/api/model/update",
#                 json={
#                     "job_id": job_id,
#                     "round": round_num,
#                     "from": self.name,
#                     "status": "sent",
#                     "weights": weights,  # In production, would compress this
#                     "loss": loss,
#                     "accuracy": accuracy,
#                     "sample_count": len(weights)
#                 },
#                 timeout=30
#             )
#             logger.info(f"Sent model update to coordinator (loss: {loss:.4f})")
#         except Exception as e:
#             logger.error(f"Failed to send model update: {e}")
    
#     def _send_training_update(self, status, data):
#         """Send training status update"""
#         if self.ws and self.connected:
#             self.ws.send(json.dumps({
#                 "type": "training_update",
#                 "from": self.name,
#                 "job_id": self.current_job_id,
#                 "round": self.current_round,
#                 "status": status,
#                 "data": data
#             }))
    
#     def _send_task_result(self, result_type, data):
#         """Send task result back to coordinator"""
#         if self.ws and self.connected:
#             data["type"] = result_type
#             data["from"] = self.name
#             self.ws.send(json.dumps(data))
    
#     def _do_custom_task(self, data):
#         """Handle custom tasks from registry"""
#         # This can be extended to support custom tasks
#         logger.info("Custom task received")
    
#     def poll_tasks(self):
#         while self.running:
#             time.sleep(10)
            
#             if self.ws and self.connected:
#                 self.system_info = self._get_system_info()
                
#                 try:
#                     import torch
#                     if torch.cuda.is_available():
#                         self.system_info["gpu"] = torch.cuda.get_device_name(0)
#                 except:
#                     pass
                
#                 self.ws.send(json.dumps({
#                     "type": "poll",
#                     "from": self.name,
#                     "role": self.role or "PREP",
#                     "system": self.system_info
#                 }))
    
#     def handle_input(self):
#         print("\n===========================================")
#         print("COMMANDS:")
#         print("  list              - List all participants")
#         print("  ping <name>       - Ping a participant")
#         print("  send <name> <msg> - Send message to participant")
#         print("  connect <name>   - Establish direct P2P connection")
#         print("  p2p <name> <msg> - Send P2P message directly")
#         print("  task <type>       - Submit a task (prep/train/keybert)")
#         print("  myinfo            - Show my system info")
#         print("  tasks             - Show available task types")
#         print("  quit              - Exit")
#         print("===========================================\n")
        
#         while self.running:
#             try:
#                 cmd = input("> ").strip().split()
#                 if not cmd:
#                     continue
                
#                 if cmd[0] == "quit":
#                     self.running = False
#                     print("Goodbye!")
#                     break
#                 elif cmd[0] == "list":
#                     if self.ws and self.connected:
#                         self.ws.send(json.dumps({"type": "get_participants"}))
#                 elif cmd[0] == "myinfo":
#                     print(f"\n{'='*50}")
#                     print(f"Name: {self.name}")
#                     print(f"Role: {self.role}")
#                     print(f"IP: {self._get_local_ip()}")
#                     print(f"System: {json.dumps(self.system_info, indent=2)}")
#                     print(f"{'='*50}\n")
#                 elif cmd[0] == "tasks":
#                     print("\nAvailable task types:")
#                     print("  prep       - Data preprocessing")
#                     print("  train      - Model training")
#                     print("  keybert    - Keyword extraction")
#                 elif cmd[0] == "task" and len(cmd) > 1:
#                     task_type = cmd[1]
#                     if self.ws and self.connected:
#                         self.ws.send(json.dumps({
#                             "type": "submit_task",
#                             "task_type": task_type,
#                             "from": self.name
#                         }))
#                 elif cmd[0] == "ping" and len(cmd) > 1:
#                     target = cmd[1]
#                     if self.ws and self.connected:
#                         self.ws.send(json.dumps({
#                             "type": "ping",
#                             "to": target,
#                             "from": self.name
#                         }))
#                 elif cmd[0] == "send" and len(cmd) > 2:
#                     target = cmd[1]
#                     content = " ".join(cmd[2:])
#                     if self.ws and self.connected:
#                         self.ws.send(json.dumps({
#                             "type": "message",
#                             "to": target,
#                             "content": content,
#                             "from": self.name
#                         }))
#                 else:
#                     print("Unknown command")
#             except EOFError:
#                 break
#             except Exception as e:
#                 print(f"Error: {e}")
    
#     def run(self):
#         # Register with coordinator
#         if not self.register():
#             logger.error("Failed to register with coordinator")
#             return
        
#         # Start WebSocket connection in background
#         ws_thread = threading.Thread(target=self.connect_websocket, daemon=True)
#         ws_thread.start()
        
#         # Start polling in background
#         poll_thread = threading.Thread(target=self.poll_tasks, daemon=True)
#         poll_thread.start()
        
#         # Handle user input
#         self.handle_input()


# def main():
#     import argparse
    
#     parser = argparse.ArgumentParser(description="P2P Network Participant")
#     parser.add_argument("-c", "--coordinator", required=True, help="Coordinator URL")
#     parser.add_argument("-n", "--name", help="Participant name")
#     parser.add_argument("-r", "--role", help="Role (PREP/PROC)")
    
#     args = parser.parse_args()
    
#     participant = Participant(args.coordinator, args.role)
    
#     if args.name:
#         participant.name = args.name
    
#     participant.run()


# if __name__ == "__main__":
#     main()
