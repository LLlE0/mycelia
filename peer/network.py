"""
P2P Network utilities for direct node-to-node communication
Handles batch transfer between PREP and PROC nodes without going through coordinator
"""

import json
import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import urlopen, Request
from urllib.error import URLError

logger = logging.getLogger(__name__)


class BatchServer:
    """HTTP server to serve batches to other nodes"""
    
    def __init__(self, participant, port=None):
        self.participant = participant
        self.port = port or 0
        self.server = None
        self.running = False
        
    def start(self):
        """Start the batch server"""
        if self.running:
            return self.port
            
        # Find available port
        import socket
        for p in range(11130, 11160):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('0.0.0.0', p))
                s.close()
                self.port = p
                break
            except:
                continue
        
        if self.port == 0:
            self.port = 11130
            
        handler = self._create_handler()
        self.server = HTTPServer(('0.0.0.0', self.port), handler)
        self.running = True
        
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        
        logger.info(f"Batch server started on port {self.port}")
        return self.port
    
    def _create_handler(self):
        """Create request handler with access to participant"""
        participant = self.participant
        
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress logging
                
            def do_GET(self):
                # Parse path: /batch/<job_id>/<batch_number>
                path = self.path.strip('/')
                parts = path.split('/')
                
                if len(parts) >= 3 and parts[0] == 'batch':
                    job_id = parts[1]
                    try:
                        batch_num = int(parts[2])
                    except:
                        self.send_error(400, "Invalid batch number")
                        return
                    
                    batch_key = f"{job_id}_batch_{batch_num}"
                    local_batches = getattr(participant, 'local_batches', {})
                    
                    if batch_key in local_batches:
                        batch_data = local_batches[batch_key]
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps(batch_data).encode())
                    else:
                        self.send_error(404, "Batch not found")
                else:
                    self.send_error(400, "Invalid request")
                    
            def do_POST(self):
                # Health check / register endpoint
                if self.path == '/health':
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        'status': 'ok',
                        'name': participant.name,
                        'role': participant.role,
                        'port': self.server.server_port
                    }).encode())
                else:
                    self.send_error(400, "Invalid request")
        
        return Handler
    
    def stop(self):
        """Stop the server"""
        if self.server:
            self.running = False
            self.server.shutdown()
            logger.info("Batch server stopped")


class BatchClient:
    """HTTP client to request batches from PREP nodes"""
    
    def __init__(self, timeout=10):
        self.timeout = timeout
    
    def request_batch(self, ip, port, job_id, batch_num):
        """Request a batch from a PREP node"""
        url = f"http://{ip}:{port}/batch/{job_id}/{batch_num}"
        
        try:
            req = Request(url)
            req.add_header('Accept', 'application/json')
            
            with urlopen(req, timeout=self.timeout) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    logger.info(f"Received batch {batch_num} from {ip}:{port}")
                    return data
                else:
                    logger.warning(f"Batch request failed: HTTP {response.status}")
                    return None
                    
        except URLError as e:
            logger.error(f"Failed to request batch {batch_num} from {ip}:{port}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error requesting batch: {e}")
            return None
    
    def check_node(self, ip, port):
        """Check if a node is reachable"""
        url = f"http://{ip}:{port}/health"
        
        try:
            req = Request(url)
            with urlopen(req, timeout=5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    return data
                return None
        except:
            return None


# Global instances
_batch_server = None
_batch_client = None


def get_batch_server(participant=None, port=None):
    """Get or create batch server"""
    global _batch_server
    if _batch_server is None:
        _batch_server = BatchServer(participant, port)
    return _batch_server


def get_batch_client(timeout=10):
    """Get or create batch client"""
    global _batch_client
    if _batch_client is None:
        _batch_client = BatchClient(timeout)
    return _batch_client


def request_batch_direct(ip, port, job_id, batch_num, timeout=10):
    """Convenience function to request a batch directly"""
    client = get_batch_client(timeout)
    return client.request_batch(ip, port, job_id, batch_num)


class RelayClient:
    """Client for connecting to coordinator's relay server for P2P connections"""

    def __init__(self, relay_ip, relay_port):
        self.relay_ip = relay_ip
        self.relay_port = relay_port

    def connect_for_session(self, session_id):
        """Connect to relay server for a specific session"""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.relay_ip, self.relay_port))
            # Send session ID
            sock.send(session_id.encode())
            return sock
        except Exception as e:
            logger.error(f"Failed to connect to relay: {e}")
            return None


def get_relay_client(relay_ip, relay_port):
    """Get relay client instance"""
    return RelayClient(relay_ip, relay_port)