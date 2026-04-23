package coordinator

import (
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"go.mongodb.org/mongo-driver/bson"
)

type Server struct {
	db          *Database
	upgrader    websocket.Upgrader
	clients     map[string]*websocket.Conn
	clientsMu   sync.RWMutex
	addr        string
	coordinator *CoordinatorInfo
	ccThreshold float64

	// Relay server for P2P connection establishment
	relayServer   *RelayServer
	relayListener net.Listener
	relayPort     int
}

type CoordinatorInfo struct {
	Name    string                 `json:"name"`
	Address string                 `json:"address"`
	Port    int                    `json:"port"`
	Status  string                 `json:"status"`
	Role    string                 `json:"role"`
	System  map[string]interface{} `json:"system"`
}

type RelayServer struct {
	sessions map[string]*RelaySession
	mu       sync.RWMutex
}

type RelaySession struct {
	id     string
	connA  net.Conn
	connB  net.Conn
	active bool
}

func NewRelayServer() *RelayServer {
	return &RelayServer{
		sessions: make(map[string]*RelaySession),
	}
}

func (rs *RelayServer) Start(port int) error {
	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return err
	}

	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				log.Printf("Relay server accept error: %v", err)
				continue
			}
			go rs.handleConnection(conn)
		}
	}()

	log.Printf("Relay server started on port %d", port)
	return nil
}

func (rs *RelayServer) handleConnection(conn net.Conn) {
	defer conn.Close()

	// Read session ID from client
	buf := make([]byte, 1024)
	n, err := conn.Read(buf)
	if err != nil {
		log.Printf("Failed to read session ID: %v", err)
		return
	}

	sessionID := string(buf[:n])
	sessionID = strings.TrimSpace(sessionID)

	rs.mu.Lock()
	session, exists := rs.sessions[sessionID]
	if !exists {
		// Create new session
		session = &RelaySession{
			id:     sessionID,
			active: false,
		}
		rs.sessions[sessionID] = session
	}

	if session.connA == nil {
		session.connA = conn
		log.Printf("Node A connected to session %s", sessionID)
		rs.mu.Unlock()

		// Wait for node B
		for !session.active {
			time.Sleep(100 * time.Millisecond)
		}
		rs.relayData(session.connA, session.connB)
	} else if session.connB == nil {
		session.connB = conn
		session.active = true
		log.Printf("Node B connected to session %s, starting relay", sessionID)
		rs.mu.Unlock()
		rs.relayData(session.connB, session.connA)
	} else {
		rs.mu.Unlock()
		log.Printf("Session %s already has two connections", sessionID)
	}
}

func (rs *RelayServer) relayData(src, dst net.Conn) {
	defer src.Close()
	defer dst.Close()

	buf := make([]byte, 4096)
	for {
		n, err := src.Read(buf)
		if err != nil {
			break
		}
		_, err = dst.Write(buf[:n])
		if err != nil {
			break
		}
	}
}

func NewServer(db *Database, addr string) *Server {
	// Get local IP and hostname
	hostname, _ := os.Hostname()
	localIP := getLocalIP()

	// Get CC threshold from environment (default 1.0)
	ccThreshold := 0.71
	if val := os.Getenv("CC_THRESHOLD"); val != "" {
		if parsed, err := strconv.ParseFloat(val, 64); err == nil {
			ccThreshold = parsed
		}
	}
	log.Printf("CC Threshold: %.2f", ccThreshold)

	// Find available port for coordinator
	coordinatorPort := findAvailablePort(11130)

	// Initialize relay server
	relayPort := findAvailablePort(11150)
	relayServer := NewRelayServer()
	if err := relayServer.Start(relayPort); err != nil {
		log.Printf("Failed to start relay server: %v", err)
	}

	coordinator := &CoordinatorInfo{
		Name:    fmt.Sprintf("%s-%s-%s-%s", hostname, runtime.GOOS, runtime.GOARCH, "COORD"),
		Address: localIP,
		Port:    coordinatorPort,
		Status:  "online",
		Role:    "COORD",
		System:  getSystemInfo(),
	}

	return &Server{
		db: db,
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool { return true },
		},
		clients:     make(map[string]*websocket.Conn),
		addr:        addr,
		coordinator: coordinator,
		ccThreshold: ccThreshold,
		relayServer: relayServer,
		relayPort:   relayPort,
	}
}

// configurePingHandler sets up ping/pong handlers for the WebSocket connection
// This ensures connections stay alive during heavy processing
func configurePingHandler(conn *websocket.Conn) {
	// Set pong handler - respond to pings automatically
	conn.SetPongHandler(func(string) error {
		return nil
	})

	// Set ping handler - respond to ping with pong
	conn.SetPingHandler(func(appData string) error {
		log.Printf("Received ping, sending pong")
		err := conn.WriteControl(websocket.PongMessage, []byte{}, time.Now().Add(time.Second))
		if err != nil {
			log.Printf("Error sending pong: %v", err)
		}
		return err
	})
}

func getLocalIP() string {
	conn, err := net.Dial("udp", "8.8.8.8:80")
	if err != nil {
		return "127.0.0.1"
	}
	defer conn.Close()

	localAddr := conn.LocalAddr().(*net.UDPAddr)
	return localAddr.IP.String()
}

func findAvailablePort(startPort int) int {
	for port := startPort; port < startPort+100; port++ {
		ln, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
		if err == nil {
			ln.Close()
			return port
		}
	}
	return startPort
}

func getSystemInfo() map[string]interface{} {
	return map[string]interface{}{
		"cpu":    0.0,
		"memory": 0.0,
		"disk":   0.0,
		"gpu":    "coordinator",
	}
}

func (s *Server) Start() error {
	r := gin.Default()

	// Set up template
	r.SetHTMLTemplate(htmlTemplate)

	// HTTP routes
	r.GET("/", s.handleIndex)
	r.GET("/participants", s.handleParticipants)
	r.GET("/messages", s.handleMessages)
	r.GET("/ws", s.handleWebSocket)
	r.GET("/coordinator", s.handleCoordinatorInfo)

	// API routes
	r.POST("/api/register", s.handleRegister)
	r.GET("/api/participants", s.handleAPIParticipants)
	r.POST("/api/participant/update", s.handleUpdateParticipant)
	r.GET("/api/participant/:name", s.handleAPIGetParticipant)
	r.DELETE("/api/participant/:name", s.handleAPIDeleteParticipant)
	r.POST("/api/message", s.handleAPISendMessage)
	r.GET("/api/ping/:name", s.handleAPIPing)
	r.POST("/api/command", s.handleCommand)

	// Task API routes
	r.POST("/api/task", s.handleCreateTask)
	r.GET("/api/tasks", s.handleGetTasks)
	r.GET("/api/task/:id", s.handleGetTask)
	r.PUT("/api/task/:id/status", s.handleUpdateTaskStatus)
	r.DELETE("/api/task/:id", s.handleDeleteTask)

	// KeyBERT task endpoint
	r.POST("/api/task/keybert", s.handleKeyBERTTask)
	r.POST("/api/task/result", s.handleTaskResult)

	// Training Job API routes (Federated Learning)
	r.POST("/api/training", s.handleCreateTrainingJob)
	r.GET("/api/training", s.handleGetTrainingJobs)
	r.GET("/api/training/:id", s.handleGetTrainingJob)
	r.PUT("/api/training/:id/start", s.handleStartTrainingJob)
	r.PUT("/api/training/:id/stop", s.handleStopTrainingJob)
	r.GET("/api/training/:id/batches", s.handleGetBatches)
	r.GET("/api/training/:id/progress", s.handleGetTrainingProgress)

	// Batch progress from PREP nodes
	r.POST("/api/batch/progress", s.handleBatchProgress)

	// Model updates from PROC nodes
	r.POST("/api/model/update", s.handleModelUpdate)

	// Poll endpoint for tasks (HTTP alternative to WebSocket polling)
	r.GET("/api/poll/tasks", s.handlePollTasks)

	// Batch sources endpoint for PROC nodes
	r.GET("/api/batch_sources/:job_id", s.handleGetBatchSources)

	// Training round status
	r.GET("/api/training/:id/round/:round", s.handleGetTrainingRound)

	// STUN endpoint - helps nodes discover their public IP:port for NAT traversal
	r.GET("/api/stun", s.handleSTUN)

	// Relay server info for P2P connections
	r.GET("/api/relay", s.handleGetRelayInfo)

	log.Printf("===========================================")
	log.Printf("Coordinator Name: %s", s.coordinator.Name)
	log.Printf("Coordinator Address: %s", s.coordinator.Address)
	log.Printf("Coordinator Port: %d", s.coordinator.Port)
	log.Printf("Coordinator Role: %s", s.coordinator.Role)
	log.Printf("===========================================")

	log.Printf("Coordinator server starting on %s", s.addr)
	return r.Run(s.addr)
}

func (s *Server) handleIndex(c *gin.Context) {
	participants, _ := s.db.GetParticipants()
	messages, _ := s.db.GetMessages()

	// Include coordinator in the list
	allMembers := append([]Participant{{
		Name:    s.coordinator.Name,
		Address: s.coordinator.Address,
		Port:    s.coordinator.Port,
		Status:  s.coordinator.Status,
		Role:    s.coordinator.Role,
		System:  s.coordinator.System,
	}}, participants...)

	c.HTML(http.StatusOK, "index", gin.H{
		"Members":     allMembers,
		"Messages":    messages,
		"Coordinator": s.coordinator,
	})
}

func (s *Server) handleParticipants(c *gin.Context) {
	participants, err := s.db.GetParticipants()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.HTML(http.StatusOK, "participants", gin.H{"Participants": participants})
}

func (s *Server) handleMessages(c *gin.Context) {
	messages, err := s.db.GetMessages()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.HTML(http.StatusOK, "messages", gin.H{"Messages": messages})
}

func (s *Server) handleCoordinatorInfo(c *gin.Context) {
	c.JSON(http.StatusOK, s.coordinator)
}

func (s *Server) handleWebSocket(c *gin.Context) {
	name := c.Query("name")
	if name == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "name required"})
		return
	}

	conn, err := s.upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return
	}
	defer conn.Close()

	// Configure ping/pong handlers to keep connection alive during heavy processing
	configurePingHandler(conn)

	s.clientsMu.Lock()
	s.clients[name] = conn
	s.clientsMu.Unlock()

	log.Printf("Client %s connected via WebSocket", name)

	// Get participant info from database to send role
	p, err := s.db.GetParticipantByName(name)
	var role string
	if err == nil {
		role = p.Role
		log.Printf("Sending role %s to client %s", role, name)
		// Send role assignment immediately after connection
		conn.WriteJSON(gin.H{
			"type": "role_assigned",
			"role": role,
		})
	}
	participants, _ := s.db.GetParticipants()

	// Build combined list with coordinator
	allParticipants := make([]map[string]interface{}, 0)
	allParticipants = append(allParticipants, map[string]interface{}{
		"name":    s.coordinator.Name,
		"address": s.coordinator.Address,
		"port":    s.coordinator.Port,
		"status":  s.coordinator.Status,
		"role":    s.coordinator.Role,
	})
	for _, p := range participants {
		allParticipants = append(allParticipants, map[string]interface{}{
			"name":    p.Name,
			"address": p.Address,
			"port":    p.Port,
			"status":  p.Status,
			"role":    p.Role,
		})
	}

	conn.WriteJSON(gin.H{"type": "participants_list", "participants": allParticipants})

	for {
		_, msg, err := conn.ReadMessage()
		if err != nil {
			break
		}

		var msgData map[string]interface{}
		if err := json.Unmarshal(msg, &msgData); err != nil {
			log.Printf("Failed to parse message from %s: %v", name, err)
			continue
		}

		msgType, _ := msgData["type"].(string)

		if msgType == "get_participants" {
			// Send updated participant list
			participants, _ := s.db.GetParticipants()
			allParticipants := make([]map[string]interface{}, 0)
			allParticipants = append(allParticipants, map[string]interface{}{
				"name":    s.coordinator.Name,
				"address": s.coordinator.Address,
				"port":    s.coordinator.Port,
				"status":  s.coordinator.Status,
				"role":    s.coordinator.Role,
				"online":  true,
			})
			for _, p := range participants {
				// Check if participant is connected via WebSocket
				s.clientsMu.RLock()
				_, online := s.clients[p.Name]
				s.clientsMu.RUnlock()

				allParticipants = append(allParticipants, map[string]interface{}{
					"name":    p.Name,
					"address": p.Address,
					"port":    p.Port,
					"status":  p.Status,
					"role":    p.Role,
					"online":  online,
				})
			}
			// Send as JSON object instead of string
			conn.WriteJSON(gin.H{"type": "participants_list", "participants": allParticipants})
		} else if msgType == "ping" {
			target, _ := msgData["target"].(string)
			s.sendPingRequest(name, target, conn)
		} else if msgType == "message" {
			target, _ := msgData["target"].(string)
			content, _ := msgData["content"].(string)
			s.forwardMessage(name, target, content)
		} else if msgType == "pong" {
			from, _ := msgData["from"].(string)
			s.forwardToClient(name, gin.H{"type": "pong", "from": from, "status": "online"})
		} else if msgType == "task_completed" {
			task_id, _ := msgData["task_id"].(string)
			from, _ := msgData["from"].(string)
			log.Printf("Task %s completed by %s", task_id, from)
			err := s.db.UpdatePendingTask(task_id, bson.M{"status": "completed"})
			if err != nil {
				log.Printf("Failed to update task %s: %v", task_id, err)
			}
		} else if msgType == "delivered" {
			from, _ := msgData["from"].(string)
			content, _ := msgData["content"].(string)
			s.forwardToClient(from, gin.H{"type": "delivered", "content": content})
		} else if msgType == "p2p_message" {
			target, _ := msgData["target"].(string)
			content, _ := msgData["content"].(string)
			s.clientsMu.RLock()
			targetConn, exists := s.clients[target]
			s.clientsMu.RUnlock()

			if exists {
				targetConn.WriteJSON(gin.H{"type": "p2p_message", "from": name, "content": content})
				s.forwardToClient(name, gin.H{"type": "delivered", "content": content})
			} else {
				s.forwardToClient(name, gin.H{"type": "error", "message": "peer offline"})
			}
		} else if msgType == "get_peer_info" {
			target, _ := msgData["target"].(string)
			addr, port, err := s.db.GetParticipantAddress(target)
			if err != nil {
				conn.WriteJSON(gin.H{"type": "peer_info", "peer_name": target, "available": false})
			} else {
				conn.WriteJSON(gin.H{
					"type":      "peer_info",
					"peer_name": target,
					"ip":        addr,
					"port":      port,
					"available": true,
				})
			}
		} else if msgType == "sysinfo" {
			// Update system info for participant (quiet)
		} else if msgType == "poll" {
			from, _ := msgData["from"].(string)
			role, _ := msgData["role"].(string)
			log.Printf("Poll from %s (role: %s)", from, role)
			if role == "PREP" || role == "PROC" {
				tasks, err := s.db.GetPendingTasksByRole(role)
				if err != nil {
					log.Printf("Poll error: %v", err)
					conn.WriteJSON(gin.H{"type": "poll_ack", "from": s.coordinator.Name})
					continue
				}
				if len(tasks) == 0 {
					conn.WriteJSON(gin.H{"type": "poll_ack", "from": s.coordinator.Name})
					continue
				}
				task := tasks[0]
				err = s.db.UpdatePendingTask(task.ID.Hex(), bson.M{
					"status":      "assigned",
					"assigned_to": from,
				})
				if err != nil {
					log.Printf("Failed to assign task %s: %v", task.ID.Hex(), err)
					conn.WriteJSON(gin.H{"type": "poll_ack", "from": s.coordinator.Name})
					continue
				}
				log.Printf("Assigned task %s (%s) to %s", task.ID.Hex(), task.TaskType, from)
				taskResponse := gin.H{
					"type":      "task_assigned",
					"id":        task.ID.Hex(),
					"job_id":    task.JobID,
					"job_name":  task.JobName,
					"task_type": task.TaskType,
					"role":      task.Role,
					"status":    "assigned",
					"data":      task.Data,
				}
				conn.WriteJSON(taskResponse)
			} else {
				conn.WriteJSON(gin.H{"type": "poll_ack", "from": s.coordinator.Name})
			}
		} else if msgType == "batch_progress" {
			// Handle batch progress from PREP nodes
			jobID, _ := msgData["job_id"].(string)

			// Handle batch_number - could be string or number
			batchNum := 0
			if batchNumStr, ok := msgData["batch_number"].(string); ok {
				if n, err := strconv.Atoi(batchNumStr); err == nil {
					batchNum = n
				}
			} else if batchNumFloat, ok := msgData["batch_number"].(float64); ok {
				batchNum = int(batchNumFloat)
			}

			status, _ := msgData["status"].(string)
			from, _ := msgData["from"].(string)

			// Extract record_count and progress if present
			recordCount := 0
			if rc, ok := msgData["record_count"].(float64); ok {
				recordCount = int(rc)
			}
			progress := 0.0
			if prog, ok := msgData["progress"].(float64); ok {
				progress = prog
			}

			log.Printf("Batch progress from %s: job=%s, batch=%d, status=%s, records=%d, progress=%.1f",
				from, jobID, batchNum, status, recordCount, progress)

			// Update batch in database
			batch := &Batch{
				JobID:       jobID,
				BatchNumber: batchNum,
				Status:      status,
				StoredOn:    from,
				RecordCount: recordCount,
				Progress:    progress,
			}
			s.db.SaveBatch(batch)

			// Update training job progress
			job, err := s.db.GetTrainingJobByID(jobID)
			if err == nil {
				ready := 0
				batches, _ := s.db.GetBatchesByJob(jobID)
				for _, b := range batches {
					if b.Status == "ready" || b.Status == "training" {
						ready++
					}
				}
				// Calculate actual total batches based on dataset size and batch size (ceiling division)
				actualTotalBatches := job.TotalBatches
				if job.DatasetSize > 0 && job.BatchSize > 0 {
					actualTotalBatches = (job.DatasetSize + job.BatchSize - 1) / job.BatchSize
				}
				prepProgress := float64(ready) / float64(actualTotalBatches) * 100
				s.db.UpdateTrainingJob(jobID, bson.M{"progress": prepProgress})

				// Check if all batches ready - start training
				if ready >= actualTotalBatches && job.Status == "preprocessing" {
					log.Printf("All %d batches ready (%.1f%%), starting training round for job %s", ready, prepProgress, jobID)
					s.startTrainingRound(jobID)
				}
			}
		} else if msgType == "request_batch" {
			// Handle batch request from PROC node - forward to PREP node
			targetPeer, _ := msgData["target_peer"].(string)
			jobID, _ := msgData["job_id"].(string)
			batchNum := 0
			if batchNumStr, ok := msgData["batch_number"].(string); ok {
				if n, err := strconv.Atoi(batchNumStr); err == nil {
					batchNum = n
				}
			} else if batchNumFloat, ok := msgData["batch_number"].(float64); ok {
				batchNum = int(batchNumFloat)
			}

			from, _ := msgData["from"].(string)
			log.Printf("Batch request from %s for batch %d from %s", from, batchNum, targetPeer)

			// Forward request to PREP node
			s.forwardToClient(targetPeer, gin.H{
				"type":         "send_batch",
				"job_id":       jobID,
				"batch_number": batchNum,
				"request_from": from,
			})
		} else if msgType == "task_completed" {
			// Handle task completion from PREP/PROC nodes
			taskID, _ := msgData["task_id"].(string)
			from, _ := msgData["from"].(string)
			jobID, _ := msgData["job_id"].(string)

			log.Printf("Task %s completed by %s", taskID, from)

			// Update preprocessing task status
			err := s.db.UpdatePreprocessingTaskByTaskID(taskID, bson.M{"status": "completed"})
			if err != nil {
				log.Printf("Failed to update preprocessing task %s: %v", taskID, err)
			}

			// If this is a PREP node task completion, check if all batches are ready
			if jobID != "" {
				job, err := s.db.GetTrainingJobByID(jobID)
				if err == nil && job != nil && job.Status == "preprocessing" {
					ready := 0
					batches, _ := s.db.GetBatchesByJob(jobID)
					for _, b := range batches {
						if b.Status == "ready" || b.Status == "training" {
							ready++
						}
					}
					// Calculate actual total batches based on dataset size and batch size (ceiling division)
					actualTotalBatches := job.TotalBatches
					if job.DatasetSize > 0 && job.BatchSize > 0 {
						actualTotalBatches = (job.DatasetSize + job.BatchSize - 1) / job.BatchSize
					}
					if ready >= actualTotalBatches {
						prepProgress := float64(ready) / float64(actualTotalBatches) * 100
						log.Printf("All %d batches ready (%.1f%%), starting training round for job %s (triggered by task_completed)", ready, prepProgress, jobID)
						s.startTrainingRound(jobID)
					}
				}
			}

		} else if msgType == "task_error" {
			// Handle task error from PREP/PROC nodes
			taskID, _ := msgData["task_id"].(string)
			from, _ := msgData["from"].(string)
			errorMsg, _ := msgData["error"].(string)

			log.Printf("Task %s failed by %s: %s", taskID, from, errorMsg)

			// Update preprocessing task status to failed
			err := s.db.UpdatePreprocessingTaskByTaskID(taskID, bson.M{"status": "failed"})
			if err != nil {
				log.Printf("Failed to update preprocessing task %s: %v", taskID, err)
			}

		} else if msgType == "batch_data" {
			// Handle batch data from PREP node - forward to requesting PROC node
			requestFrom, _ := msgData["request_from"].(string)
			if requestFrom != "" {
				// Forward the batch data to the requesting PROC node
				s.forwardToClient(requestFrom, msgData)
				log.Printf("Forwarded batch data to %s", requestFrom)
			}
		}
	}

	s.clientsMu.Lock()
	delete(s.clients, name)
	s.clientsMu.Unlock()
	log.Printf("Client %s disconnected", name)

	// Check if this participant had an incomplete preprocessing task
	s.handlePrepNodeDisconnect(name)

	// Delete participant from database on disconnect
	if err := s.db.DeleteParticipant(name); err != nil {
		log.Printf("Failed to delete participant %s: %v", name, err)
	} else {
		log.Printf("Participant %s removed from database", name)
	}
}

func (s *Server) sendPingRequest(from, target string, conn *websocket.Conn) {
	// Check if target is coordinator
	if target == s.coordinator.Name {
		conn.WriteJSON(gin.H{"type": "pong", "from": s.coordinator.Name, "status": "online"})
		return
	}

	s.clientsMu.RLock()
	targetConn, exists := s.clients[target]
	s.clientsMu.RUnlock()

	if exists {
		targetConn.WriteJSON(gin.H{"type": "ping", "from": from})
	} else {
		// Try direct connection if not connected to coordinator
		addr, port, err := s.db.GetParticipantAddress(target)
		if err == nil {
			go s.directPing(from, addr, port)
			return
		}
		conn.WriteJSON(gin.H{"type": "ping_response", "from": target, "status": "offline"})
	}
}

func (s *Server) directPing(from, address string, port int) {
	conn, err := connectToPeer(address, port)
	if err != nil {
		s.forwardToClient(from, gin.H{"type": "ping_response", "status": "offline"})
		return
	}
	defer conn.Close()

	s.forwardToClient(from, gin.H{"type": "ping_response", "status": "online"})
}

func (s *Server) forwardMessage(from, target, content string) {
	// Check if target is coordinator
	if target == s.coordinator.Name {
		log.Printf("Message to coordinator from %s: %s", from, content)
		msg := &Message{From: from, To: target, Content: content}
		s.db.SaveMessage(msg)

		// Send to sender that it was delivered
		s.forwardToClient(from, gin.H{"type": "delivered", "content": content})
		return
	}

	// Save message to database
	msg := &Message{From: from, To: target, Content: content}
	s.db.SaveMessage(msg)

	s.clientsMu.RLock()
	targetConn, exists := s.clients[target]
	s.clientsMu.RUnlock()

	if exists {
		targetConn.WriteJSON(gin.H{"type": "message", "from": from, "content": content})
		s.forwardToClient(from, gin.H{"type": "delivered", "content": content})
	} else {
		s.forwardToClient(from, gin.H{"type": "error", "message": "participant offline"})
	}
}

func (s *Server) forwardToClient(name string, data gin.H) {
	s.clientsMu.RLock()
	conn, exists := s.clients[name]
	s.clientsMu.RUnlock()

	if exists {
		if err := conn.WriteJSON(data); err != nil {
			log.Printf("ERROR: Failed to send to %s: %v", name, err)
		}
	} else {
		log.Printf("WARNING: Cannot forward to %s - not connected via WebSocket", name)
	}
}

// API Handlers
func (s *Server) handleRegister(c *gin.Context) {
	var p Participant
	if err := c.ShouldBindJSON(&p); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if p.Role == "" {
		p.Role = "PREP"
	}

	// Calculate ComputeCredit if system info is provided
	if p.System != nil {
		// Log received system info for debugging
		log.Printf("=== Received System Info from %s ===", p.Name)
		if gpu, ok := p.System["gpu"].(string); ok {
			log.Printf("  gpu: %s", gpu)
		}
		if cpu, ok := p.System["cpu"].(float64); ok {
			log.Printf("  cpu: %.2f", cpu)
		}
		if cores, ok := p.System["cpu_cores"].(float64); ok {
			log.Printf("  cpu_cores: %.0f", cores)
		}
		if mem, ok := p.System["memory"].(float64); ok {
			log.Printf("  memory: %.2f%%", mem)
		}
		if cuda, ok := p.System["cuda"].(bool); ok {
			log.Printf("  cuda: %v", cuda)
		}
		log.Printf("===========================================")

		cc := GetComputeCreditForSystemInfo(p.System)
		log.Printf("ComputeCredit for %s: %.4f", p.Name, cc)

		// Add ComputeCredit to system info
		if p.System == nil {
			p.System = make(map[string]interface{})
		}
		p.System["compute_credit"] = cc

		// Determine role based on CC threshold
		if cc >= s.ccThreshold {
			p.Role = "PROC"
		} else {
			p.Role = "PREP"
		}

		// Log detailed CC calculation
		ccObj := &ComputeCredit{}
		ccObj.Calculate(p.System)
		log.Printf("=== CC Calculation for %s ===", p.Name)
		log.Printf("  S(gpu) = %.4f (TFLOPS=%.2f, VRAM=%.2fGB)", ccObj.S_GPU, ccObj.GPUInfo.TFLOPS, ccObj.GPUInfo.VRAMGB)
		log.Printf("  S(cpu) = %.4f (Cores=%d, Clock=%.0fMHz)", ccObj.S_CPU, ccObj.CPUInfo.Cores, ccObj.CPUInfo.ClockMHz)
		log.Printf("  S(vram) = %.4f (VRAM=%.2fGB)", ccObj.S_VRAM, ccObj.VRAMGB)
		log.Printf("  CUDA = %v", ccObj.CUDAEnabled)
		log.Printf("  CC = %.4f (threshold=%.2f) -> Role: %s", cc, s.ccThreshold, p.Role)
	}

	if err := s.db.AddParticipant(&p); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "registered", "participant": p})
}

func (s *Server) handleUpdateParticipant(c *gin.Context) {
	var data map[string]interface{}
	if err := c.ShouldBindJSON(&data); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	name := data["name"].(string)
	port := int(data["port"].(float64))

	// Update participant's batch server port
	s.db.UpdateParticipant(name, bson.M{"port": port})

	log.Printf("Updated participant %s: batch_port=%d", name, port)
	c.JSON(http.StatusOK, gin.H{"message": "updated"})
}

func (s *Server) handleAPIParticipants(c *gin.Context) {
	participants, err := s.db.GetParticipants()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Include coordinator
	all := append([]Participant{{
		Name:    s.coordinator.Name,
		Address: s.coordinator.Address,
		Port:    s.coordinator.Port,
		Status:  s.coordinator.Status,
		Role:    s.coordinator.Role,
	}}, participants...)

	c.JSON(http.StatusOK, all)
}

func (s *Server) handleAPIGetParticipant(c *gin.Context) {
	name := c.Param("name")

	// Check if it's coordinator
	if name == s.coordinator.Name {
		c.JSON(http.StatusOK, s.coordinator)
		return
	}

	p, err := s.db.GetParticipantByName(name)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "not found"})
		return
	}
	c.JSON(http.StatusOK, p)
}

func (s *Server) handleAPIDeleteParticipant(c *gin.Context) {
	name := c.Param("name")
	if err := s.db.DeleteParticipant(name); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	s.clientsMu.Lock()
	if conn, exists := s.clients[name]; exists {
		conn.Close()
		delete(s.clients, name)
	}
	s.clientsMu.Unlock()

	c.JSON(http.StatusOK, gin.H{"message": "deleted"})
}

func (s *Server) handleAPISendMessage(c *gin.Context) {
	var msg Message
	if err := c.ShouldBindJSON(&msg); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	s.forwardMessage(msg.From, msg.To, msg.Content)
	c.JSON(http.StatusOK, gin.H{"message": "sent"})
}

func (s *Server) handleAPIPing(c *gin.Context) {
	name := c.Param("name")

	// Check if it's coordinator
	if name == s.coordinator.Name {
		c.JSON(http.StatusOK, gin.H{"status": "online"})
		return
	}

	s.clientsMu.RLock()
	_, exists := s.clients[name]
	s.clientsMu.RUnlock()

	if exists {
		c.JSON(http.StatusOK, gin.H{"status": "online"})
	} else {
		addr, port, err := s.db.GetParticipantAddress(name)
		if err != nil {
			c.JSON(http.StatusOK, gin.H{"status": "offline"})
			return
		}

		conn, err := connectToPeer(addr, port)
		if err != nil {
			c.JSON(http.StatusOK, gin.H{"status": "offline"})
			return
		}
		conn.Close()
		c.JSON(http.StatusOK, gin.H{"status": "online"})
	}
}

func (s *Server) handleCommand(c *gin.Context) {
	var cmd struct {
		Type    string `json:"type"`
		Target  string `json:"target"`
		Content string `json:"content"`
	}

	if err := c.ShouldBindJSON(&cmd); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if cmd.Type == "send" {
		// Send from coordinator
		s.forwardMessage(s.coordinator.Name, cmd.Target, cmd.Content)
		c.JSON(http.StatusOK, gin.H{"message": "sent from coordinator"})
	} else if cmd.Type == "ping" {
		s.clientsMu.RLock()
		_, exists := s.clients[cmd.Target]
		s.clientsMu.RUnlock()

		if exists {
			c.JSON(http.StatusOK, gin.H{"status": "online"})
		} else {
			c.JSON(http.StatusOK, gin.H{"status": "offline"})
		}
	}
}

func connectToPeer(address string, port int) (*websocket.Conn, error) {
	url := fmt.Sprintf("ws://%s:%d/ws", address, port)
	conn, _, err := websocket.DefaultDialer.Dial(url, nil)
	return conn, err
}

func toJSON(v interface{}) string {
	b, _ := json.Marshal(v)
	return string(b)
}

// Task API Handlers

func (s *Server) handleCreateTask(c *gin.Context) {
	var task Task
	if err := c.ShouldBindJSON(&task); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	task.Status = "pending"
	task.CreatedAt = time.Now()
	task.UpdatedAt = time.Now()

	if err := s.db.SaveTask(&task); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	log.Printf("Task created: %s (type: %s)", task.Name, task.Type)

	// Assign task to available participant
	s.assignTaskToParticipant(&task)

	c.JSON(http.StatusOK, gin.H{"message": "task created", "task": task})
}

func (s *Server) handleGetTasks(c *gin.Context) {
	tasks, err := s.db.GetTasks()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, tasks)
}

func (s *Server) handleGetTask(c *gin.Context) {
	id := c.Param("id")
	task, err := s.db.GetTaskByID(id)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "task not found"})
		return
	}
	c.JSON(http.StatusOK, task)
}

func (s *Server) handleUpdateTaskStatus(c *gin.Context) {
	id := c.Param("id")
	var update struct {
		Status string  `json:"status"`
		Loss   float64 `json:"loss,omitempty"`
	}
	if err := c.ShouldBindJSON(&update); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Get current task to check loss threshold
	task, err := s.db.GetTaskByID(id)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "task not found"})
		return
	}

	output := map[string]interface{}{
		"status": update.Status,
	}

	if update.Loss > 0 {
		output["loss"] = update.Loss
	}

	if err := s.db.UpdateTaskStatus(id, update.Status, output); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Check if training should stop based on loss threshold
	if update.Status == "completed" && task.Threshold > 0 && update.Loss < task.Threshold {
		log.Printf("Task %s completed - loss %.4f below threshold %.4f", id, update.Loss, task.Threshold)
		// Could notify all participants to stop
	}

	c.JSON(http.StatusOK, gin.H{"message": "status updated"})
}

func (s *Server) handleDeleteTask(c *gin.Context) {
	id := c.Param("id")
	if err := s.db.DeleteTask(id); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"message": "task deleted"})
}

func (s *Server) handleKeyBERTTask(c *gin.Context) {
	var req struct {
		TextData    []string `json:"texts"`
		ModelName   string   `json:"model_name"`
		NumKeywords int      `json:"num_keywords"`
		MinDF       int      `json:"min_df"`
		Threshold   float64  `json:"threshold"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Set defaults
	if req.ModelName == "" {
		req.ModelName = "paraphrase-multilingual-MiniLM-L12-v2"
	}
	if req.NumKeywords == 0 {
		req.NumKeywords = 10
	}
	if req.MinDF == 0 {
		req.MinDF = 1
	}

	// Create a task in the database
	task := &Task{
		Name:      "keybert-extraction",
		Type:      "keybert",
		Status:    "pending",
		Threshold: req.Threshold,
		InputData: map[string]interface{}{
			"texts":        req.TextData,
			"model_name":   req.ModelName,
			"num_keywords": req.NumKeywords,
			"min_df":       req.MinDF,
		},
	}

	if err := s.db.SaveTask(task); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Find a PROC node to assign the task
	procNode := s.findPROCNode()
	if procNode == "" {
		c.JSON(http.StatusOK, gin.H{
			"message": "task created, waiting for PROC node",
			"task_id": task.ID.Hex(),
			"status":  "pending",
		})
		return
	}

	// Send task to PROC node via WebSocket
	taskMsg := gin.H{
		"type":         "task_keybert",
		"task_id":      task.ID.Hex(),
		"texts":        req.TextData,
		"model_name":   req.ModelName,
		"num_keywords": req.NumKeywords,
		"min_df":       req.MinDF,
		"threshold":    req.Threshold,
	}

	s.forwardToClient(procNode, taskMsg)

	// Update task status
	now := time.Now()
	s.db.UpdateTaskStatus(task.ID.Hex(), "running", map[string]interface{}{
		"assigned_to": procNode,
		"started_at":  now,
	})

	log.Printf("KeyBERT task %s assigned to %s", task.ID.Hex(), procNode)

	c.JSON(http.StatusOK, gin.H{
		"message":     "task sent to PROC node",
		"task_id":     task.ID.Hex(),
		"assigned_to": procNode,
		"status":      "running",
	})
}

func (s *Server) findPROCNode() string {
	s.clientsMu.RLock()
	defer s.clientsMu.RUnlock()

	for name, conn := range s.clients {
		// Check if this is a PROC node
		p, err := s.db.GetParticipantByName(name)
		if err == nil && p.Role == "PROC" {
			log.Printf("Found PROC node: %s", name)
			return name
		}
		_ = conn // suppress unused warning
	}

	// If no PROC node connected via WS, try database
	participants, _ := s.db.GetParticipants()
	for _, p := range participants {
		if p.Role == "PROC" && p.Status == "online" {
			return p.Name
		}
	}

	return ""
}

func (s *Server) assignTaskToParticipant(task *Task) {
	// Find appropriate node based on task type
	var targetRole string
	switch task.Type {
	case "keybert", "train", "model_train":
		targetRole = "PROC"
	case "preprocess", "prep", "data_preprocess":
		targetRole = "PREP"
	default:
		targetRole = "PROC"
	}

	targetNode := s.findNodeByRole(targetRole)
	if targetNode == "" {
		log.Printf("No %s node available for task %s", targetRole, task.Name)
		return
	}

	// Send task via WebSocket
	taskMsg := gin.H{
		"type":       "task_custom",
		"task_name":  task.Type,
		"task_id":    task.ID.Hex(),
		"input_data": task.InputData,
	}

	if task.Threshold > 0 {
		taskMsg["threshold"] = task.Threshold
	}

	s.forwardToClient(targetNode, taskMsg)

	log.Printf("Task %s assigned to %s", task.Name, targetNode)
}

func (s *Server) findNodeByRole(role string) string {
	s.clientsMu.RLock()
	defer s.clientsMu.RUnlock()

	for name := range s.clients {
		p, err := s.db.GetParticipantByName(name)
		if err == nil && p.Role == role {
			return name
		}
	}

	// Fallback to database
	participants, _ := s.db.GetParticipants()
	for _, p := range participants {
		if p.Role == role && p.Status == "online" {
			return p.Name
		}
	}

	return ""
}

// Handle task results from participants
func (s *Server) handleTaskResult(c *gin.Context) {
	var result struct {
		TaskID     string                 `json:"task_id"`
		From       string                 `json:"from"`
		Status     string                 `json:"status"`
		ResultData map[string]interface{} `json:"result_data"`
		Loss       float64                `json:"loss,omitempty"`
	}

	if err := c.ShouldBindJSON(&result); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Update task in database
	output := result.ResultData
	if result.Loss > 0 {
		output["loss"] = result.Loss
	}

	if err := s.db.UpdateTaskStatus(result.TaskID, result.Status, output); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	log.Printf("Task %s completed by %s with status: %s", result.TaskID, result.From, result.Status)

	c.JSON(http.StatusOK, gin.H{"message": "result received"})
}

// Training Job API Handlers

func (s *Server) handleCreateTrainingJob(c *gin.Context) {
	var job TrainingJob
	if err := c.ShouldBindJSON(&job); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Set defaults
	if job.BatchSize == 0 {
		job.BatchSize = 32
	}
	if job.Epochs == 0 {
		job.Epochs = 10
	}
	if job.LearningRate == 0 {
		job.LearningRate = 0.001
	}
	if job.MaxRounds == 0 {
		job.MaxRounds = 5
	}
	if job.TotalBatches == 0 {
		job.TotalBatches = 100 // default
	}
	if job.DatasetSize == 0 {
		job.DatasetSize = 50000 // default dataset size
	}
	if job.ModelName == "" {
		job.ModelName = "bert-base-uncased" // default model for training
	}
	if job.ModelType == "" {
		job.ModelType = "bert" // default model type
	}

	job.Status = "pending"

	if err := s.db.SaveTrainingJob(&job); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	log.Printf("Training job created: %s (model: %s, dataset: %s)", job.Name, job.ModelType, job.DatasetURL)

	// Auto-start the training job
	log.Printf("Auto-starting training job %s...", job.ID.Hex())
	s.startTrainingJob(job.ID.Hex())

	c.JSON(http.StatusOK, gin.H{"message": "training job created", "job": job})
}

// startTrainingJob is the internal function that starts a training job
func (s *Server) startTrainingJob(id string) {
	job, err := s.db.GetTrainingJobByID(id)
	if err != nil {
		log.Printf("ERROR: Training job not found: %s", id)
		return
	}

	// Find PREP nodes to preprocess data
	prepNodes := s.findNodesByRole("PREP")
	log.Printf("startTrainingJob: found %d PREP nodes: %v", len(prepNodes), prepNodes)

	log.Printf("Job total batches: %d", job.TotalBatches)

	if len(prepNodes) == 0 {
		log.Printf("ERROR: No PREP nodes available for training job %s", id)
		return
	}

	// Start preprocessing phase
	s.db.UpdateTrainingJob(id, bson.M{"status": "preprocessing", "current_round": 0})

	// Create preprocessing subtasks divided by 1000 rows each
	// Use actual DatasetSize, not TotalBatches * BatchSize (which may be larger due to ceiling)
	dataset_size := job.DatasetSize
	if dataset_size == 0 {
		dataset_size = job.TotalBatches * job.BatchSize
	}
	subtask_size := 1000
	num_subtasks := (dataset_size + subtask_size - 1) / subtask_size
	log.Printf("Creating %d subtasks for dataset size %d (total_batches=%d, batch_size=%d)", num_subtasks, dataset_size, job.TotalBatches, job.BatchSize)
	for i := 0; i < num_subtasks; i++ {
		start_offset := i * subtask_size
		end_offset := start_offset + subtask_size
		if end_offset > dataset_size {
			end_offset = dataset_size
		}
		start_batch_num := start_offset / job.BatchSize

		// Save pending task to database for polling
		pendingTask := &PendingTask{
			JobID:    id,
			JobName:  job.Name,
			TaskType: "preprocess",
			Role:     "PREP",
			Status:   "pending",
			Data: map[string]interface{}{
				"job_id":          id,
				"dataset_url":     job.DatasetURL,
				"dataset_type":    job.DatasetType,
				"batch_size":      job.BatchSize,
				"start_offset":    start_offset,
				"end_offset":      end_offset,
				"start_batch_num": start_batch_num,
				"model_type":      job.ModelType,
				"model_name":      job.ModelName,
			},
		}
		err = s.db.SavePendingTask(pendingTask)
		if err != nil {
			log.Printf("Failed to save pending task: %v", err)
		} else {
			log.Printf("Saved pending subtask %d: records %d-%d", i, start_offset, end_offset)
		}
	}

	log.Printf("Training job %s started with subtasks created", id)
}

// handlePrepNodeDisconnect handles the case when a PREP node disconnects mid-task
func (s *Server) handlePrepNodeDisconnect(nodeName string) {
	// Find batches that were being processed by this node but not completed
	pendingBatches, err := s.db.GetBatchesByStatus("pending")
	if err != nil {
		log.Printf("Error getting pending batches: %v", err)
		return
	}

	var incompleteBatches []Batch
	for _, batch := range pendingBatches {
		if batch.StoredOn == nodeName {
			incompleteBatches = append(incompleteBatches, batch)
		}
	}

	if len(incompleteBatches) == 0 {
		return
	}

	log.Printf("PREP node %s disconnected with %d incomplete batches, reassigning...", nodeName, len(incompleteBatches))

	// Get job ID from the first incomplete batch
	if len(incompleteBatches) > 0 {
		jobID := incompleteBatches[0].JobID

		// Get available PREP nodes
		prepNodes, err := s.db.GetParticipantsByRole("PREP")
		if err != nil || len(prepNodes) == 0 {
			log.Printf("No available PREP nodes to reassign batches")
			// Mark batches as failed so job can be retried
			for _, batch := range incompleteBatches {
				s.db.SaveBatch(&Batch{
					JobID:       batch.JobID,
					BatchNumber: batch.BatchNumber,
					Status:      "failed",
					StoredOn:    nodeName,
				})
			}
			return
		}

		// Find a new PREP node (different from the disconnected one)
		var newNode string
		for _, node := range prepNodes {
			if node.Name != nodeName {
				newNode = node.Name
				break
			}
		}

		if newNode == "" {
			log.Printf("No alternative PREP node available")
			return
		}

		// Calculate new batch range for the new node
		job, err := s.db.GetTrainingJobByID(jobID)
		if err != nil {
			log.Printf("Error getting job: %v", err)
			return
		}

		// Find the highest batch number already assigned
		allBatches, _ := s.db.GetBatchesByJob(jobID)
		maxBatch := 0
		for _, b := range allBatches {
			if b.BatchNumber > maxBatch {
				maxBatch = b.BatchNumber
			}
		}

		// Assign remaining batches to new node
		startBatch := maxBatch + 1
		endBatch := startBatch + len(incompleteBatches)
		if endBatch > job.TotalBatches {
			endBatch = job.TotalBatches
		}

		// Get job details
		newJob, _ := s.db.GetTrainingJobByID(jobID)
		if newJob == nil {
			return
		}

		// Send task to new node with correct offset
		s.forwardToClient(newNode, gin.H{
			"type":           "task_preprocess",
			"job_id":         jobID,
			"job_name":       newJob.Name,
			"dataset_url":    newJob.DatasetURL,
			"dataset_type":   newJob.DatasetType,
			"batch_size":     newJob.BatchSize,
			"start_batch":    startBatch,
			"end_batch":      endBatch,
			"dataset_offset": startBatch * newJob.BatchSize,
			"model_type":     newJob.ModelType,
			"model_name":     newJob.ModelName,
		})

		log.Printf("Reassigned batches %d-%d to new PREP node %s", startBatch, endBatch, newNode)
	}
}

func (s *Server) handleGetTrainingJobs(c *gin.Context) {
	jobs, err := s.db.GetTrainingJobs()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, jobs)
}

func (s *Server) handleGetTrainingJob(c *gin.Context) {
	id := c.Param("id")
	job, err := s.db.GetTrainingJobByID(id)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "training job not found"})
		return
	}
	c.JSON(http.StatusOK, job)
}

func (s *Server) handleStartTrainingJob(c *gin.Context) {
	id := c.Param("id")

	job, err := s.db.GetTrainingJobByID(id)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "training job not found"})
		return
	}

	// Find PREP nodes to preprocess data
	prepNodes := s.findNodesByRole("PREP")
	log.Printf("handleStartTrainingJob: found %d PREP nodes: %v", len(prepNodes), prepNodes)

	if len(prepNodes) == 0 {
		log.Printf("ERROR: No PREP nodes available for training job %s", id)
		c.JSON(http.StatusBadRequest, gin.H{"error": "no PREP nodes available"})
		return
	}

	// Start preprocessing phase
	s.db.UpdateTrainingJob(id, bson.M{"status": "preprocessing", "current_round": 0})

	// Create preprocessing tasks split by batch_size
	batchSize := job.BatchSize
	totalBatches := job.TotalBatches

	for batchIndex := 0; batchIndex < totalBatches; batchIndex++ {
		startOffset := batchIndex * batchSize
		endOffset := startOffset + batchSize
		if endOffset > job.DatasetSize {
			endOffset = job.DatasetSize
		}

		// Calculate starting batch number for this task
		startBatchNum := batchIndex * ((endOffset - startOffset) / batchSize)

		// Create preprocessing task for this batch
		prepTask := &PreprocessingTask{
			JobID:         id,
			TaskID:        fmt.Sprintf("%s-prep-%d", id, batchIndex),
			StartOffset:   startOffset,
			EndOffset:     endOffset,
			StartBatchNum: startBatchNum,
			Status:        "pending",
			AssignedTo:    "",
			CreatedAt:     time.Now(),
		}
		err := s.db.SavePreprocessingTask(prepTask)
		if err != nil {
			log.Printf("Failed to save preprocessing task: %v", err)
		} else {
			log.Printf("Created preprocessing task for batch %d (records %d-%d)", batchIndex, startOffset, endOffset)
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"message":    "training started",
		"status":     "preprocessing",
		"prep_nodes": len(prepNodes),
	})
}

func (s *Server) handleStopTrainingJob(c *gin.Context) {
	id := c.Param("id")

	err := s.db.UpdateTrainingJob(id, bson.M{"status": "stopped"})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Notify all nodes to stop
	allNodes := s.getAllConnectedNodes()
	for _, node := range allNodes {
		s.forwardToClient(node, gin.H{
			"type":   "training_stop",
			"job_id": id,
		})
	}

	c.JSON(http.StatusOK, gin.H{"message": "training stopped"})
}

func (s *Server) handleGetBatches(c *gin.Context) {
	id := c.Param("id")
	batches, err := s.db.GetBatchesByJob(id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, batches)
}

func (s *Server) handleGetTrainingProgress(c *gin.Context) {
	id := c.Param("id")

	job, err := s.db.GetTrainingJobByID(id)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "training job not found"})
		return
	}

	total, ready, completed := s.db.GetBatchStats(id)

	progress := gin.H{
		"job_id":        id,
		"name":          job.Name,
		"status":        job.Status,
		"current_round": job.CurrentRound,
		"max_rounds":    job.MaxRounds,
		"current_loss":  job.CurrentLoss,
		"threshold":     job.Threshold,
		"progress":      job.Progress,
		"batches": gin.H{
			"total":     total,
			"ready":     ready,
			"completed": completed,
		},
	}

	c.JSON(http.StatusOK, progress)
}

func (s *Server) handleBatchProgress(c *gin.Context) {
	var batch Batch
	if err := c.ShouldBindJSON(&batch); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Check if batch exists
	batches, _ := s.db.GetBatchesByJob(batch.JobID)
	found := false
	for _, b := range batches {
		if b.BatchNumber == batch.BatchNumber && b.StoredOn == batch.StoredOn {
			// Update existing batch (including record_count and progress)
			s.db.UpdateBatch(b.ID.Hex(), bson.M{
				"status":       batch.Status,
				"progress":     batch.Progress,
				"record_count": batch.RecordCount,
			})
			found = true
			break
		}
	}

	if !found {
		// Create new batch record with record_count
		log.Printf("Creating batch: job=%s, batch=%d, records=%d, progress=%.1f",
			batch.JobID, batch.BatchNumber, batch.RecordCount, batch.Progress)
		s.db.SaveBatch(&batch)
	}

	// Check if all batches are ready
	total, ready, _ := s.db.GetBatchStats(batch.JobID)
	job, _ := s.db.GetTrainingJobByID(batch.JobID)
	log.Printf("handleBatchProgress: job=%s, total_batches=%d, ready_batches=%d, status=%s", 
		batch.JobID, job.TotalBatches, ready, job.Status)
	if job != nil && ready >= job.TotalBatches && job.Status == "preprocessing" {
		// All batches ready - start training
		log.Printf("=== ALL BATCHES READY: %d/%d, starting training ===", ready, job.TotalBatches)
		s.startTrainingRound(batch.JobID)
	} else if job != nil {
		if job.Status != "preprocessing" {
			log.Printf("Job %s status is '%s', not 'preprocessing' - skipping startTrainingRound", batch.JobID, job.Status)
		}
		if ready < job.TotalBatches {
			log.Printf("Not all batches ready yet: %d/%d", ready, job.TotalBatches)
		}
	}

	// Update job progress
	if job != nil {
		prepProgress := float64(ready) / float64(job.TotalBatches) * 100
		log.Printf("Updating job %s progress to %.1f%% (%d/%d batches)", batch.JobID, prepProgress, ready, job.TotalBatches)
		s.db.UpdateTrainingJob(batch.JobID, bson.M{"progress": prepProgress})
	}

	c.JSON(http.StatusOK, gin.H{"message": "batch progress updated"})
}

func (s *Server) handlePollTasks(c *gin.Context) {
	name := c.Query("name")
	role := c.Query("role")
	
	if name == "" || role == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "name and role required"})
		return
	}
	
	tasks, err := s.db.GetPendingTasksByRole(role)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	// Filter tasks for this specific node if assigned
	var nodeTasks []PendingTask
	for _, task := range tasks {
		if task.AssignedTo == "" || task.AssignedTo == name {
			nodeTasks = append(nodeTasks, task)
		}
	}
	
	if len(nodeTasks) > 0 {
		// Assign first task to this node
		task := nodeTasks[0]
		err = s.db.UpdatePendingTask(task.ID.Hex(), bson.M{
			"status":      "assigned",
			"assigned_to": name,
		})
		if err != nil {
			log.Printf("Failed to assign task %s: %v", task.ID.Hex(), err)
		} else {
			log.Printf("Assigned task %s (%s) to %s via HTTP poll", task.ID.Hex(), task.TaskType, name)
		}
		
		// Return the task
		c.JSON(http.StatusOK, gin.H{
			"tasks": []gin.H{
				{
					"_id":       task.ID.Hex(),
					"job_id":    task.JobID,
					"job_name":  task.JobName,
					"task_type": task.TaskType,
					"role":      task.Role,
					"data":      task.Data,
				},
			},
		})
	} else {
		c.JSON(http.StatusOK, gin.H{"tasks": []gin.H{}})
	}
}

func (s *Server) handleModelUpdate(c *gin.Context) {
	var update ModelUpdate
	if err := c.ShouldBindJSON(&update); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Save the model update
	update.Status = "received"
	s.db.SaveModelUpdate(&update)

	log.Printf("Received model update from %s for job %s, round %d, loss %.4f",
		update.From, update.JobID, update.Round, update.Loss)

	// Check if we have all expected updates for this round
	job, _ := s.db.GetTrainingJobByID(update.JobID)
	if job == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "job not found"})
		return
	}

	updates, _ := s.db.GetModelUpdatesByJobAndRound(update.JobID, update.Round)
	procNodes := s.findNodesByRole("PROC")

	if len(updates) >= len(procNodes) {
		// All nodes submitted updates - aggregate
		s.aggregateModelUpdates(update.JobID, update.Round)
	}

	c.JSON(http.StatusOK, gin.H{"message": "update received"})
}

func (s *Server) handleGetTrainingRound(c *gin.Context) {
	id := c.Param("id")
	roundNum, _ := strconv.Atoi(c.Param("round"))

	round, err := s.db.GetTrainingRound(id, roundNum)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "round not found"})
		return
	}

	c.JSON(http.StatusOK, round)
}

// Helper functions

func (s *Server) findNodesByRole(role string) []string {
	var nodes []string

	s.clientsMu.RLock()
	log.Printf("Looking for nodes with role %s, connected clients: %d", role, len(s.clients))
	for name := range s.clients {
		log.Printf("  Checking connected client: %s", name)
		p, err := s.db.GetParticipantByName(name)
		if err == nil {
			log.Printf("    DB lookup: name=%s, role=%s, status=%s", p.Name, p.Role, p.Status)
			if p.Role == role {
				nodes = append(nodes, name)
			}
		} else {
			log.Printf("    DB lookup failed for %s: %v", name, err)
		}
	}
	s.clientsMu.RUnlock()

	// Fallback to database
	if len(nodes) == 0 {
		log.Printf("No connected nodes with role %s, falling back to database", role)
		participants, _ := s.db.GetParticipants()
		for _, p := range participants {
			log.Printf("  Database node: name=%s, role=%s, status=%s", p.Name, p.Role, p.Status)
			if p.Role == role && p.Status == "online" {
				nodes = append(nodes, p.Name)
			}
		}
	}

	log.Printf("Found %d nodes with role %s", len(nodes), role)
	return nodes
}

func (s *Server) getAllConnectedNodes() []string {
	s.clientsMu.RLock()
	defer s.clientsMu.RUnlock()

	nodes := make([]string, 0, len(s.clients))
	for name := range s.clients {
		nodes = append(nodes, name)
	}
	return nodes
}

func (s *Server) startTrainingRound(jobID string) {
	job, _ := s.db.GetTrainingJobByID(jobID)
	if job == nil {
		log.Printf("ERROR: Job %s not found in startTrainingRound", jobID)
		return
	}

	log.Printf("=== startTrainingRound called for job %s (status=%s, round=%d) ===", jobID, job.Status, job.CurrentRound)

	// Update job status
	nextRound := job.CurrentRound + 1
	s.db.UpdateTrainingJob(jobID, bson.M{
		"status":        "training",
		"current_round": nextRound,
	})
	log.Printf("Updated job %s status to 'training', round %d", jobID, nextRound)

	// Find PROC nodes
	procNodes := s.findNodesByRole("PROC")
	log.Printf("Found %d PROC nodes: %v", len(procNodes), procNodes)
	if len(procNodes) == 0 {
		log.Printf("No PROC nodes available for training job %s", jobID)
		return
	}

	// Get ready batches
	batches, _ := s.db.GetReadyBatches(jobID)
	log.Printf("Found %d ready batches for job %s", len(batches), jobID)
	if len(batches) == 0 {
		log.Printf("No ready batches for training job %s", jobID)
		return
	}

	// Add batch source info - find which PREP node has each batch
	// Format: batch_number -> {name, ip, port}
	batchSourceInfo := make(map[string]map[string]string)
	allBatches, _ := s.db.GetBatchesByJob(jobID)
	for _, b := range allBatches {
		if b.Status == "ready" {
			// Get participant info for IP/port
			prepNode, _ := s.db.GetParticipantByName(b.StoredOn)
			batchSourceInfo[strconv.Itoa(b.BatchNumber)] = map[string]string{
				"name": b.StoredOn,
				"ip":   prepNode.Address,
				"port": strconv.Itoa(prepNode.Port), // batch server port
			}
		}
	}

	// Send batch source info to all PROC nodes FIRST
	log.Printf("Sending batch source info to %d PROC nodes", len(procNodes))
	for _, node := range procNodes {
		s.forwardToClient(node, gin.H{
			"type":          "batch_sources",
			"job_id":        jobID,
			"batch_sources": batchSourceInfo,
		})
		log.Printf("Sent batch_sources to %s with %d batch mappings", node, len(batchSourceInfo))
	}

	// Distribute batches to PROC nodes
	batchesPerNode := len(batches) / len(procNodes)
	for i, node := range procNodes {
		startIdx := i * batchesPerNode
		endIdx := startIdx + batchesPerNode
		if i == len(procNodes)-1 {
			endIdx = len(batches)
		}

		nodeBatches := batches[startIdx:endIdx]

		// Save pending task to database for polling
		pendingTask := &PendingTask{
			JobID:      jobID,
			JobName:    job.Name,
			TaskType:   "train",
			Role:       "PROC",
			Status:     "pending",
			AssignedTo: node, // Assign to specific node
			Data: map[string]interface{}{
				"round":          nextRound,
				"batches":        nodeBatches,
				"model_type":     job.ModelType,
				"model_name":     job.ModelName,
				"epochs":         job.Epochs,
				"learning_rate":  job.LearningRate,
				"batch_size":     job.BatchSize,
				"threshold":      job.Threshold,
				"is_first_round": nextRound == 1,
			},
		}
		err := s.db.SavePendingTask(pendingTask)
		if err != nil {
			log.Printf("Failed to save pending task: %v", err)
		} else {
			log.Printf("Saved pending training task for %s (round %d)", node, nextRound)
		}

		log.Printf("Training task queued for %s (round %d, %d batches) - will be delivered via polling", node, nextRound, len(nodeBatches))
	}

	// Create training round record
	round := &TrainingRound{
		JobID:           jobID,
		RoundNumber:     nextRound,
		Status:          "in_progress",
		SelectedNodes:   procNodes,
		ExpectedUpdates: len(procNodes),
		ReceivedUpdates: 0,
	}
	s.db.SaveTrainingRound(round)
	log.Printf("Created training round %d for job %s with %d expected updates", nextRound, jobID, len(procNodes))
}

func (s *Server) aggregateModelUpdates(jobID string, round int) {
	job, _ := s.db.GetTrainingJobByID(jobID)
	if job == nil {
		return
	}

	updates, _ := s.db.GetModelUpdatesByJobAndRound(jobID, round)
	if len(updates) == 0 {
		return
	}

	// Calculate average loss and accuracy
	var totalLoss, totalAccuracy float64
	for _, u := range updates {
		totalLoss += u.Loss
		totalAccuracy += u.Accuracy
	}
	avgLoss := totalLoss / float64(len(updates))
	avgAccuracy := totalAccuracy / float64(len(updates))

	log.Printf("Aggregating %d updates for round %d - Avg Loss: %.4f, Avg Accuracy: %.4f",
		len(updates), round, avgLoss, avgAccuracy)

	// Update round status
	roundRec, _ := s.db.GetTrainingRound(jobID, round)
	if roundRec != nil {
		s.db.UpdateTrainingRound(roundRec.ID.Hex(), bson.M{
			"status":           "completed",
			"received_updates": len(updates),
			"avg_loss":         avgLoss,
			"avg_accuracy":     avgAccuracy,
		})
	}

	// Check if training should stop
	if job.Threshold > 0 && avgLoss < job.Threshold {
		log.Printf("Training complete! Loss %.4f below threshold %.4f", avgLoss, job.Threshold)
		s.db.UpdateTrainingJob(jobID, bson.M{
			"status":       "completed",
			"current_loss": avgLoss,
			"progress":     100,
		})

		// Notify all nodes training is complete
		allNodes := s.getAllConnectedNodes()
		for _, node := range allNodes {
			s.forwardToClient(node, gin.H{
				"type":       "training_complete",
				"job_id":     jobID,
				"final_loss": avgLoss,
			})
		}
		return
	}

	// Check if more rounds needed
	if round >= job.MaxRounds {
		log.Printf("Max rounds %d reached", job.MaxRounds)
		s.db.UpdateTrainingJob(jobID, bson.M{
			"status":       "completed",
			"current_loss": avgLoss,
			"progress":     100,
		})
		return
	}

	// Start next round
	s.startTrainingRound(jobID)
}

// handleSTUN provides STUN-like functionality to help nodes discover their public IP:port
// This is simplified - in a real STUN scenario, multiple requests from different ports would be used
func (s *Server) handleSTUN(c *gin.Context) {
	// Get the requester's IP as seen from the server (may be NAT'd)
	clientIP := c.ClientIP()

	// Also try to get the actual remote address from the connection
	forwarded := c.GetHeader("X-Forwarded-For")
	if forwarded != "" {
		clientIP = forwarded
	}

	// Get the port from the request - nodes send their local listening port
	localPort := c.Query("port")

	log.Printf("STUN request from %s (reported port: %s)", clientIP, localPort)

	// Return the public IP and port information
	// In a real STUN scenario, the client would use this to determine NAT type
	// For hole-punching, we return the server's view of the client's IP
	c.JSON(http.StatusOK, gin.H{
		"status":        "ok",
		"external_ip":   clientIP,
		"external_port": localPort,
		"server_ip":     s.coordinator.Address,
		"server_port":   s.coordinator.Port,
	})
}

func (s *Server) handleGetRelayInfo(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"relay_ip":   s.coordinator.Address,
		"relay_port": s.relayPort,
	})
}
