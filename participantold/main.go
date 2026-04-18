package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

const Role = "PREP"

type Config struct {
	Name        string
	Coordinator string
	Port        int
}

type Message struct {
	Type    string `json:"type"`
	From    string `json:"from,omitempty"`
	To      string `json:"to,omitempty"`
	Content string `json:"content,omitempty"`
	Status  string `json:"status,omitempty"`
	Target  string `json:"target,omitempty"`
}

type SystemInfo struct {
	CPU    float64 `json:"cpu"`
	Memory float64 `json:"memory"`
	Disk   float64 `json:"disk"`
	GPU    string  `json:"gpu,omitempty"`
}

var (
	cfg       Config
	wsConn    *websocket.Conn
	inbox     = make(chan Message)
	connected = false
)

func main() {
	// Get local IP
	localIP := getLocalIP()

	// Get hostname and OS
	hostname, _ := os.Hostname()
	osName := runtime.GOOS
	arch := runtime.GOARCH

	// Auto-select port starting from 11130
	port := findAvailablePort(11130)

	// Load configuration
	cfg = Config{
		Name:        getEnv("NAME", ""),
		Coordinator: getEnv("COORDINATOR", "http://localhost:8080"),
		Port:        port,
	}

	// Set name if not provided: hostname-os-arch-role
	if cfg.Name == "" {
		cfg.Name = fmt.Sprintf("%s-%s-%s-%s", hostname, osName, arch, Role)
	}

	log.Printf("===========================================")
	log.Printf("Node Name: %s", cfg.Name)
	log.Printf("Coordinator: %s", cfg.Coordinator)
	log.Printf("Local IP: %s", localIP)
	log.Printf("Port: %d", cfg.Port)
	log.Printf("Role: %s", Role)
	log.Printf("===========================================")

	// Register with coordinator
	if err := register(); err != nil {
		log.Printf("Warning: Failed to register with coordinator: %v", err)
	}

	// Connect to coordinator WebSocket
	go connectWebSocket()

	// Start HTTP server for peer connections
	go startHTTPServer()

	// Start system info polling
	go pollSystemInfo()

	// Handle console input
	go handleInput()

	// Handle graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down...")
	if wsConn != nil {
		wsConn.Close()
	}
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

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
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

func getSystemInfo() SystemInfo {
	info := SystemInfo{}
	info.CPU = getCPUUsage()
	info.Memory = getMemoryUsage()
	info.Disk = getDiskUsage()
	info.GPU = getGPUInfo()
	return info
}

func getCPUUsage() float64 {
	var output []byte
	var err error

	switch runtime.GOOS {
	case "linux":
		cmd := exec.Command("bash", "-c", "top -bn1 | grep 'Cpu(s)' | sed 's/.*, *\\([0-9.]*\\)%.*.*/\\1/' | awk '{print 100 - $1}'")
		output, err = cmd.Output()
	case "windows":
		cmd := exec.Command("powershell", "-Command", "(Get-Counter '\\Processor(_Total)\\% Processor Time').CounterSamples.CookedValue")
		output, err = cmd.Output()
	case "darwin":
		cmd := exec.Command("sh", "-c", "top -l 1 -n 0 | grep 'CPU usage' | awk '{print $3}' | tr -d '%'")
		output, err = cmd.Output()
	default:
		return 0
	}

	if err != nil {
		return 0
	}

	var cpu float64
	fmt.Sscanf(strings.TrimSpace(string(output)), "%f", &cpu)
	return cpu
}

func getMemoryUsage() float64 {
	var output []byte
	var err error

	switch runtime.GOOS {
	case "linux":
		cmd := exec.Command("bash", "-c", "free | grep Mem | awk '{print ($3/$2) * 100}'")
		output, err = cmd.Output()
	case "windows":
		cmd := exec.Command("powershell", "-Command", "(Get-Counter '\\Memory\\% Committed Bytes In Use').CounterSamples.CookedValue")
		output, err = cmd.Output()
	case "darwin":
		cmd := exec.Command("bash", "-c", "vm_stat | head -4 | grep 'Pages active' | awk '{print $3}'")
		output, err = cmd.Output()
	default:
		return 0
	}

	if err != nil {
		return 0
	}

	var mem float64
	fmt.Sscanf(strings.TrimSpace(string(output)), "%f", &mem)
	return mem
}

func getDiskUsage() float64 {
	var output []byte
	var err error

	switch runtime.GOOS {
	case "linux":
		cmd := exec.Command("bash", "-c", "df -h / | tail -1 | awk '{print $5}' | tr -d '%'")
		output, err = cmd.Output()
	case "windows":
		cmd := exec.Command("powershell", "-Command", "Get-PSDrive C | Select-Object -ExpandProperty Used")
		output, err = cmd.Output()
	case "darwin":
		cmd := exec.Command("bash", "-c", "df -h / | tail -1 | awk '{print $5}' | tr -d '%'")
		output, err = cmd.Output()
	default:
		return 0
	}

	if err != nil {
		return 0
	}

	var disk float64
	fmt.Sscanf(strings.TrimSpace(string(output)), "%f", &disk)
	return disk
}

func getGPUInfo() string {
	var output []byte
	var err error

	switch runtime.GOOS {
	case "linux":
		cmd := exec.Command("bash", "-c", "lspci 2>/dev/null | grep -i 'vga\\|3d' | head -1")
		output, err = cmd.Output()
	case "windows":
		cmd := exec.Command("powershell", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -First 1 -ExpandProperty Name")
		output, err = cmd.Output()
	case "darwin":
		cmd := exec.Command("system_profiler", "SPDisplaysDataType")
		output, err = cmd.Output()
	default:
		return "Unknown"
	}

	if err != nil {
		return "Not detected"
	}

	return strings.TrimSpace(string(output))
}

func pollSystemInfo() {
	for {
		time.Sleep(10 * time.Second)
		info := getSystemInfo()
		if connected && wsConn != nil {
			wsConn.WriteJSON(Message{
				Type: "sysinfo",
				From: cfg.Name,
				Content: fmt.Sprintf("CPU: %.1f%%, RAM: %.1f%%, Disk: %.1f%%, GPU: %s",
					info.CPU, info.Memory, info.Disk, info.GPU),
			})
		}
	}
}

func register() error {
	localIP := getLocalIP()
	data := map[string]interface{}{
		"name":    cfg.Name,
		"address": localIP,
		"port":    cfg.Port,
		"status":  "online",
		"role":    Role,
		"system":  getSystemInfo(),
	}

	resp, err := http.Post(cfg.Coordinator+"/api/register", "application/json", strings.NewReader(toJSON(data)))
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("registration failed: %s", resp.Status)
	}

	log.Println("Registered with coordinator")
	return nil
}

func connectWebSocket() {
	for {
		url := strings.Replace(cfg.Coordinator, "http", "ws", 1) + "/ws?name=" + cfg.Name
		log.Printf("Connecting to WebSocket: %s", url)

		conn, _, err := websocket.DefaultDialer.Dial(url, nil)
		if err != nil {
			log.Printf("WebSocket connection failed: %v", err)
			time.Sleep(5 * time.Second)
			continue
		}

		wsConn = conn
		connected = true
		log.Println("Connected to coordinator")

		for {
			var msg Message
			err := conn.ReadJSON(&msg)
			if err != nil {
				log.Printf("WebSocket error: %v", err)
				connected = false
				break
			}
			handleMessage(msg)
		}

		time.Sleep(5 * time.Second)
	}
}

func handleMessage(msg Message) {
	switch msg.Type {
	case "ping":
		log.Printf("Received ping from %s", msg.From)
		sendToCoordinator(Message{Type: "pong", From: cfg.Name})
	case "message":
		msgText := fmt.Sprintf("[MESSAGE] From: %s | Content: %s", msg.From, msg.Content)
		fmt.Println(msgText)
		log.Printf("Message from %s: %s", msg.From, msg.Content)
		inbox <- msg
	case "participants":
		log.Println("=== Participant List Received ===")
		var participants []map[string]interface{}
		if err := json.Unmarshal([]byte(msg.Content), &participants); err != nil {
			log.Printf("Failed to parse participants: %v", err)
		} else {
			for i, p := range participants {
				log.Printf("%d. %s - %s:%d [%s]",
					i+1,
					p["name"],
					p["address"],
					p["port"],
					p["status"])
			}
		}
	default:
		log.Printf("Unknown message type: %s", msg.Type)
	}
}

func sendToCoordinator(msg Message) {
	if wsConn != nil && connected {
		wsConn.WriteJSON(msg)
	}
}

func startHTTPServer() {
	r := gin.Default()

	r.GET("/ws", func(c *gin.Context) {
		serveWS(c.Writer, c.Request)
	})

	r.GET("/status", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"name":   cfg.Name,
			"status": "online",
			"role":   Role,
		})
	})

	r.GET("/sysinfo", func(c *gin.Context) {
		c.JSON(200, getSystemInfo())
	})

	log.Printf("HTTP server starting on :%d", cfg.Port)
	if err := r.Run(fmt.Sprintf(":%d", cfg.Port)); err != nil {
		log.Printf("HTTP server error: %v", err)
	}
}

func serveWS(w http.ResponseWriter, r *http.Request) {
	upgrader := websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool { return true },
	}

	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("Upgrade error: %v", err)
		return
	}
	defer conn.Close()

	log.Println("Peer connected directly")

	for {
		var msg Message
		err := conn.ReadJSON(&msg)
		if err != nil {
			break
		}

		if msg.Type == "ping" {
			conn.WriteJSON(Message{Type: "pong", From: cfg.Name, Status: "online"})
		} else if msg.Type == "message" {
			log.Printf("Direct message from %s: %s", msg.From, msg.Content)
			conn.WriteJSON(Message{Type: "delivered", From: cfg.Name})
		}
	}
}

func handleInput() {
	scanner := bufio.NewScanner(os.Stdin)
	fmt.Println("\n===========================================")
	fmt.Println("Commands: ping <name>, send <name> <message>, list, myinfo, exit")
	fmt.Println("Your node name:", cfg.Name)
	fmt.Println("===========================================")

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Fields(line)

		if len(parts) == 0 {
			continue
		}

		switch parts[0] {
		case "ping":
			if len(parts) < 2 {
				fmt.Println("Usage: ping <name>")
				continue
			}
			target := parts[1]
			log.Printf("Pinging %s...", target)
			sendToCoordinator(Message{Type: "ping", Target: target})

		case "send":
			if len(parts) < 3 {
				fmt.Println("Usage: send <name> <message>")
				continue
			}
			target := parts[1]
			content := strings.Join(parts[2:], " ")
			log.Printf("Sending to %s: %s", target, content)
			sendToCoordinator(Message{Type: "message", Target: target, Content: content})

		case "list":
			log.Println("Requesting participant list from coordinator...")
			sendToCoordinator(Message{Type: "get_participants"})

		case "myinfo":
			info := getSystemInfo()
			fmt.Printf("System Info:\n  CPU: %.1f%%\n  Memory: %.1f%%\n  Disk: %.1f%%\n  GPU: %s\n",
				info.CPU, info.Memory, info.Disk, info.GPU)

		case "exit":
			return
		}
	}
}

func toJSON(v interface{}) string {
	b, _ := json.Marshal(v)
	return string(b)
}
