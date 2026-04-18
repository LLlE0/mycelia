package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"

	"p2p-network/coordinator"
)

func main() {
	// MongoDB connection settings
	mongoURI := getEnv("MONGO_URI", "mongodb://localhost:27017")
	dbName := getEnv("MONGO_DB", "p2p_network")
	serverAddr := getEnv("SERVER_ADDR", ":8080")

	// Initialize database
	db, err := coordinator.NewDatabase(mongoURI, dbName)
	if err != nil {
		log.Fatalf("Failed to connect to MongoDB: %v", err)
	}
	defer db.Close(context.Background())

	log.Println("Connected to MongoDB")

	// Create and start server
	server := coordinator.NewServer(db, serverAddr)

	// Handle graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		if err := server.Start(); err != nil {
			log.Fatalf("Server error: %v", err)
		}
	}()

	log.Println("Coordinator server running on", serverAddr)
	<-quit
	log.Println("Shutting down server...")
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
