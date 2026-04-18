#!/bin/bash
set -e

echo "Building for Linux..."
GOOS=linux GOARCH=amd64 go build -o bin/linux/amd64/coordinator main.go
GOOS=linux GOARCH=amd64 go build -o bin/linux/amd64/participant participant/main.go
GOOS=linux GOARCH=arm64 go build -o bin/linux/arm64/coordinator main.go
GOOS=linux GOARCH=arm64 go build -o bin/linux/arm64/participant participant/main.go

echo "Building for macOS..."
GOOS=darwin GOARCH=amd64 go build -o bin/darwin/amd64/coordinator main.go
GOOS=darwin GOARCH=amd64 go build -o bin/darwin/amd64/participant participant/main.go
GOOS=darwin GOARCH=arm64 go build -o bin/darwin/arm64/coordinator main.go
GOOS=darwin GOARCH=arm64 go build -o bin/darwin/arm64/participant participant/main.go

echo "Building for Windows..."
GOOS=windows GOARCH=amd64 go build -o bin/windows/amd64/coordinator.exe main.go
GOOS=windows GOARCH=amd64 go build -o bin/windows/amd64/participant.exe participant/main.go

echo "All builds complete!"
ls -la bin/*/amd64/
ls -la bin/*/arm64/
ls -la bin/windows/amd64/
