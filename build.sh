#!/bin/bash
set -e

echo "Building coordinator..."
go build -o bin/coordinator main.go

echo "Building participant..."
go build -o bin/participant participant/main.go

echo "Build complete!"
