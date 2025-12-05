#!/bin/bash
# Convenience script to rebuild and restart the Docker container

set -e  # Exit on error

# Clear log file
> src/tmp/setup.log

echo "🔄 Stopping existing container..."
docker compose down >> src/tmp/setup.log 2>&1

echo "🔨 Rebuilding Docker image..."
docker compose build >> src/tmp/setup.log 2>&1

echo "🚀 Starting container..."
docker compose up -d >> src/tmp/setup.log 2>&1

echo "✅ Container ready!"
