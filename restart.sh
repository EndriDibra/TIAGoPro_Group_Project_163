#!/bin/bash
# Convenience script to rebuild and restart the Docker container

set -e  # Exit on error

echo "🔄 Stopping existing container..."
docker compose down

echo "🔨 Rebuilding Docker image..."
docker compose build

echo "🚀 Starting container..."
docker compose up -d

echo "✅ Container ready! Entering shell..."
docker compose exec -it tiago_sim bash
