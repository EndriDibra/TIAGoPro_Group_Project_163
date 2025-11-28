#!/bin/bash
# Convenience script to enter the Docker container

set -e  # Exit on error

echo "✅ Entering shell..."
docker compose exec -it tiago_sim bash
