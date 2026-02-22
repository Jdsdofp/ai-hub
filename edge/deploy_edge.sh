#!/bin/bash
# SmartX Vision Platform v3 — Edge Deployment Script
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "==========================================="
echo " SmartX Vision v3 — Edge Deployment"
echo "==========================================="

# Check GPU
if command -v nvidia-smi &>/dev/null; then
    echo "[OK] GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
else
    echo "[WARNING] No GPU detected. Running on CPU."
fi

# Check .env
if [ ! -f .env ]; then
    if [ -f edge/.env.edge ]; then
        cp edge/.env.edge .env
        echo "[OK] Copied edge/.env.edge -> .env (edit credentials)"
    else
        echo "[ERROR] No .env file found"
        exit 1
    fi
fi

case "${1:-}" in
    --stop)
        echo "Stopping edge services..."
        docker compose -f edge/docker-compose.edge.yml down
        echo "[OK] Stopped"
        ;;
    --update-model)
        echo "Pulling latest model from server..."
        echo "[TODO] Configure MQTT model sync"
        ;;
    *)
        echo "Building and starting edge services..."
        docker compose -f edge/docker-compose.edge.yml up -d --build
        sleep 5
        # Health check
        IP=$(hostname -I | awk '{print $1}')
        if curl -sf http://localhost:5000/health > /dev/null 2>&1; then
            echo ""
            echo "==========================================="
            echo " [OK] Edge device running!"
            echo " Dashboard: http://${IP}:5000"
            echo " API:       http://${IP}:5000/api/v1/epi"
            echo " Health:    http://${IP}:5000/health"
            echo "==========================================="
        else
            echo "[WARNING] Health check failed. Check logs:"
            echo "  docker logs smartx-vision-edge"
        fi
        ;;
esac
