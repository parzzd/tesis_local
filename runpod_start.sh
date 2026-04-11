#!/usr/bin/env bash
# Bootstrap del servidor de inferencia en RunPod tras un Stop/Start.
# Reinstala tmux, deps Python, corrige el conflicto opencv y lanza uvicorn.
#
# Uso (dentro del pod, por SSH):
#   cd /workspace/app && git pull && bash runpod_start.sh
set -e

APP_DIR="/workspace/app"
MODELS_DIR="/workspace/models_mix2"
POSE_WEIGHTS="/workspace/yolo11s-pose.pt"
PORT="${PORT:-8000}"
SESSION="infer"

echo "[runpod_start] apt: instalando tmux"
apt-get update -qq
apt-get install -y -qq tmux >/dev/null

echo "[runpod_start] pip: instalando deps de inferencia"
cd "$APP_DIR"
pip install --no-cache-dir -r requirements-inference.txt

echo "[runpod_start] pip: forzando opencv-python-headless (evita segfault con TF)"
pip uninstall -y opencv-python >/dev/null 2>&1 || true
pip install --no-cache-dir opencv-python-headless==4.10.0.84

echo "[runpod_start] git pull"
git pull --ff-only || true

echo "[runpod_start] matando sesion tmux previa si existe"
tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "[runpod_start] lanzando uvicorn en tmux ($SESSION) puerto $PORT"
tmux new -d -s "$SESSION" \
  "cd $APP_DIR && MODELS_DIR=$MODELS_DIR POSE_WEIGHTS=$POSE_WEIGHTS uvicorn app.server_inference:app --host 0.0.0.0 --port $PORT 2>&1 | tee /workspace/infer.log"

sleep 2
echo "[runpod_start] listo. Comprueba con:"
echo "  tmux attach -t $SESSION   (Ctrl+B luego D para salir)"
echo "  curl http://localhost:$PORT/health"
