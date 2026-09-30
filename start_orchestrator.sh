#!/bin/bash
# VoxEMW orchestrator 网页入口启动（模板：Mac 部署请按 README-MAC.md 改三处路径）
set -a; source "$(dirname "$0")/.env.local"; set +a
# export HF_HUB_OFFLINE=1
unset ALL_PROXY all_proxy HTTPS_PROXY https_proxy HTTP_PROXY http_proxy
# LAN TLS（手机麦克风需 https）：证书用 scripts/make_lan_tls.sh 在 Mac 上重新生成（原脚本是 Mac 专用）
export VOX_TLS_CERT="${VOX_TLS_CERT:-$PWD/scripts/lan_tls/cert.pem}"
export VOX_TLS_KEY="${VOX_TLS_KEY:-$PWD/scripts/lan_tls/key.pem}"
export VOX_TLS_PORT="${VOX_TLS_PORT:-9444}"
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"
mkdir -p "$APP_DIR/logs"
exec "${VOX_PYTHON:-python3}" -m voxemw.gateway.orchestrator --config configs/assistant.yaml >> "$APP_DIR/logs/orchestrator.log" 2>&1
