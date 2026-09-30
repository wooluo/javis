#!/bin/bash
# VoxEMW 语音管线启动（模板：Mac 部署请按 README-MAC.md 改三处路径）
# 台式机原版：set -a; source /home/wooluo/voxemw-deploy/voxemw-app/.env.local; set +a
set -a; source "$(dirname "$0")/.env.local"; set +a
# HF 缓存离线模式（模型全量落地后开启；首次部署需联网下载，先注释掉这行）
# export HF_HUB_OFFLINE=1
# 清空代理（socks/http 代理会劫持 httpx；GLM 直连不需要代理）
unset ALL_PROXY all_proxy HTTPS_PROXY https_proxy HTTP_PROXY http_proxy
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"
mkdir -p "$APP_DIR/logs"
exec "${VOX_PYTHON:-python3}" -m voxemw.pipeline.launch --config configs/assistant.yaml >> "$APP_DIR/logs/runtime.log" 2>&1
