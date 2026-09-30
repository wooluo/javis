#!/bin/bash
# jarvis —— 贾维斯一键启停（Mac 本地版，2026-09-30）
# 用法：jarvis [start|stop|restart|status|logs]   默认 start
# 大脑=LM Studio(:1234 gemma) 管线=:8765 网页=:8000
set -u
JARVIS_DIR="$HOME/DEV/javis"
PY="$JARVIS_DIR/main-env/bin/python"
URL="http://127.0.0.1:8000"

say() { printf "\033[1;36m[jarvis]\033[0m %s\n" "$*"; }

# 本机服务一律直连：clash 代理会劫持 localhost 的 ws/http
unset ALL_PROXY all_proxy HTTPS_PROXY https_proxy HTTP_PROXY http_proxy
export no_proxy="127.0.0.1,localhost,::1"
export NO_PROXY="$no_proxy"

up() { pgrep -f "$1" >/dev/null 2>&1; }
port_open() { nc -z 127.0.0.1 "$1" >/dev/null 2>&1; }
lm_up() { curl -s --max-time 2 http://127.0.0.1:1234/v1/models >/dev/null 2>&1; }

case "${1:-start}" in
start)
  cd "$JARVIS_DIR" || exit 1
  [ -x "$PY" ] || { say "缺 main-env 环境，按 README-MAC.md 重新部署"; exit 1; }

  # 0) 大脑：LM Studio（本地 gemma，没起就拉应用并等 API）
  if lm_up; then
    say "大脑 LM Studio ✓ :1234"
  else
    say "大脑 LM Studio 未启动，拉起应用…"
    open -a "LM Studio" 2>/dev/null || { say "找不到 LM Studio.app，请手动启动（并确认 Server 已开）"; exit 1; }
    for i in $(seq 1 60); do
      lm_up && break
      sleep 2
    done
    lm_up && say "LM Studio 就绪" || say "LM Studio API 迟迟未就绪（Server 未开或模型加载慢），先继续…"
  fi

  # 1) 语音管线（ASR 加载 ~20s）。判活看端口不看进程——退到一半的垂死进程
  #    会被 pgrep 误报"在运行"（2026-09-30 restart 实测踩过）
  if port_open 8765; then
    say "语音管线已在运行 :8765"
  else
    up "voxemw.pipeline.launch" && { pkill -f "voxemw.pipeline.launch"; sleep 2; }
    say "启动语音管线（模型加载 ~20s）…"
    VOX_PYTHON="$PY" nohup ./start_voxemw.sh >/dev/null 2>&1 &
    ok=""
    for i in $(seq 1 36); do
      if tail -6 logs/runtime.log 2>/dev/null | grep -q "Uvicorn running"; then ok=1; break; fi
      up "voxemw.pipeline.launch" || { say "管线进程退出了：tail -30 logs/runtime.log 排障"; exit 1; }
      sleep 5
    done
    [ -n "$ok" ] && say "语音管线 ✓ :8765" || { say "管线就绪超时，查 logs/runtime.log"; exit 1; }
  fi

  # 2) 网页入口（同样以端口为准）
  if port_open 8000; then
    say "orchestrator 已在运行 :8000"
  else
    up "voxemw.gateway.orchestrator" && { pkill -f "voxemw.gateway.orchestrator"; sleep 2; }
    VOX_PYTHON="$PY" nohup ./start_orchestrator.sh >/dev/null 2>&1 &
    for i in $(seq 1 15); do
      curl -s -o /dev/null --max-time 2 "$URL" && break
      sleep 2
    done
    say "orchestrator ✓ $URL"
  fi

  open "$URL"
  say "贾维斯上线（浏览器已打开，麦克风记得点允许）"
  ;;
stop)
  pkill -f "voxemw.pipeline.launch" 2>/dev/null
  pkill -f "voxemw.gateway.orchestrator" 2>/dev/null
  for i in $(seq 1 15); do
    port_open 8765 || port_open 8000 || break
    sleep 1
  done
  say "语音管线与网页入口已停止（LM Studio 保持运行）"
  ;;
restart)
  "$0" stop; sleep 2; exec "$0" start
  ;;
status)
  port_open 8765 && s1="✓ 运行中 :8765" || s1="✗ 未运行"
  port_open 8000 && s2="✓ 运行中 :8000" || s2="✗ 未运行"
  lm_up && s3="✓ :1234" || s3="✗ 未运行"
  say "语音管线 $s1"
  say "网页入口 $s2"
  say "大脑 LM Studio $s3"
  say "日志：$JARVIS_DIR/logs/runtime.log + orchestrator.log（或 jarvis logs）"
  ;;
logs)
  touch "$JARVIS_DIR/logs/runtime.log" "$JARVIS_DIR/logs/orchestrator.log" 2>/dev/null
  tail -f "$JARVIS_DIR/logs/runtime.log" "$JARVIS_DIR/logs/orchestrator.log"
  ;;
*)
  say "用法：jarvis [start|stop|restart|status|logs]"
  ;;
esac
