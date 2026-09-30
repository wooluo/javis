# 贾维斯 Mac 部署指南

> 快照来源：台式机（Ryzen 7 9800X3D + RTX 5080）~/voxemw-deploy/ 定稿版，2026-09-30 同步。
> 本仓库不含密钥/模型/证书/venv——这些都按本指南在 Mac 上现场生成或下载。

## 架构

```
麦克风 → VAD(silero,CPU) → ASR(Qwen3-ASR-1.7B) → 大脑(GLM-5.3@智谱云) → TTS(edge-tts Brian +20%)
                                                                    ↓
                              网页 UI(8000/http, 9444/https) ← orchestrator ← pipeline(:8765 WS)
```

- 台式机实测：说完→答完 6.0~6.4s，edge 后端显存 ~5.8G（Mac 无 N 卡则跑 CPU/MPS，见「性能预期」）
- TTS 定稿：`en-US-BrianMultilingualNeural` rate +20%（可直接合成中文，贾维斯气质）
- 人设：`personas/jarvis.md`（人设与音色独立——改人设免重启，改音色要改 yaml + 重启管线）

## 1. 前置依赖

```bash
# Homebrew（没有就装）
brew install python@3.11 ffmpeg git

# 克隆
git clone git@github.com:wooluo/javis.git ~/javis && cd ~/javis
```

## 2. 环境与依赖

```bash
python3.11 -m venv main-env
source main-env/bin/activate
pip install torch torchvision  # Mac 用 CPU/MPS 版（无 cu130）
pip install --no-deps speech-to-speech && pip check   # 按缺补齐，装完必查
pip install edge-tts openai soundfile scipy nltk
# silero-vad 离线缓存
python -c "import torch; torch.hub.load('snakers4/silero-vad', 'hub_ui')"
```

## 3. 上游依赖坑（台式机实测记录）

- `speech-to-speech` 必须 `--no-deps` 再手动补依赖；缺啥 `pip install` 后必须 `pip check`
- nltk.download 顶层无超时且代理下会挂死 → 台式机已补丁为 no-op；Mac 首次运行若卡住，看「排障」
- smart-turn 模型文件名是 `smart-turn-v3.2-cpu.onnx`（v3.2 不是 v3）
- GLM thinking：yaml `thinking_budget_tokens: 1024` → low 档；延迟敏感选 `glm-5.3-flash`（关思考 0 chunk）

## 3. 密钥

```bash
cp .env.example .env.local && chmod 600 .env.local
# 编辑 .env.local：GLM_API_KEY=你的智谱 key（台式机的 key 不入仓库）
```

## 4. 配置

```bash
cp configs/assistant-5080.yaml configs/assistant.yaml
```

编辑 `configs/assistant.yaml`：
- `asr/stt/vad` 模型路径指向本地 HF 缓存（首次运行自动下载到 `~/.cache/huggingface`）
- `tts: {backend: edge, voice: en-US-BrianMultilingualNeural, rate: "+20%", volume: "+0%", pitch: "+0Hz"}`
- `personas.default: jarvis`
- `personas.list` 加 `jarvis: personas/jarvis.md`

## 5. TLS 证书（手机访问需要）

手机麦克风 getUserMedia 要求 https。Mac 上直接用仓库自带脚本：

```bash
scripts/make_lan_tls.sh   # 原脚本就是 Mac 专用（ipconfig/mkcert），生成 scripts/lan_tls/
```

注意：证书 SAN 必须含 Mac 局域网 IP；rootCA.pem 放 web/ 下供手机下载安装（iPhone：装描述文件 + 设置→通用→关于本机→证书信任设置里勾选，两步缺一不可）。

## 6. 启动与验证

```bash
chmod +x start_voxemw.sh start_orchestrator.sh
./start_voxemw.sh &        # 语音管线 :8765，模型加载 ~40s
./start_orchestrator.sh &  # 网页 :8000 + TLS :9444
```

冒烟测试（服务起来后必做）：

```bash
python3 scripts/e2e_smoke.py
# 通过标准：ASR 转写正确 + response.output_audio.delta 有 PCM + response.done
```

浏览器打开 `http://127.0.0.1:8000`（本机）或 `https://<Mac局域网IP>:9444`（手机）。

## 7. 排障速查（台式机实战全记录）

| 症状 | 病根 | 处置 |
|---|---|---|
| 又慢又粗（语速慢1.5倍+音调低沉） | TTS 输出采样率 bug：edge mp3 实际 24k 被当 16k 播 | 查 tts_edge.py 重采样条件必须是 `if sr != PIPELINE_SAMPLE_RATE`；比值=铁证（日志时长/直读时长≈1.50） |
| 手机连不上/麦克风不可用 | http 下 getUserMedia 被禁 | 走 :9444 TLS 入口 + 手机装 CA 证书 |
| 「贾维斯不在线」/网页连不上 | 旧标签页/僵尸 WS 占着唯一 slot | 关旧标签页重连，槽位几十秒自愈，别急着重启 |
| 重启管线后网页无反应 | orchestrator 对 8765 断连×7 后放弃该会话 | 刷新页面/重开会话（必须交代给使用者） |
| ASR/判停模型加载卡死 | nltk.download 代理挂死 | 台式机已补丁 no-op；Mac 检查同款位置 |
| edge-tts 变慢 | 微软 CDN 晚高峰抖动 | 过时段自愈，勿误判配置问题 |
| 判活误报 | pgrep -f 匹配自身查询 shell | 用 `pgrep -af "main-env/bin/python -m voxemw"` |
| 重启窗口 40~50s | 模型重载 | kill 单独跑，别和其他命令串同一 terminal 调用 |

## 性能预期（Mac 无 N 卡）

台式机（5080）说完→答完 6.0s。Mac 走 CPU/MPS 推理 ASR+VAD 会显著变慢（预估 2~4 倍），若不可接受：
- 换 `glm-5.3-flash`（云端大脑不变）
- 或 ASR 模型换小杯（yaml `asr.model`）
- TTS 是云端 edge-tts，不受影响

## 迁移清单（台式机 → Mac）

- [x] 代码快照（本仓库，orphan 分支无历史包袱）
- [x] 启动脚本（模板化路径，VOX_PYTHON 可覆盖）
- [x] 部署手册（本文件）
- [ ] `.env.local`（Mac 现场填 GLM key）
- [ ] 模型（HF 自动下载：Qwen3-ASR-1.7B、silero-vad、smart-turn v3.2）
- [ ] TLS 证书（make_lan_tls.sh 现场生成）
