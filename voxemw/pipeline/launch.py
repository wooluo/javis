"""VoxEMW 语音管线启动器：YAML 积木配置 → speech-to-speech realtime 服务。

用法：
    python -m voxemw.pipeline.launch [--config configs/assistant.yaml] [--dry-run]

- 上游 2026-08 重构（main @5a0c79f）：工厂函数废弃，改 BackendSpec 注册表。
  我们先 register_custom_backends() 把 qwen3asr 插进注册表，
  之后 --stt qwen3asr 就是合法 CLI 参数，走标准 parse/serve 流程。
- persona 人设不进管线进程：realtime 模式下 instructions 由客户端
  （voxemw.gateway.orchestrator）经 session.update 注入。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from voxemw.config import load_config, load_dotenv  # noqa: E402
from voxemw.pipeline.args import render_s2s_argv  # noqa: E402

DEFAULT_CONFIG = "configs/assistant.yaml"


def _patch_torch_flex_attention_compat() -> None:
    """transformers 5.13 的 flex_attention 集成模块顶层 import torch 2.9 才有的
    AuxRequest，torch 2.8 没有 → ImportError 炸在 import 链上（我们的模型根本不用
    flex attention，占位即可）。"""
    try:
        from torch.nn.attention.flex_attention import AuxRequest  # noqa: F401
    except ImportError:
        import torch.nn.attention.flex_attention as _fa

        class AuxRequest:  # 占位：torch 2.9 才有真身；flex attention 不会被调用
            pass

        _fa.AuxRequest = AuxRequest


def _patch_torch_hub_offline_fallback() -> None:
    """silero VAD 走 torch.hub.load("snakers4/silero-vad"):每次启动都向
    github.com 发校验请求,而本机 GitHub 时通时断,断则启动失败。
    加离线兜底:网络失败且本地缓存存在时,source="local" 从缓存加载。"""
    from pathlib import Path

    import torch

    orig_load = torch.hub.load

    def _load_with_local_fallback(repo_or_dir, model, *args, **kwargs):
        try:
            return orig_load(repo_or_dir, model, *args, **kwargs)
        except Exception:
            if repo_or_dir == "snakers4/silero-vad" and kwargs.get("source", "github") == "github":
                local = Path(torch.hub.get_dir()) / "snakers4_silero-vad_master"
                if local.is_dir():
                    kwargs.pop("force_reload", None)
                    return orig_load(str(local), model, *args, source="local", **kwargs)
            raise

    torch.hub.load = _load_with_local_fallback


def _patch_llm_channel_strip() -> None:
    """LM Studio gemma 模板在工具回合后会把思维通道头原样漏进 content
    （实测 2026-09-30：'<|channel>thought\\n<channel|>' + 正文；chat_template_kwargs
    enable_thinking=false 服务端无视）。在 TextDelta 产生点包一层有状态过滤：
    剥完整 thought 段（含思维内容）与一切通道标记变体，跨 delta 截断的标记
    尾巴扣住待下一片拼接，杜绝 TTS 把壳念出来。"""
    import re

    from speech_to_speech.LLM import chat_completions_language_model as cc

    # 完整 thought 段：<|channel>thought …（思维内容）…<|channel|>final → 连 final 标签一并剥
    thought_re = re.compile(
        r"<\|?channel\|?>\s*thought[\s\S]*?<\|?channel\|?>\s*(final)?", re.IGNORECASE)
    # 未闭合的 thought 开头（闭标记还没流到 → 整段扣住，防思维内容漏出）
    open_re = re.compile(r"<\|?channel\|?>\s*thought", re.IGNORECASE)
    # 一切残留标记变体（channel/message/end/system，正反斜杠混排的 malformed 也算）
    marker_re = re.compile(r"<\|?[a-z]+\|?>", re.IGNORECASE)
    # 回合开头的裸 thought 引导词（标记先被剥掉/服务端已滤时，只剩单词——
    # 2026-09-30 实测流式下 <|channel> 与 thought 分包到达就会这样）
    lead_re = re.compile(r"^thought(?=$|[\s\n])", re.IGNORECASE)
    # 流式截断防护：结尾像「未闭合标记开头」或「thought 撕开的半截」的片段扣住
    tail_re = re.compile(r"<[a-z|]*$|t(?:h(?:o(?:u(?:g(?:h(?:t)?)?)?)?)?)?$", re.IGNORECASE)

    class _Filter:
        def __init__(self):
            self.buf = ""
            self.started = False  # 回合开头才需要识别 thought 引导词

        def _strip_head(self):
            """回合开头：剥标记 → 剥裸 thought 引导词。返回是否仍需扣住。"""
            self.buf = thought_re.sub("", self.buf, count=1)
            if open_re.search(self.buf):
                return True  # 未闭合 thought 段：全扣
            self.buf = marker_re.sub("", self.buf)
            self.buf = lead_re.sub("", self.buf, count=1)
            return False

        def feed(self, text):
            self.buf += text
            if not self.started:
                held = self._strip_head()
                if held:
                    return ""  # 未闭合 thought 段：全扣等闭标记
                self.buf = self.buf.lstrip("\n\r \t")  # 引导词剥后的残余换行
                if not self.buf:
                    return ""  # 开头全是壳/引导词，等内容
            else:
                self.buf = thought_re.sub("", self.buf, count=1)
                if open_re.search(self.buf):
                    return ""
                self.buf = marker_re.sub("", self.buf)
            m = tail_re.search(self.buf)
            if m:
                out, self.buf = self.buf[: m.start()], m.group(0)
            else:
                out, self.buf = self.buf, ""
            if out.strip():
                self.started = True
            elif not self.started:
                out = ""  # 开场纯空白（thought 剥后残余换行等）：吞掉
            return out

        def flush(self):
            out, self.buf = self.buf, ""
            if not self.started:
                return ""  # 开场扣住的全是壳/引导词/未闭合思维：整体丢弃
            m = open_re.search(out)
            if m:
                out = out[: m.start()]  # 流结束仍未闭合的 thought 段：从开口处丢弃
            out = marker_re.sub("", out)
            out = re.sub(r"<[a-z|]*$", "", out)  # 尾部残缺标记片段
            return out

    def _wrap(orig):
        def _iter(api_response):
            f = _Filter()
            for ev in orig(api_response):
                text = getattr(ev, "text", None)
                if isinstance(text, str) and text:
                    ev.text = f.feed(text)
                    if ev.text:
                        yield ev
                    # 清洗后暂空（扣住的截断尾巴/未闭合段）：吞掉本片等内容到齐
                else:
                    yield ev
            tail = f.flush()
            if tail:
                yield cc.TextDelta(text=tail)
        return _iter

    cc._iter_chat_stream_events = _wrap(cc._iter_chat_stream_events)
    cc._iter_chat_response_events = _wrap(cc._iter_chat_response_events)


def _patch_smart_turn_gpu() -> None:
    """上游 SmartTurnAnalyzer 硬编 CPUExecutionProvider。smart_turn_model_path 指到
    *-gpu.onnx 且 CUDA 可用时换成 GPU 优先（复核 ~80ms → ~10ms）。
    做法：包一层 __init__，建完 CPU 会话后原地重建成 CUDA（模型 20MB，双载无感）。
    ⚠️ 2026-08-17 实测：与 AVTR-1 TRT 渲染在同卡上初始化互斥（pipeline 必炸
    illegal memory access），当前配置用 CPU 版模型，本补丁不触发。保留备用——
    换双卡或 TRT 冲突解决后可重新启用。"""
    import logging

    import onnxruntime as ort
    from speech_to_speech.VAD import smart_turn as st_mod

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return  # 环境没装 onnxruntime-gpu，不动

    logger = logging.getLogger(__name__)
    orig_init = st_mod.SmartTurnAnalyzer.__init__

    def _init_gpu(self, **kw):
        orig_init(self, **kw)
        mp = str(kw.get("model_path") or "")
        if mp.endswith("-gpu.onnx"):
            self.session = ort.InferenceSession(
                mp, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            logger.info("SmartTurn 走 GPU: %s", mp)

    st_mod.SmartTurnAnalyzer.__init__ = _init_gpu


_STAGE_DIRECTION = re.compile(r"[（(][^（）()]{1,20}[)）]")


def strip_stage_directions(text: str) -> str:
    """删除括号舞台指示（（乐）（拍大腿）等——纯函数，便于单测）。

    LLM 偶尔输出这类动作标注（人设禁止但没强制力），不剥掉 TTS 会照字面
    把「括号乐」念出来。必须在句子级调用（delta 级括号对会被 token 切开）。"""
    return _STAGE_DIRECTION.sub(" ", text)


def main() -> None:
    parser = argparse.ArgumentParser(description="VoxEMW 数字人语音管线启动器")
    parser.add_argument(
        "--config",
        default=os.environ.get("VOXEMW_CONFIG", DEFAULT_CONFIG),
        help="YAML 配置路径（默认 %(default)s，可用 VOXEMW_CONFIG 覆盖）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印渲染出的 s2s argv，不启动管线（本机无 GPU/无依赖也可跑）",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.is_file():
        sys.exit(f"ERROR: 配置不存在: {config_path}")

    load_dotenv(REPO_ROOT / ".env.local")
    config = load_config(config_path)
    argv = render_s2s_argv(config, env=os.environ)

    if args.dry_run:
        pairs = [f"{argv[i]} {argv[i + 1]}" for i in range(0, len(argv) - 1, 2)]
        print("speech-to-speech \\")
        print("  " + " \\\n  ".join(pairs))
        print(f"\n# stt={config['stt']['backend']} / tts={config['tts']['backend']}（运行时注册，不在 CLI 里）")
        return

    # ── 以下需要 speech_to_speech 依赖与 GPU，仅在服务器上执行 ──
    # 注意顺序：ModuleArguments 的 stt/tts choices 在 import 时固化，
    # 必须先注册自定义积木再 import s2s_pipeline，否则 --stt qwen3asr 报 invalid choice
    from voxemw.pipeline.backends import register_custom_backends

    register_custom_backends(config)

    import speech_to_speech.s2s_pipeline as s2s

    _patch_torch_flex_attention_compat()
    _patch_torch_hub_offline_fallback()
    _patch_smart_turn_gpu()
    _patch_llm_channel_strip()

    # 新上游标准 serve 流程（s2s_pipeline.run_pipeline_command 复刻）
    parsed = s2s.parse_arguments(argv, command="serve")
    s2s.setup_logger(parsed.module_kwargs.log_level)
    s2s.prepare_all_args(parsed)

    from threading import Event

    stop_event = Event()
    pipeline_manager = s2s.build_pipeline(parsed, stop_event)

    import signal

    def _shutdown(_sig, _frame):
        pipeline_manager.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    pipeline_manager.start()
    pipeline_manager.wait()


if __name__ == "__main__":
    main()
