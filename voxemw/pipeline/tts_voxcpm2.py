"""VoxCPM2 TTS 积木（5080 满血版）：官方 PyTorch + CUDA。

两种出声模式（按人设 frontmatter 决定）：
- 音色设计：persona 带 voice_control 描述词 → 描述词拼进 text（VoxCPM2 的
  (control)text 约定），不给参考音，模型按描述凭空造嗓音
- 零样本克隆：persona 只有 ref_wav/ref_text → 传统克隆路径

与上游 qwen3 handler 同契约：process(TTSInput) → 产出 int16 16k mono 音频块，
结束补 AUDIO_RESPONSE_DONE。VoxCPM2 出 48kHz，resample_poly 降到管线 16kHz。

2026-09-28 提速改造（贾维斯固化）：
- setup 一次性 build_prompt_cache 固化参考音特征（读 wav + VAE encode 只做一次），
  process 直接走 _generate_with_prompt_cache，每句省 ~0.5-1s；
- 克隆模式锁定（voice_control 已从 jarvis.md 注释掉）。
原版备份: backups/20260928-voxcpm2-switch/tts_voxcpm2.py.orig
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from threading import Event

from collections import deque

import numpy as np

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE

from voxemw.pipeline.launch import strip_stage_directions

logger = logging.getLogger(__name__)

VCPM_SAMPLE_RATE = 48000
PIPELINE_SAMPLE_RATE = 16000


class LoudnessNormalizer:
    """逐句响度锚定（2026-09-29 跨句响度漂移修复）。

    实测克隆模式跨句 RMS 极差 4~5dB（钉种子无效，A/B 验证），响度是模型逐句
    生成的固有方差。本类以参考音 RMS 为目标做流式自适应增益：
    - 200ms 滑窗估当前响度 → 增益 = 目标/当前，±6dB 限幅
    - 单极点慢平滑（alpha=0.06，~0.3s 适应）防音节级抽泵
    - 句间 reset：每句从单位增益起步，句内动态（强调/轻声）保留
    """

    def __init__(self, target_db: float, sr: int = VCPM_SAMPLE_RATE, win_s: float = 0.2,
                 max_gain_db: float = 6.0, alpha: float = 0.06,
                 sub_block: int = 1024, silence_db: float = -45.0) -> None:
        self.target_lin = float(10 ** (target_db / 20))
        self.win: deque = deque(maxlen=int(sr * win_s))
        self.max_gain = 10 ** (max_gain_db / 20)
        self.alpha = alpha
        self.sub_block = sub_block
        self.silence_lin = float(10 ** (silence_db / 20))
        self.gain = 1.0

    def reset(self) -> None:
        # 只清测量窗，保留收敛增益（暖启动）：短句 2~4s 从 1.0 冷启动自适应
        # 吃掉半句（实测极差只从 2.9 压到 2.3dB）；同音色连续句响度相近，
        # 沿用上句收敛增益 = 下一句瞬时校正。真换声（setup 重建）才会回 1.0。
        self.win.clear()

    def process(self, x: np.ndarray) -> np.ndarray:
        """按 1024 样本子块自适应：块内增益恒定（向量化的乘法），块间更新。

        增益按子块更新而非按调用更新——chunk 可能是几百 ms 大块，按调用
        更新 α=0.06 一句只走一两步根本不收敛（离线单测踩过）。静音窗
        （RMS < -45dBFS）不更新，防句首静音把增益误拉满。
        """
        xs = np.asarray(x, dtype=np.float32).reshape(-1)
        out = np.empty_like(xs)
        for i in range(0, xs.size, self.sub_block):
            seg = xs[i:i + self.sub_block]
            self.win.extend((seg * seg).tolist())
            rms = float(np.sqrt(np.mean(self.win)) + 1e-8)
            if rms > self.silence_lin:
                raw = float(np.clip(self.target_lin / rms, 1.0 / self.max_gain, self.max_gain))
                self.gain += self.alpha * (raw - self.gain)
            out[i:i + self.sub_block] = np.clip(seg * self.gain, -1.0, 1.0)
        return out


class VoxCPM2TTSHandler(BaseHandler):
    """VoxCPM2 零样本克隆 TTS handler。"""

    def setup(
        self,
        should_listen: Event,
        model_name: str = "openbmb/VoxCPM2",
        ref_audio: str | Path | None = None,
        ref_text: str = "",
        voice_control: str = "",
        voice_seed: int = 42,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        device: str = "cuda",
        blocksize: int = 512,
        cancel_scope=None,
        speculative_turns=None,
        **_unused,
    ) -> None:
        from voxcpm import VoxCPM

        self.should_listen = should_listen
        self.blocksize = blocksize
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.voice_control = (voice_control or "").strip()
        self.voice_seed = int(voice_seed)  # 设计模式钉种子：每句同一噪声起点，音色逐句一致
        # 设计模式不需要参考音；voice_control 优先于 ref_*（两个 ref 参数必须同 None）
        self.ref_audio = None if self.voice_control else (str(ref_audio) if ref_audio else None)
        self.ref_text = None if self.voice_control else ref_text
        self.cfg_value = cfg_value
        self.inference_timesteps = inference_timesteps

        logger.info("加载 VoxCPM2: %s (device=%s, 模式=%s)", model_name, device,
                    "音色设计" if self.voice_control else "零样本克隆")
        t0 = time.perf_counter()
        self.model = VoxCPM.from_pretrained(model_name, load_denoiser=False)
        logger.info("VoxCPM2 加载完成 %.1fs", time.perf_counter() - t0)

        # 提速改造: setup 一次性固化 prompt_cache(读wav+VAE encode)，
        # process 不再每次重 build（每句省 ~0.5-1s）
        # 0929 修「好的摸」句首幽灵音：改用 reference 克隆模式（结构性隔离，
        # 参考音只定音色不续写）。旧 continuation 模式每句被当参考音「…效劳的
        # 吗？」的续篇，起手回声「好的吗」——离线复刻实测 ASR 复现。
        self.prompt_cache = None
        if self.ref_audio:
            t0 = time.perf_counter()
            self.prompt_cache = self.model.tts_model.build_prompt_cache(
                reference_wav_path=self.ref_audio,
            )
            logger.info("prompt_cache 固化完成 %.1fs mode=reference (ref=%s)",
                        time.perf_counter() - t0, self.ref_audio)

        # 响度锚定：克隆模式以参考音 RMS 为目标；设计模式无锚，fallback -20 dBFS
        target_db = self._measure_ref_db(self.ref_audio)
        self._norm = LoudnessNormalizer(target_db)
        logger.info("响度锚定: 目标 RMS %.1f dBFS (ref=%s)", target_db, self.ref_audio)

        # 预热（首次调用编译开销大）；预热也走 cache 路径，顺便验证固化结果
        t0 = time.perf_counter()
        if self.prompt_cache is not None:
            self._gen_with_cache = True
            gen = self.model.tts_model.generate_with_prompt_cache_streaming(
                target_text="你好。",
                prompt_cache=self.prompt_cache,
                cfg_value=self.cfg_value,
                inference_timesteps=self.inference_timesteps,
            )
            for _wav, _prev, _tok in gen:
                pass
        else:
            self._gen_with_cache = False
            list(self.model.generate_streaming(
                text=self._design_text("你好。"), prompt_wav_path=self.ref_audio,
                prompt_text=self.ref_text,
                cfg_value=self.cfg_value, inference_timesteps=self.inference_timesteps,
            ))
        logger.info("VoxCPM2 预热完成 %.1fs", time.perf_counter() - t0)

    def _measure_ref_db(self, path: str | None) -> float:
        """读参考音整体 RMS（dBFS）。读不到（设计模式/文件缺失）回 -20 dBFS。"""
        if not path:
            return -20.0
        try:
            from scipy.io import wavfile
            sr, data = wavfile.read(path)
            x = data.astype(np.float32) / (32768.0 if data.dtype == np.int16 else 1.0)
            x = x.reshape(-1)
            if x.size < sr // 4:
                return -20.0
            return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9))
        except Exception as e:  # noqa: BLE001 — 锚定失败不挡主链路
            logger.warning("参考音响度测量失败(%s)，回 -20 dBFS", e)
            return -20.0

    def _design_text(self, text: str) -> str:
        """音色设计模式：按 VoxCPM2 的 (control)text 约定拼描述词。"""
        return f"({self.voice_control}){text}" if self.voice_control else text

    def _to_pipeline_pcm(self, chunk: np.ndarray) -> bytes:
        """48k float32 → 16k int16（resample_poly 3:1 抽取）。"""
        from scipy.signal import resample_poly

        x = np.asarray(chunk, dtype=np.float32).reshape(-1)  # 2D(1,T)展平: cache流式yield的是squeeze(1)后2D张量
        if x.size < 8:
            return b""
        # 去掉 3 的倍数余量，边界样本留给下一块（防分块伪影）
        usable = (x.size // 3) * 3
        out = resample_poly(x[:usable], 1, 3)
        int16 = np.clip(out * 32768.0, -32768, 32767).astype(np.int16)
        return int16.tobytes()

    def process(self, tts_input):
        # 句子级入口（上游按 sentence batch 下发，括号对完整）：
        # 剥掉 LLM 偶尔冒出的舞台指示（（乐）（拍大腿）），防照字面念出。
        text = strip_stage_directions(getattr(tts_input, "text", "") or "").strip()
        if not text:
            yield AUDIO_RESPONSE_DONE
            return
        # 设计模式的描述词在剥括号之后拼（剥括号只针对 LLM 产出的舞台指示）
        text = self._design_text(text)
        if self.voice_control:
            # 钉种子：设计模式每次生成默认从随机噪声采样（同描述词音色逐句漂移），
            # 固定种子让每句从同一噪声起点出发，音色确定且一致
            import torch
            torch.manual_seed(self.voice_seed)
            torch.cuda.manual_seed_all(self.voice_seed)
        t0 = time.perf_counter()
        n_samples = 0
        first = True
        self._norm.reset()  # 句间重置：句内动态保留，跨句锚定同一目标
        if self._gen_with_cache:
            gen = self.model.tts_model.generate_with_prompt_cache_streaming(
                target_text=text,
                prompt_cache=self.prompt_cache,
                cfg_value=self.cfg_value,
                inference_timesteps=self.inference_timesteps,
            )
        else:
            gen = self.model.generate_streaming(
                text=text,
                prompt_wav_path=self.ref_audio,
                prompt_text=self.ref_text,
                cfg_value=self.cfg_value,
                inference_timesteps=self.inference_timesteps,
            )
        for chunk in gen:
            # cache 路径产出 (wav, prev, tokens) 三元组，取 wav；普通路径直接是 wav
            if isinstance(chunk, tuple):
                chunk = chunk[0]
            if self.cancel_scope is not None and self.cancel_scope.discarding:
                break  # 打断：本轮输出已被废弃
            pcm = self._to_pipeline_pcm(self._norm.process(chunk))
            if not pcm:
                continue
            n_samples += len(pcm) // 2
            if first:
                logger.info("VoxCPM2 TTFA: %.2fs", time.perf_counter() - t0)
                first = False
            for i in range(0, len(pcm), self.blocksize):
                yield pcm[i:i + self.blocksize]
        dur = n_samples / PIPELINE_SAMPLE_RATE
        logger.info("VoxCPM2 生成 %.2fs 音频，耗时 %.2fs（RTF %.2f）",
                    dur, time.perf_counter() - t0,
                    (time.perf_counter() - t0) / max(dur, 1e-6))
        yield AUDIO_RESPONSE_DONE
