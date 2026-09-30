# -*- coding: utf-8 -*-
"""贾维斯工具层：天气 / 股票快照 / K线历史 / ZCode 编码代理 / Hermes 全技能桥。

设计原则（照抄 look_at_camera 模式）：
- 工具是增强，挂了不能拖累对话主链路——所有异常静默兜底。
- 全部异步（run_in_executor 包阻塞 IO），不卡 orchestrator 事件循环。
- 慢工具（zcode/hermes）超时回"后台办理中"话术，不吊死会话。

数据源：
- 天气：wttr.in 免费无 key（Open-Meteo 备胎）。
- 股票快照：腾讯 qt.gtimg.cn（记忆约定：外网请求绕开 clash 代理）。
- K线历史：本地缓存 ~/oddindicators/data/klines_cache/*.json
  （5367 只全量历史，股票名→代码映射 pickle 常驻）。
- ZCode：`zcode.cjs -p "<task>" --cwd <dir>` 无头模式——编码/修 bug/
  文件操作/跑命令/Git 等开发类任务的专属通道，超时转后台继续跑，
  check_zcode 可查进度与结果（日志落盘 logs/zcode_jobs/）。
- Hermes：`hermes -z "<prompt>"` 无头模式，贾维斯自然语言点菜 →
  Hermes 带全部本地技能干活 → 返回结论（查资料/分析等非开发类杂务）。
"""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import pickle
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ── 路径 ─────────────────────────────────────────────
_KLINE_DIR = os.path.expanduser("~/oddindicators/data/klines_cache")
_NAME_MAP_PKL = Path(__file__).resolve().parent / "data" / "stock_names.pkl"
_HERMES_BIN = os.path.expanduser("~/.local/bin/hermes")

# ── 工具声明（OpenAI Realtime function 格式，session.update 用）──
JARVIS_TOOLS: list[dict] = [
    {
        "type": "function",
        "name": "get_weather",
        "description": (
            "查询指定城市的实时天气与今明预报。用户问天气、气温、下雨、"
            "穿什么、适不适合出门时调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "城市名，如 北京、上海、Tokyo"},
            },
            "required": ["location"],
        },
    },
    {
        "type": "function",
        "name": "get_stock_quote",
        "description": (
            "查询A股/港股/美股/指数的实时行情快照：现价、涨跌幅、成交量、"
            "换手率、PE。用户问某只股票现在多少钱、涨没涨时调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "codes_or_names": {
                    "type": "string",
                    "description": "股票代码或名称，逗号分隔，如 '平安银行,600519'",
                },
            },
            "required": ["codes_or_names"],
        },
    },
    {
        "type": "function",
        "name": "get_kline_history",
        "description": (
            "查本地K线库：某股近N日日线历史（开高低收+成交量）与区间统计。"
            "用户问走势、最近表现、多少天新高等。只覆盖A股个股。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code_or_name": {"type": "string", "description": "股票代码或名称，如 平安银行"},
                "days": {"type": "integer", "description": "最近多少个交易日，默认60"},
            },
            "required": ["code_or_name"],
        },
    },
    {
        "type": "function",
        "name": "run_zcode",
        "description": (
            "编码代理：把写代码、改bug、重构、建项目、文件读写、整理目录、"
            "跑shell命令、Git操作等一切开发/文件/命令类任务交给 ZCode（本机"
            "顶级AI编程代理，可直接读写指定目录文件、执行命令、多步完成）。"
            "task 必须自包含完整（含绝对路径、语言、验收标准），ZCode 拿到"
            "即可独立开工。耗时30秒到几分钟；超时会自动转后台继续办理，"
            "届时告知用户稍后用 check_zcode 查询，不许假装已完成。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "完整任务描述：做什么、在哪个目录、语言/框架、验收标准",
                },
                "workdir": {
                    "type": "string",
                    "description": "工作目录绝对路径，默认用户主目录 ~",
                },
            },
            "required": ["task"],
        },
    },
    {
        "type": "function",
        "name": "check_zcode",
        "description": (
            "查询 run_zcode 派出的后台任务进度：仍在跑 / 已完成（附结果）。"
            "用户回来问「那个活儿好了没」「办得怎么样了」时调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "run_zcode 返回的任务号，如 zc-3"},
            },
            "required": ["job_id"],
        },
    },
    {
        "type": "function",
        "name": "ask_hermes",
        "description": (
            "杂务后台：把非开发类的杂务交给 Hermes（本机全能AI助手，200+技能："
            "查资料、深度分析、写作翻译、订提醒、发消息等）。"
            "注意：编码/文件操作/跑命令/Git 一律走 run_zcode，天气走 get_weather，"
            "股票走 get_stock_quote。返回最终答复文本，耗时10-60秒。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "要 Hermes 干的活，自然语言描述"},
            },
            "required": ["task"],
        },
    },
]

# ── 网络小工具：绕开 clash 代理直连 ──────────────────
def _http_get(url: str, timeout: int = 8, referer: str | None = None) -> str | None:
    import urllib.request

    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        headers = {"User-Agent": "curl/8.0"}
        if referer:
            headers["Referer"] = referer
        req = urllib.request.Request(url, headers=headers)
        with opener.open(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception as e:
        logger.info("HTTP失败 %s: %s", url[:70], e)
        return None


def _http_get_gbk(url: str, timeout: int = 8, referer: str | None = None) -> str | None:
    import urllib.request

    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        headers = {"User-Agent": "curl/8.0"}
        if referer:
            headers["Referer"] = referer
        req = urllib.request.Request(url, headers=headers)
        with opener.open(req, timeout=timeout) as r:
            return r.read().decode("gbk", errors="replace")
    except Exception as e:
        logger.info("HTTP失败 %s: %s", url[:70], e)
        return None


# ── 天气 ─────────────────────────────────────────────
_WTTR_ZH = {
    "113": "晴", "116": "多云", "119": "阴", "122": "浓阴", "143": "薄雾",
    "176": "零星小雨", "200": "雷阵雨", "263": "零星阵雨", "266": "小雨",
    "293": "小雨", "296": "小雨", "299": "中雨", "302": "中雨", "305": "大雨",
    "308": "大雨", "311": "暴雨", "314": "大暴雨", "353": "阵雨", "356": "阵雨",
    "359": "大阵雨", "386": "零星阵雨", "389": "雷雨", "179": "零星雪",
    "182": "雨夹雪", "185": "雨夹雪", "227": "阵雪", "230": "大雪",
    "320": "雨雪", "323": "零星小雪", "326": "零星雨雪", "329": "小雨夹雪",
    "332": "中雪", "335": "中雪", "338": "大雪", "350": "冰雹", "374": "雨夹雪",
    "377": "雨夹雪", "392": "零星雷雨", "395": "雷雨伴雪",
}

_OM_ZH = {
    0: "晴", 1: "大致晴", 2: "多云", 3: "阴", 45: "雾", 48: "雾凇",
    51: "毛毛雨", 53: "毛毛雨", 55: "毛毛雨", 56: "冻毛毛雨", 57: "冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨", 66: "冻雨", 67: "冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "阵雨", 81: "中阵雨", 82: "强阵雨", 85: "阵雪", 86: "阵雪",
    95: "雷雨", 96: "雷雨伴冰雹", 99: "雷雨伴冰雹",
}


def _wttr_zh(h: dict) -> str:
    code = str(h.get("weatherCode", ""))
    if code in _WTTR_ZH:
        return _WTTR_ZH[code]
    desc = h.get("weatherDesc", [{}])
    return desc[0].get("value", "?") if desc else "?"


def _om_zh(code) -> str:
    if isinstance(code, int) and code in _OM_ZH:
        return _OM_ZH[code]
    return f"代码{code}"


async def get_weather(location: str) -> str:
    """wttr.in JSON（免 key）→ 解析失败转 Open-Meteo。"""
    import urllib.parse
    from urllib.parse import quote

    loop = asyncio.get_running_loop()
    loc_q = quote(location.strip())  # 中文城市名必须 URL 编码（ascii 拼接会炸）

    raw = await loop.run_in_executor(
        None, lambda: _http_get(f"https://wttr.in/{loc_q}?format=j1&lang=zh", timeout=8)
    )
    if raw:
        try:
            data = json.loads(raw)
            cur = data["current_condition"][0]
            today = data["weather"][0]
            tomorrow = data["weather"][1] if len(data["weather"]) > 1 else None

            def _day(d: dict) -> str:
                rain = max(int(h["chanceofrain"]) for h in d["hourly"])
                return (
                    f"{_wttr_zh(d['hourly'][4])} {d['mintempC']}~{d['maxtempC']}°C "
                    f"降水概率{rain}%"
                )

            lines = [
                f"地点:{location}",
                f"当前:{_wttr_zh(cur)} 气温{cur['temp_C']}°C(体感{cur['FeelsLikeC']}°C) "
                f"湿度{cur['humidity']}% 风{cur['windspeedKmph']}km/h",
                f"今日:{_day(today)}",
            ]
            if tomorrow:
                lines.append(f"明日:{_day(tomorrow)}")
            return "\n".join(lines)
        except Exception as e:
            logger.info("wttr解析失败: %s", e)

    # Open-Meteo 兜底
    import urllib.parse

    geo_raw = await loop.run_in_executor(
        None,
        lambda: _http_get(
            "https://geocoding-api.open-meteo.com/v1/search?name="
            + urllib.parse.quote(location) + "&count=1&language=zh"
        ),
    )
    if geo_raw:
        try:
            geo = json.loads(geo_raw)
            if geo.get("results"):
                r = geo["results"][0]
                w_raw = await loop.run_in_executor(
                    None,
                    lambda: _http_get(
                        f"https://api.open-meteo.com/v1/forecast?latitude={r['latitude']}"
                        f"&longitude={r['longitude']}"
                        "&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
                        "&daily=weather_code,temperature_2m_max,temperature_2m_min,"
                        "precipitation_probability_max&timezone=auto&forecast_days=2"
                    ),
                )
                if w_raw:
                    w = json.loads(w_raw)
                    c, d = w["current"], w["daily"]
                    return "\n".join([
                        f"地点:{location}",
                        f"当前:{_om_zh(c['weather_code'])} 气温{c['temperature_2m']:.0f}°C "
                        f"湿度{c['relative_humidity_2m']:.0f}% 风{c['wind_speed_10m']:.0f}km/h",
                        f"今日:{_om_zh(d['weather_code'][0])} {d['temperature_2m_min'][0]:.0f}~"
                        f"{d['temperature_2m_max'][0]:.0f}°C 降水概率{d['precipitation_probability_max'][0]}%",
                        f"明日:{_om_zh(d['weather_code'][1])} {d['temperature_2m_min'][1]:.0f}~"
                        f"{d['temperature_2m_max'][1]:.0f}°C 降水概率{d['precipitation_probability_max'][1]}%",
                    ])
        except Exception as e:
            logger.info("open-meteo兜底失败: %s", e)
    return "（气象站暂时失联，稍后再试）"


# ── 股票名→代码映射（本地K线缓存）────────────────────
_NAME_MAP: dict[str, str] | None = None


def _load_name_map() -> dict[str, str]:
    global _NAME_MAP
    if _NAME_MAP is not None:
        return _NAME_MAP
    if _NAME_MAP_PKL.exists():
        try:
            with open(_NAME_MAP_PKL, "rb") as f:
                _NAME_MAP = pickle.load(f)
            return _NAME_MAP
        except Exception as e:
            logger.info("股票名映射加载失败: %s", e)
    m: dict[str, str] = {}
    for f in glob.glob(os.path.join(_KLINE_DIR, "*.json")):
        if f.endswith(".bak"):
            continue
        try:
            with open(f) as fp:
                d = json.load(fp)
            m[str(d["code"])] = str(d["name"])
        except Exception:
            pass
    _NAME_MAP = m
    try:
        _NAME_MAP_PKL.parent.mkdir(parents=True, exist_ok=True)
        with open(_NAME_MAP_PKL, "wb") as fp:
            pickle.dump(m, fp)
    except Exception as e:
            logger.info("股票名映射保存失败: %s", e)
    return m


_INDEX_ALIAS = {
    "上证指数": "sh000001", "上证": "sh000001", "沪指": "sh000001", "大盘": "sh000001",
    "深成指": "sz399001", "创业板指": "sz399006", "创业板": "sz399006",
    "科创50": "sh000688", "北证50": "bj899050", "沪深300": "sh000300",
    "恒生指数": "hkHSI", "恒指": "hkHSI", "道琼斯": "usDJI", "纳斯达克": "usIXIC",
    "标普500": "usINX", "标普": "usINX",
}


def _match_code(token: str, name_map: dict[str, str]) -> str | None:
    """token → 腾讯符号（sh600519 等）。支持指数别名/纯代码/去前缀/名字模糊。"""
    t = token.strip()
    if not t:
        return None
    for alias, sym in _INDEX_ALIAS.items():
        if t in (alias, alias.replace("指数", "")):
            return sym
    if re.fullmatch(r"[A-Za-z]{2}\d{6}", t):
        return t.lower()
    if re.fullmatch(r"\d{6}", t):
        if t[0] in "03":
            return f"sz{t}"
        if t[0] == "6":
            return f"sh{t}"
        if t[0] in "84":
            return f"bj{t}"
        return f"sh{t}"
    exact = [c for c, n in name_map.items() if n == t]
    if exact:
        return _six_to_symbol(exact[0])
    contains = sorted(
        ((c, n) for c, n in name_map.items() if t in n), key=lambda x: len(x[1])
    )
    if contains:
        return _six_to_symbol(contains[0][0])
    return None


def _six_to_symbol(code: str) -> str:
    if code[0] in "03":
        return f"sz{code}"
    if code[0] == "6":
        return f"sh{code}"
    if code[0] in "84":
        return f"bj{code}"
    return f"sh{code}"


async def get_stock_quote(codes_or_names: str) -> str:
    """腾讯 qt.gtimg.cn 批量快照。一行一股。"""
    loop = asyncio.get_running_loop()
    name_map = _load_name_map()
    tokens = [t.strip() for t in re.split(r"[,，、\s]+", codes_or_names) if t.strip()]
    symbols = [s for s in (_match_code(t, name_map) for t in tokens) if s]
    if not symbols:
        return "（没找到这只股票，试试六位代码，如 600519）"

    raw = await loop.run_in_executor(
        None,
        lambda: _http_get_gbk(
            "https://qt.gtimg.cn/q=" + ",".join(symbols),
            referer="https://gu.qq.com",
        ),
    )
    if not raw:
        return "（行情线路繁忙，稍后再试）"
    out = []
    for seg in raw.split(";"):
        m = re.search(r'="(.+)"', seg)
        if not m:
            continue
        f = m.group(1).split("~")
        if len(f) < 45:
            continue
        # 1名称 2代码 3现价 31涨跌额 32涨跌幅 37成交额(万) 38换手 39PE
        out.append(
            f"{f[1]}({f[2]}): 现价{f[3]} 涨跌{f[31]}({f[32]}%) "
            f"成交额{f[37]}万 换手{f[38]}% PE{f[39]}"
        )
    return "\n".join(out) if out else "（行情数据解析失败）"


async def get_kline_history(code_or_name: str, days: int = 60) -> str:
    """本地K线缓存：区间统计 + 近10日明细。"""
    name_map = _load_name_map()
    code = None
    t = code_or_name.strip()
    m = re.fullmatch(r"[A-Za-z]{2}(\d{6})", t)
    if m:
        code = m.group(1)
    elif re.fullmatch(r"\d{6}", t):
        code = t
    else:
        exact = [c for c, n in name_map.items() if n == t]
        if not exact:
            contains = sorted(
                ((c, n) for c, n in name_map.items() if t in n),
                key=lambda x: len(x[1]),
            )
            exact = [contains[0][0]] if contains else []
        code = exact[0] if exact else None
    if code is None:
        return "（本地没这只票，可用 get_stock_quote 查实时）"
    f = os.path.join(_KLINE_DIR, f"{code}.json")
    if not os.path.exists(f):
        return "（本地无K线文件）"
    try:
        with open(f) as fp:
            d = json.load(fp)
        n = max(1, min(int(days), 500))
        kls = d["klines"][-n:]
        closes = [k["close"] for k in kls]
        chg = (closes[-1] / closes[0] - 1) * 100
        hi = max(k["high"] for k in kls)
        lo = min(k["low"] for k in kls)
        avg = sum(closes) / len(closes)
        head = "; ".join(
            f"{k['date'][5:]}收{k['close']}量{k['volume']}" for k in kls[-10:]
        )
        return (
            f"{d['name']}({code}) 近{len(kls)}日: 区间涨跌{chg:+.1f}% "
            f"最高{hi} 最低{lo} 均价{avg:.2f} 现价{closes[-1]}(截至{d['fetched_at'][:10]})\n"
            f"近10日: {head}"
        )
    except Exception as e:
        logger.info("K线读取失败 %s: %s", code, e)
        return "（K线数据读取失败）"


async def ask_hermes(task: str) -> str:
    """Hermes 无头模式。120s 超时杀进程回"后台办理中"话术（防僵尸空烧）。"""
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            _HERMES_BIN, "-z", task,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=os.path.expanduser("~"),
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
        text = out.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            logger.info("ask_hermes 非零退出 %s", proc.returncode)
            return "（Hermes 执行出错，已记录）"
        if not text:
            return "（Hermes 没有返回内容）"
        return text[-4000:]
    except asyncio.TimeoutError:
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        logger.info("ask_hermes 超时(120s)已杀: %s", task[:60])
        return "（此事较复杂，Hermes 仍在后台办理中。请如实告知用户：已派后台，稍后可再来问进度）"
    except Exception as e:
        logger.info("ask_hermes 失败: %s", e)
        return "（Hermes 线路故障）"


# ── ZCode 编码代理（无头 -p 模式）────────────────────
_NODE_BIN = os.path.expanduser("~/.nvm/versions/node/v22.16.0/bin/node")
_ZCODE_CJS = "/Applications/ZCode.app/Contents/Resources/glm/zcode.cjs"
_ZCODE_SYNC_TIMEOUT_S = 300          # 同步等待上限；超时转后台继续跑
_ZCODE_TAIL_CHARS = 3500             # 回注给 LLM 的结果尾部长度
_ZCODE_JOBS_DIR = Path(__file__).resolve().parents[2] / "logs" / "zcode_jobs"
_ZCODE_JOBS: dict[str, dict] = {}    # job_id -> {proc, task, log, started}
_ZCODE_SEQ = 0


def _zcode_available() -> bool:
    return os.path.isfile(_ZCODE_CJS) and os.path.isfile(_NODE_BIN)


def _log_tail(path: Path, chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[-chars:].strip()
    except Exception:
        return ""


async def run_zcode(task: str, workdir: str = "") -> str:
    """ZCode 无头执行。stdout 落盘（防 PIPE 缓冲区死锁），300s 内完成直接
    回结果；超时不杀进程、转后台登记，用 check_zcode 查。"""
    global _ZCODE_SEQ
    if not _zcode_available():
        return "（ZCode CLI 不在本机预期路径，无法执行开发任务）"
    task = (task or "").strip()
    if not task:
        return "（任务描述为空）"
    wd = os.path.expanduser(workdir.strip() or "~")
    if not os.path.isdir(wd):
        return f"（工作目录不存在: {wd}）"

    _ZCODE_SEQ += 1
    job_id = f"zc-{_ZCODE_SEQ}"
    _ZCODE_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _ZCODE_JOBS_DIR / f"{job_id}.log"

    prompt = f"{task}\n\n（完成后用一小段中文汇报：做了什么、结果如何、关键文件路径）"
    env = dict(os.environ, NO_COLOR="1")
    try:
        with open(log_path, "w", encoding="utf-8") as lf:
            proc = await asyncio.create_subprocess_exec(
                _NODE_BIN, _ZCODE_CJS, "-p", prompt, "--cwd", wd,
                stdout=lf, stderr=asyncio.subprocess.STDOUT,
                cwd=wd, env=env,
            )
    except Exception as e:
        logger.info("run_zcode 启动失败: %s", e)
        return "（ZCode 启动失败，已记录）"

    _ZCODE_JOBS[job_id] = {
        "proc": proc, "task": task, "log": log_path,
        "started": asyncio.get_running_loop().time(),
    }
    try:
        await asyncio.wait_for(proc.wait(), timeout=_ZCODE_SYNC_TIMEOUT_S)
    except asyncio.TimeoutError:
        asyncio.create_task(proc.wait())  # 孤儿收尸，防僵尸
        logger.info("run_zcode 转后台 %s (pid=%s): %s", job_id, proc.pid, task[:60])
        return (
            f"（任务 {job_id} 较重，ZCode 已转后台继续办理，进程仍在跑。"
            f"请如实告知用户：已派后台办理，稍后问「{job_id} 好了没」可查进度）"
        )

    out = _log_tail(log_path, _ZCODE_TAIL_CHARS)
    logger.info("run_zcode %s 完成 rc=%s 耗时~%ds: %s",
                job_id, proc.returncode,
                int(asyncio.get_running_loop().time() - _ZCODE_JOBS[job_id]["started"]),
                task[:60])
    if proc.returncode != 0:
        return f"（ZCode 退出码 {proc.returncode}，日志尾部：\n{out or '(无输出)'}）"
    return out or "（ZCode 正常结束但没有输出）"


async def check_zcode(job_id: str) -> str:
    job = _ZCODE_JOBS.get((job_id or "").strip())
    if job is None:
        # 编排器重启后内存表丢失，磁盘日志还在——按号读档兜底
        log_path = _ZCODE_JOBS_DIR / f"{(job_id or '').strip()}.log"
        if (job_id or "").strip().startswith("zc-") and log_path.is_file():
            return f"（任务 {job_id.strip()} 的日志存档：\n{_log_tail(log_path, _ZCODE_TAIL_CHARS)}）"
        return "（没有这个任务号；注意编排器重启后只能查日志存档）"
    proc = job["proc"]
    elapsed = int(asyncio.get_running_loop().time() - job["started"])
    if proc.returncode is None:
        return (
            f"（任务 {job_id} 仍在后台运行（已 {elapsed} 秒），任务：{job['task'][:80]}。"
            f"请告知用户还在办，稍后再查）"
        )
    out = _log_tail(job["log"], _ZCODE_TAIL_CHARS)
    return f"（任务 {job_id} 已完成（共 {elapsed} 秒，退出码 {proc.returncode}）。结果：\n{out}）"


async def execute_tool(name: str, arguments: dict) -> str:
    """统一入口：按名分发。异常兜底文案，绝不抛出。"""
    try:
        if name == "get_weather":
            return await get_weather(str(arguments.get("location", "")).strip() or "北京")
        if name == "get_stock_quote":
            return await get_stock_quote(arguments.get("codes_or_names", ""))
        if name == "get_kline_history":
            return await get_kline_history(
                arguments.get("code_or_name", ""), int(arguments.get("days", 60) or 60)
            )
        if name == "ask_hermes":
            return await ask_hermes(arguments.get("task", ""))
        if name == "run_zcode":
            return await run_zcode(
                arguments.get("task", ""), arguments.get("workdir", "")
            )
        if name == "check_zcode":
            return await check_zcode(arguments.get("job_id", ""))
        return f"（未知工具 {name}）"
    except Exception as e:
        logger.info("工具 %s 执行异常: %s", name, e)
        return f"（工具执行异常: {type(e).__name__}）"
