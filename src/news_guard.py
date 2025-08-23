# src/news_guard.py
"""
NewsGuard: pure-Python calendar gate (no server, no API keys).

- Pulls public ForexFactory JSON (this week / next week mirrors).
- Robust timestamp parsing (sec/ms/ISO), currency inference.
- Multi-currency support, rule-based impact, configurable quiet windows.
- Module-level caching to avoid repeated network calls in your loop.

Env (optional):
  FF_URLS               comma-separated feed URLs
  FF_CACHE_TTL_SEC      cache TTL seconds (default 1800)
  AFFECTS_JSON          JSON dict symbol -> [ccys] to override defaults
  WINDOWS_JSON          JSON dict impact -> [pre_min, post_min] to override windows
"""

import os, json, re, time, logging, requests, tempfile
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
LOG = logging.getLogger("news_guard")
if not LOG.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"))

# ---------------- Config ----------------
FF_DEFAULT_URLS = [
    "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.json",
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "https://cdn-nfs.faireconomy.media/ff_calendar_nextweek.json",
    "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
]
FF_URLS: List[str] = [u.strip() for u in os.getenv("FF_URLS", ",".join(FF_DEFAULT_URLS)).split(",") if u.strip()]
CACHE_FILE = os.path.join(tempfile.gettempdir(), "ff_calendar.json")
CACHE_TTL  = int(os.getenv("FF_CACHE_TTL_SEC", "1800"))  # 30 min

# Default symbol→currency map (extend as needed)
AFFECTS: Dict[str, List[str]] = {
    "EURUSD": ["EUR","USD"], "GBPUSD": ["GBP","USD"], "USDJPY": ["USD","JPY"],
    "USDCHF": ["USD","CHF"], "AUDUSD": ["AUD","USD"], "NZDUSD": ["NZD","USD"],
    "USDCAD": ["USD","CAD"], "XAUUSD": ["USD"], "XAGUSD": ["USD"],
    "WTICOUSD": ["USD"], "UKOIL": ["GBP","USD"],
    "NAS100": ["USD"], "US100": ["USD"], "US500": ["USD"], "SPX500": ["USD"], "US30": ["USD"],
    "DE40": ["EUR"], "EU50": ["EUR"], "FR40": ["EUR"], "UK100": ["GBP"], "JP225": ["JPY"],
}
if os.getenv("AFFECTS_JSON"):
    try:
        AFFECTS.update(json.loads(os.getenv("AFFECTS_JSON","{}")))
        LOG.info("AFFECTS overridden via AFFECTS_JSON")
    except Exception as ex:
        LOG.warning("AFFECTS_JSON parse failed: %s", ex)

# Quiet windows (minutes before/after)
WINDOWS: Dict[str, Tuple[int,int]] = {"extreme":(-90,45), "high":(-60,30), "medium":(-30,15)}
if os.getenv("WINDOWS_JSON"):
    try:
        raw = json.loads(os.getenv("WINDOWS_JSON","{}"))
        for k, v in raw.items():
            if isinstance(v, list) and len(v)==2:
                WINDOWS[k] = (int(v[0]), int(v[1]))
        LOG.info("WINDOWS overridden via WINDOWS_JSON: %s", WINDOWS)
    except Exception as ex:
        LOG.warning("WINDOWS_JSON parse failed: %s", ex)

# Impact rules
KW_EXTREME = re.compile(r"(NFP|CPI(?!\s*preview)|FOMC|Rate Decision|Interest Rate|ECB|BoE|BoJ|SNB|Core PCE)", re.I)
KW_HIGH    = re.compile(r"(GDP|CPI|Payrolls|ISM|PMI)", re.I)

CCY_FROM_COUNTRY = {
    "United States":"USD","U.S.":"USD","USA":"USD",
    "Euro Area":"EUR","Germany":"EUR","France":"EUR","Italy":"EUR","Spain":"EUR","Netherlands":"EUR",
    "United Kingdom":"GBP","Japan":"JPY","Switzerland":"CHF",
    "Canada":"CAD","Australia":"AUD","New Zealand":"NZD","China":"CNY",
}

# --------------- Cache ---------------
_events_cache: Optional[List[Dict[str,Any]]] = None
_cache_time: float = 0.0

def _load_cache() -> Optional[List[Dict[str,Any]]]:
    global _events_cache, _cache_time
    if _events_cache is not None and (time.time() - _cache_time) < CACHE_TTL:
        return _events_cache
    if os.path.exists(CACHE_FILE) and (time.time() - os.path.getmtime(CACHE_FILE) < CACHE_TTL):
        try:
            with open(CACHE_FILE,"r",encoding="utf-8") as f:
                _events_cache = json.load(f)
                _cache_time = time.time()
                return _events_cache
        except Exception:
            pass
    return None

def _save_cache(events: List[Dict[str,Any]]) -> None:
    global _events_cache, _cache_time
    _events_cache = events
    _cache_time = time.time()
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE,"w",encoding="utf-8") as f:
            json.dump(events, f)
    except Exception as ex:
        LOG.debug("Cache save failed: %s", ex)

# --------------- Fetch & normalize ---------------
def _fetch_ff() -> List[Dict[str,Any]]:
    events: List[Dict[str,Any]] = []
    headers = {"User-Agent":"Mozilla/5.0"}
    for url in FF_URLS:
        try:
            r = requests.get(url, timeout=20, headers=headers)
            if r.status_code != 200:
                continue
            payload = r.json()
        except Exception:
            continue
        if isinstance(payload, list):
            events.extend(payload); continue
        if isinstance(payload, dict):
            for key in ("events","calendar","data","items"):
                if key in payload and isinstance(payload[key], list):
                    events.extend(payload[key]); break
    return events

def _parse_ts(ts) -> Optional[datetime]:
    try:
        if isinstance(ts,(int,float)):
            t = int(ts);  t = (t//1000) if t>10_000_000_000 else t
            return datetime.fromtimestamp(t, tz=UTC)
        if isinstance(ts,str):
            if ts.isdigit():
                t = int(ts); t = (t//1000) if t>10_000_000_000 else t
                return datetime.fromtimestamp(t, tz=UTC)
            return datetime.fromisoformat(ts.replace("Z","")).replace(tzinfo=UTC)
    except Exception:
        return None
    return None

def _guess_ccy(ccy: str, e: Dict[str,Any]) -> str:
    if ccy: return ccy
    ctry = e.get("country") or e.get("Country") or e.get("region") or ""
    if ctry in CCY_FROM_COUNTRY: return CCY_FROM_COUNTRY[ctry]
    t = (e.get("title") or e.get("event") or "").lower()
    if any(k in t for k in ["fed","fomc","pce","payroll","nfp"," cpi","us "]): return "USD"
    if "ecb" in t or "euro" in t: return "EUR"
    if "boe" in t or " uk " in t:  return "GBP"
    if "boj" in t or "japan" in t: return "JPY"
    if "snb" in t or "swiss" in t: return "CHF"
    return ""

def _ensure_list_ccy(curr: str) -> List[str]:
    curr = (curr or "").replace(";",",").replace("/",",")
    return [c.strip().upper() for c in curr.split(",") if c.strip()]

def _classify(title: str, impact_src: str|None) -> str:
    if KW_EXTREME.search(title): return "extreme"
    if KW_HIGH.search(title):    return "high"
    if impact_src:
        s = impact_src.lower()
        if "high" in s: return "high"
        if "medium" in s: return "medium"
        if "low" in s: return "low"
    return "low"

def _normalize(raw: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    out: List[Dict[str,Any]] = []
    for e in raw:
        title   = (e.get("title") or e.get("event") or e.get("Event") or "").strip()
        impact  = e.get("impact") or e.get("Impact") or e.get("importance") or ""
        ccy     = _guess_ccy(e.get("currency") or e.get("Currency") or "", e)
        ts      = e.get("timestamp") or e.get("date_unix") or e.get("unix") or e.get("date") or e.get("Date")
        when    = _parse_ts(ts)
        if not when:
            # try separate date/time strings
            ds = e.get("dateStr") or e.get("dateText")
            ts = e.get("timeStr") or e.get("timeText")
            if not (ds and ts): 
                continue
            try:
                when = datetime.strptime(f"{ds} {ts}", "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
            except Exception:
                continue

        curr_list = _ensure_list_ccy(ccy) or ([ccy] if ccy else [])
        out.append({
            "time_utc": when.isoformat(),
            "currency": curr_list,     # list
            "title": title,
            "impact_src": str(impact),
            "forecast": e.get("forecast") or e.get("Forecast"),
            "previous": e.get("previous") or e.get("Previous"),
            "actual":   e.get("actual")   or e.get("Actual"),
        })
    out.sort(key=lambda x: x["time_utc"])
    return out

def _load_events() -> List[Dict[str,Any]]:
    cached = _load_cache()
    if cached is not None:
        return cached
    raw = _fetch_ff()
    norm = _normalize(raw)
    _save_cache(norm)
    return norm

# --------------- Public API ---------------
def news_gate(
    symbols: List[str],
    lookahead_min: int = 120,
    impacts_that_block: Optional[List[str]] = None,
    now: Optional[datetime] = None,
) -> Dict[str,Any]:
    """
    Returns:
      {
        "allowed_to_trade": bool (true if not currently inside a blocking window),
        "now_utc": ISO string,
        "results": [
           {"symbol":"EURUSD","allowed_to_trade":bool,"blockers":[...]}
        ],
        "blockers": [...combined...]
      }
    """
    now = now or datetime.now(UTC)
    events = _load_events()
    impacts_that_block = [i.lower() for i in (impacts_that_block or ["extreme","high"])]
    end = now + timedelta(minutes=lookahead_min)

    results = []
    combined: List[Dict[str,Any]] = []
    all_allowed = True

    for symbol in symbols:
        ccy_list = set(AFFECTS.get(symbol.upper(), []))
        blockers: List[Dict[str,Any]] = []
        sym_allowed_now = True

        for ev in events:
            ev_ccys = set(ev["currency"]) if isinstance(ev["currency"], list) else {ev["currency"]}
            if ccy_list and not (ev_ccys & ccy_list):
                continue

            when = datetime.fromisoformat(ev["time_utc"])
            if when < now - timedelta(hours=1):
                continue

            impact = _classify(ev["title"], ev["impact_src"])
            pre, post = WINDOWS.get(impact, (0,0))
            in_window_now   = (when + timedelta(minutes=pre)) <= now <= (when + timedelta(minutes=post))
            incoming_window = now <= when <= end

            if in_window_now or incoming_window:
                b = {
                    "time_utc": ev["time_utc"],
                    "title": ev["title"],
                    "currency": list(ev_ccys) if ev_ccys else [],
                    "impact": impact,
                    "forecast": ev.get("forecast"),
                    "previous": ev.get("previous"),
                    "actual": ev.get("actual"),
                    "window": [pre, post],
                }
                blockers.append(b)
                combined.append(b)

            # Hard block only if *currently* inside a window AND impact is in blocking list
            if in_window_now and (impact in impacts_that_block):
                sym_allowed_now = False

        results.append({"symbol": symbol.upper(), "allowed_to_trade": sym_allowed_now, "blockers": blockers[:10]})
        all_allowed = all_allowed and sym_allowed_now

    return {"allowed_to_trade": all_allowed, "now_utc": now.isoformat(), "results": results, "blockers": combined}
