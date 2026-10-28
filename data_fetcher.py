# -*- coding: utf-8 -*-
"""
data_fetcher.py — 数据抓取门面层（兼容接口）

本模块保留原有函数签名（供 app.py 直接调用），内部委托给 data_source.SourceManager，
由工厂按路由表选择 akshare / tushare 并在失败时自动降级。
Streamlit 的 @cache_data 缓存放在此层，保证切换数据源后可通过"强制刷新"清缓存。
"""
from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd

from data_source.factory import get_default_manager, reload_manager

# Streamlit 环境下自动缓存，命令行环境下退化为不缓存
try:  # pragma: no cover
    from streamlit import cache_data
    import streamlit as st
except Exception:  # pragma: no cover
    st = None

    def cache_data(ttl=None, **_kwargs):
        def deco(fn):
            return fn
        return deco

# 全局数据源管理器（懒加载单例）
_ds = get_default_manager()

# 进程级全局缓存（跨用户共享，带 TTL，带懒清理）
# 结构：{cache_key: (timestamp, DataFrame)}
_GLOBAL_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
_GLOBAL_TTL = 12 * 3600  # 12 小时


def _get_global(key: str, fetch_fn, *args,
                _progress_callback=None, **kwargs) -> pd.DataFrame:
    """带进度回调的全局缓存读取器。

    - 命中且未过期 → 直接返回，不触发回调（不显示进度条）
    - 未命中 → 调用 fetch_fn 并传入回调，结果写入全局缓存
    - 每次访问时懒清理过期 entry，避免堆积
    """
    now = time.time()
    # 懒清理：删除所有过期的 key
    for k in list(_GLOBAL_CACHE.keys()):
        if now - _GLOBAL_CACHE[k][0] >= _GLOBAL_TTL:
            del _GLOBAL_CACHE[k]
    entry = _GLOBAL_CACHE.get(key)
    if entry and now - entry[0] < _GLOBAL_TTL:
        return entry[1]
    result = fetch_fn(*args, _progress_callback=_progress_callback, **kwargs)
    _GLOBAL_CACHE[key] = (now, result)
    return result


def clear_global_cache(prefix: str | None = None) -> None:
    """清除全局缓存。prefix 为 None 时清全部，否则只清指定前缀的 key。"""
    if prefix is None:
        _GLOBAL_CACHE.clear()
    else:
        for k in list(_GLOBAL_CACHE.keys()):
            if k.startswith(prefix):
                del _GLOBAL_CACHE[k]


def is_cached(key: str, lookback_calendar_days: int | None = None) -> bool:
    """检查全局缓存是否已覆盖目标回溯区间（供 app 层判断是否需要显示进度条）。

    仅判断 key 是否存在是不够的：增量缓存会一直存在，但每次调整回溯天数
    仍需补拉缺失区间，此时必须显示进度条。因此传入 lookback 时还要校验：
      - 缓存最早日期 <= 目标起始日期（往前已覆盖）
      - 缓存最晚日期 >= 今天-3天（往后基本跟到最新，容忍周末/短假）
    """
    entry = _GLOBAL_CACHE.get(key)
    if entry is None or time.time() - entry[0] >= _GLOBAL_TTL:
        return False
    if lookback_calendar_days is None:
        return True

    df = entry[1]
    if df is None or df.empty or "date" not in df.columns:
        return False
    dates = pd.to_datetime(df["date"], errors="coerce").dt.normalize().dropna()
    if dates.empty:
        return False

    target_start = date.today() - timedelta(days=lookback_calendar_days)
    fresh_before = date.today() - timedelta(days=3)
    return dates.min().date() <= target_start and dates.max().date() >= fresh_before


def _get_incremental(cache_key: str, fetch_fn, lookback_calendar_days: int,
                     _progress_callback=None, **fetch_kwargs) -> pd.DataFrame:
    """增量拉取：缓存中保存已拉的全量数据，只补拉缺失的日期段。

    - cache_key: 存储累计数据的 key（不随 lookback 变化）
    - fetch_fn: 数据源方法，接受 (lookback_calendar_days=, start_date=, _progress_callback=)
    - lookback_calendar_days: 用户请求的回溯自然日数
    - **fetch_kwargs: 透传给 fetch_fn 的额外关键字参数（如 vars_list）；
      注意必须用关键字传递——这些方法第一个位置参数就是 lookback_calendar_days

    策略：
      1. 目标区间 = [今天 - lookback, 昨天]
      2. 已有缓存区间 = [cached_min, cached_max]
      3. 若目标起点早于 cached_min → 往前补到目标起点
      4. 若昨天晚于 cached_max → 往后补（新交易日）
      5. 合并去重后返回目标区间内的数据
    """
    now = time.time()
    for k in list(_GLOBAL_CACHE.keys()):
        if now - _GLOBAL_CACHE[k][0] >= _GLOBAL_TTL:
            del _GLOBAL_CACHE[k]

    target_start = date.today() - timedelta(days=lookback_calendar_days)
    yesterday = date.today() - timedelta(days=1)

    entry = _GLOBAL_CACHE.get(cache_key)
    cached = entry[1] if (entry and now - entry[0] < _GLOBAL_TTL) else pd.DataFrame()

    # 计算需要拉取的起点
    if cached.empty or "date" not in cached.columns:
        fetch_start = target_start
    else:
        cached_dates = pd.to_datetime(cached["date"], errors="coerce").dt.normalize().dropna()
        if cached_dates.empty:
            fetch_start = target_start
        else:
            cached_min = cached_dates.min().date()
            cached_max = cached_dates.max().date()
            # 需要补的最早日期
            fetch_start = min(target_start, cached_max + timedelta(days=1))
            # 如果 fetch_start > yesterday，说明无需拉取
            if fetch_start > yesterday:
                result = cached[cached_dates >= pd.Timestamp(target_start)]
                return result.sort_values("date").reset_index(drop=True)

    new_data = fetch_fn(lookback_calendar_days=0, start_date=fetch_start,
                        _progress_callback=_progress_callback, **fetch_kwargs)

    if new_data is not None and not new_data.empty:
        if cached.empty:
            merged = new_data
        else:
            merged = pd.concat([cached, new_data], ignore_index=True)
            # 按所有列去重，避免重复日期的数据
            merged = merged.drop_duplicates()
        _GLOBAL_CACHE[cache_key] = (now, merged)
        cached = merged

    if cached.empty or "date" not in cached.columns:
        return pd.DataFrame()
    cached_dates = pd.to_datetime(cached["date"], errors="coerce").dt.normalize()
    result = cached[cached_dates >= pd.Timestamp(target_start)]
    return result.sort_values("date").reset_index(drop=True)


def set_data_source(source: str) -> None:
    """运行时切换默认数据源（重建管理器）。切换后建议清缓存。"""
    global _ds
    _ds = reload_manager(default_source=source)


@cache_data(ttl=12 * 3600)
def get_bond_10y(start_date: str = "20100101") -> pd.DataFrame:
    """10 年期国债收益率，返回列: date, bond_yield"""
    return _ds.get_bond_10y(start_date)


@cache_data(ttl=24 * 3600)
def get_index_pe_monthly(symbol: str = "沪深300") -> pd.DataFrame:
    """指数市盈率（月频），返回列: date, pe_ttm"""
    return _ds.get_index_pe_monthly(symbol)


@cache_data(ttl=12 * 3600)
def get_index_pe_daily(symbol: str = "沪深300") -> pd.DataFrame:
    """指数市盈率（日频，全历史），返回列: date, pe_ttm"""
    return _ds.get_index_pe_daily(symbol)


@cache_data(ttl=12 * 3600)
def get_index_pe_daily_csindex(csindex_code: str = "000300") -> pd.DataFrame:
    """指数市盈率（日频），返回列: date, pe_static, pe_ttm, div_yield"""
    return _ds.get_index_pe_daily_csindex(csindex_code)


@cache_data(ttl=12 * 3600)
def get_index_daily(sina_code: str = "sh000300") -> pd.DataFrame:
    """大盘指数日线，返回列: date, open, high, low, close, volume"""
    return _ds.get_index_daily(sina_code)


# ---------------------------------------------------------------------------
# 指数分钟数据本地缓存 + K 线构建
# ---------------------------------------------------------------------------
from pathlib import Path as _Path

_MINUTE_CACHE_DIR = _Path(__file__).resolve().parent / "data" / "minute_cache"
_MINUTE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_index_minute_cached(sina_code: str = "sh000016") -> pd.DataFrame:
    """获取指数分钟数据并本地持久化缓存。

    akshare 新浪接口仅返回最近约 5 个交易日的 1 分钟数据，
    每次调用时：读取本地已有缓存 → 拉取最新增量 → 合并去重 → 写回缓存。
    返回列: day(datetime), open, high, low, close, volume, amount
    """
    import akshare as ak

    cache_file = _MINUTE_CACHE_DIR / f"{sina_code}.parquet"
    # 1. 读取已有缓存
    old = pd.DataFrame()
    if cache_file.exists():
        try:
            old = pd.read_parquet(cache_file)
        except Exception:
            old = pd.DataFrame()
    # 2. 拉取最新分钟数据（新浪仅返回最近几天）
    try:
        new = ak.stock_zh_a_minute(symbol=sina_code, period="1")
    except Exception:
        new = pd.DataFrame()
    # 3. 合并去重
    if not new.empty:
        new["day"] = pd.to_datetime(new["day"])
        for c in ["open", "high", "low", "close", "volume", "amount"]:
            if c in new.columns:
                new[c] = pd.to_numeric(new[c], errors="coerce")
        combined = pd.concat([old, new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["day"], keep="last")
        combined = combined.sort_values("day").reset_index(drop=True)
    else:
        combined = old
    # 4. 写回缓存
    if not combined.empty:
        try:
            combined.to_parquet(cache_file, index=False)
        except Exception:
            pass  # 缓存写入失败不影响返回
    return combined


def get_index_kline(sina_code: str = "sh000016",
                    lookback_days: int = 120) -> pd.DataFrame:
    """构建指数日 K 线：分钟数据聚合 OHLC + tushare 日线成交量。

    规则：
    - 有分钟数据的日期：OHLC 由分钟数据聚合，成交量用 tushare 日线
    - 无分钟数据的日期：open = 昨日 close（来自 tushare 日线），
      high/low/close 用 tushare 日线，成交量用 tushare 日线
    返回列: date, open, high, low, close, volume
    """
    # tushare 日线（含 OHLCV，全历史）
    daily = get_index_daily(sina_code)
    if daily.empty:
        return pd.DataFrame()
    daily = daily.sort_values("date").reset_index(drop=True)
    # 截取 lookback
    daily = daily.tail(lookback_days).reset_index(drop=True)

    # 分钟数据聚合为日 OHLC
    minute = get_index_minute_cached(sina_code)
    minute_daily = pd.DataFrame()
    if not minute.empty:
        minute = minute.copy()
        minute["date"] = minute["day"].dt.normalize()
        minute_daily = minute.groupby("date", as_index=False).agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        )

    # 合并：优先用分钟 OHLC，缺失用 tushare 日线补齐
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    if not minute_daily.empty:
        merged = daily.merge(minute_daily, on="date", how="left",
                             suffixes=("", "_min"))
        # 有分钟数据的用分钟 OHLC
        has_min = merged["open_min"].notna()
        merged.loc[has_min, "open"] = merged.loc[has_min, "open_min"]
        merged.loc[has_min, "high"] = merged.loc[has_min, "high_min"]
        merged.loc[has_min, "low"] = merged.loc[has_min, "low_min"]
        merged.loc[has_min, "close"] = merged.loc[has_min, "close_min"]
        merged = merged.drop(columns=[c for c in merged.columns
                                      if c.endswith("_min")])
    else:
        merged = daily

    # 无分钟数据的日期：open = 昨日 close
    merged["prev_close"] = merged["close"].shift(1)
    no_min_mask = merged["open"].isna()
    merged.loc[no_min_mask, "open"] = merged.loc[no_min_mask, "prev_close"]
    # 若 open 仍为 NaN（首日），用 close 填充
    merged["open"] = merged["open"].fillna(merged["close"])
    merged = merged.drop(columns=["prev_close"])

    # 确保 high/low 合理
    merged["high"] = merged[["high", "open", "close"]].max(axis=1)
    merged["low"] = merged[["low", "open", "close"]].min(axis=1)

    return merged[["date", "open", "high", "low", "close", "volume"]].dropna()



@cache_data(ttl=12 * 3600)
def get_shibor() -> pd.DataFrame:
    """SHIBOR 期限结构，返回列: date + 8 个期限"""
    return _ds.get_shibor()


@cache_data(ttl=6 * 3600)
def get_shibor_irs_curve(lookback_days: int = 30) -> pd.DataFrame:
    """Shibor3M 利率互换收盘曲线，返回长格式: date, tenor, bid, avg, ask（单位 %）。

    数据来自中国货币网 chinamoney.com.cn，仅提供近一年数据。
    """
    return _ds.get_shibor_irs_curve(lookback_days=lookback_days)


def get_option_pcr_history(lookback_calendar_days: int = 45,
                           _progress_callback=None) -> pd.DataFrame:
    """期权 PCR 历史，返回列: date, underlying, call_volume, put_volume, ...

    _progress_callback(current, total) 用于 UI 显示加载进度；
    采用增量拉取：缓存已拉数据，仅补拉缺失日期（不随 lookback 全量重拉）。
    """
    return _get_incremental("pcr_accumulated", _ds.get_option_pcr_history,
                            lookback_calendar_days, _progress_callback=_progress_callback)


def get_option_broker_holding_history(lookback_calendar_days: int = 45,
                                      _progress_callback=None) -> pd.DataFrame:
    """期权券商持仓排名历史（上交所前 3 标的 × 认购/认沽前 5 会员）。

    返回列: date, underlying, broker, call_oi, put_oi, pcr_oi
    采用增量拉取：缓存已拉数据，仅补拉缺失日期。
    """
    return _get_incremental("opt_broker_accumulated", _ds.get_option_broker_holding_history,
                            lookback_calendar_days, _progress_callback=_progress_callback)


@cache_data(ttl=12 * 3600)
def get_index_futures_basis_history(lookback_calendar_days: int = 45) -> pd.DataFrame:
    """股指期货升贴水历史，返回列: date, variety, ..., basis, basis_pct"""
    return _ds.get_index_futures_basis_history(lookback_calendar_days)


def get_citic_position_history(lookback_calendar_days: int = 45,
                               vars_list: list[str] | None = None,
                               _progress_callback=None) -> pd.DataFrame:
    """中信期货多空持仓历史，返回列: date, variety, contract, long_oi, ...

    _progress_callback(current, total) 用于 UI 显示加载进度；
    采用增量拉取：缓存已拉数据，仅补拉缺失日期。
    """
    return _get_incremental("citic_accumulated", _ds.get_citic_position_history,
                            lookback_calendar_days, vars_list=vars_list,
                            _progress_callback=_progress_callback)


@cache_data(ttl=12 * 3600)
def get_gold_curve_latest() -> pd.DataFrame:
    """黄金期货期限结构，返回列: date, contract, expire_date, close, ..."""
    return _ds.get_gold_curve_latest()


@cache_data(ttl=12 * 3600)
def get_gold_main_daily(start_date: str = "20180101",
                        end_date: str | None = None) -> pd.DataFrame:
    """黄金主力连续日线，返回列: date, close, volume, open_interest"""
    return _ds.get_gold_main_daily(start_date, end_date)


_COT_CACHE_DIR = _Path(__file__).resolve().parent / "data" / "cot_cache"
_COT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@cache_data(ttl=24 * 3600)
def get_cot_gold(years: int = 3) -> pd.DataFrame:
    """CFTC COT 黄金持仓，返回列: date, open_interest, mm_long, ..., mm_net

    cftc.gov 国内网络经常超时，因此做本地磁盘缓存兜底：
    拉取成功时写入缓存；拉取失败时回退到上次成功的缓存，避免模块整体报错。
    """
    cache_file = _COT_CACHE_DIR / "gold.parquet"
    try:
        df = _ds.get_cot_gold(years)
    except Exception:
        if cache_file.exists():
            return pd.read_parquet(cache_file)
        raise
    if df is not None and not df.empty:
        try:
            df.to_parquet(cache_file, index=False)
        except Exception:
            pass
        return df
    if cache_file.exists():
        return pd.read_parquet(cache_file)
    return df


# ---------------------------------------------------------------------------
# 离岸在岸息差系统
# ---------------------------------------------------------------------------
@cache_data(ttl=12 * 3600)
def get_effr(start_date: str = "20200101") -> pd.DataFrame:
    """有效联邦基金利率，返回列: date, effr（单位 %）"""
    return _ds.get_effr(start_date)


@cache_data(ttl=12 * 3600)
def get_sofr_futures(start_date: str = "20200101") -> pd.DataFrame:
    """3个月SOFR期货收盘价，返回列: date, sr3_close（隐含离岸利率 = 100 - sr3_close）"""
    return _ds.get_sofr_futures(start_date)


@cache_data(ttl=12 * 3600)
def get_cnh_hibor() -> pd.DataFrame:
    """离岸人民币3个月Hibor，返回列: date, cnh_hibor_3m（单位 %）"""
    return _ds.get_cnh_hibor()


@cache_data(ttl=12 * 3600)
def get_usd_cny() -> pd.DataFrame:
    """在岸人民币汇率，返回列: date, usd_cny"""
    return _ds.get_usd_cny()


@cache_data(ttl=12 * 3600)
def get_usd_cnh() -> pd.DataFrame:
    """离岸人民币汇率，返回列: date, usd_cnh"""
    return _ds.get_usd_cnh()


@cache_data(ttl=12 * 3600)
def get_bond_yield_curve(start_date: str = "20100101") -> pd.DataFrame:
    """中国国债收益率曲线(2Y+10Y)，返回列: date, bond_2y, bond_10y（单位 %）"""
    return _ds.get_bond_yield_curve(start_date)


@cache_data(ttl=12 * 3600)
def get_us_macro_events(start_date: str = "20200101") -> pd.DataFrame:
    """美国宏观事件日历（FOMC 利率决议 / CPI / PPI 公布日）。
    返回列: event_date, event_type, event_name, value, prev_value, change,
            surprise_hint, detail
    """
    return _ds.get_us_macro_events(start_date)


@cache_data(ttl=12 * 3600)
def get_us_treasury_yields(start_date: str = "20200101") -> pd.DataFrame:
    """美国国债各期限收益率，返回长表: date, tenor, yield_pct（单位 %）。"""
    return _ds.get_us_treasury_yields(start_date)


def available_sources() -> list[str]:
    """返回当前可用的数据源列表（供 UI 展示）"""
    return _ds.available_sources


def get_source_info(method_name: str) -> tuple[str, str]:
    """返回某方法实际使用的数据源信息：(一级源, 二级渠道说明)。

    一级源为实际命中的库（akshare / tushare），二级为原始发布渠道 + 频率。
    若该方法尚未被调用过，则按路由表主源返回兜底信息。
    """
    src = _ds.get_last_source(method_name)
    if src:
        channels = _load_source_channels(src)
        if channels and method_name in channels:
            return src, channels[method_name]
    # 兜底：根据路由表取主源
    from data_source.factory import METHOD_ROUTING
    chain = METHOD_ROUTING.get(method_name, [])
    primary = chain[0] if chain else "unknown"
    channels = _load_source_channels(primary)
    if channels and method_name in channels:
        return primary, channels[method_name]
    return primary, "（渠道说明未配置）"


def _load_source_channels(src_name: str) -> dict[str, str] | None:
    """懒加载对应数据源的 SOURCE_CHANNELS 字典。"""
    try:
        if src_name == "akshare":
            from data_source.akshare_source import SOURCE_CHANNELS
        elif src_name == "tushare":
            from data_source.tushare_source import SOURCE_CHANNELS
        else:
            return None
        return SOURCE_CHANNELS
    except Exception:
        return None
