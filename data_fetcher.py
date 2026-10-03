# -*- coding: utf-8 -*-
"""
data_fetcher.py — 数据抓取门面层（兼容接口）

本模块保留原有函数签名（供 app.py 直接调用），内部委托给 data_source.SourceManager，
由工厂按路由表选择 akshare / tushare 并在失败时自动降级。
Streamlit 的 @cache_data 缓存放在此层，保证切换数据源后可通过"强制刷新"清缓存。
"""
from __future__ import annotations

import time

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


def is_cached(key: str) -> bool:
    """检查全局缓存中是否存在未过期的 key（供 app 层预知是否需要显示进度条）。"""
    entry = _GLOBAL_CACHE.get(key)
    if entry is None:
        return False
    return time.time() - entry[0] < _GLOBAL_TTL


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
def get_index_pe_daily_csindex(csindex_code: str = "000300") -> pd.DataFrame:
    """指数市盈率（日频），返回列: date, pe_static, pe_ttm, div_yield"""
    return _ds.get_index_pe_daily_csindex(csindex_code)


@cache_data(ttl=12 * 3600)
def get_index_daily(sina_code: str = "sh000300") -> pd.DataFrame:
    """大盘指数日线，返回列: date, open, high, low, close, volume"""
    return _ds.get_index_daily(sina_code)


@cache_data(ttl=12 * 3600)
def get_shibor() -> pd.DataFrame:
    """SHIBOR 期限结构，返回列: date + 8 个期限"""
    return _ds.get_shibor()


def get_option_pcr_history(lookback_calendar_days: int = 45,
                           _progress_callback=None) -> pd.DataFrame:
    """期权 PCR 历史，返回列: date, underlying, call_volume, put_volume, ...

    _progress_callback(current, total) 用于 UI 显示加载进度；
    进程级全局缓存（跨用户共享，TTL 12 小时），命中时不触发回调。
    """
    cache_key = f"pcr_{lookback_calendar_days}"
    return _get_global(cache_key, _ds.get_option_pcr_history,
                       lookback_calendar_days, _progress_callback=_progress_callback)


def get_option_broker_holding_history(lookback_calendar_days: int = 45,
                                      _progress_callback=None) -> pd.DataFrame:
    """期权券商持仓排名历史（上交所前 3 标的 × 认购/认沽前 5 会员）。

    返回列: date, underlying, broker, call_oi, put_oi, pcr_oi
    进程级全局缓存（跨用户共享，TTL 12 小时）。
    """
    cache_key = f"opt_broker_{lookback_calendar_days}"
    return _get_global(cache_key, _ds.get_option_broker_holding_history,
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
    进程级全局缓存（跨用户共享，TTL 12 小时）。
    """
    cache_key = f"citic_{lookback_calendar_days}"
    return _get_global(cache_key, _ds.get_citic_position_history,
                       lookback_calendar_days, vars_list,
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


@cache_data(ttl=24 * 3600)
def get_cot_gold(years: int = 3) -> pd.DataFrame:
    """CFTC COT 黄金持仓，返回列: date, open_interest, mm_long, ..., mm_net"""
    return _ds.get_cot_gold(years)


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
