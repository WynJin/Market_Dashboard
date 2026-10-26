# -*- coding: utf-8 -*-
"""
factory.py — 数据源工厂 + 路由 + 自动降级

设计模式：
  - 工厂模式（Factory）：根据配置创建数据源实例
  - 责任链模式（Chain of Responsibility）：主源失败自动试备源

路由表 METHOD_ROUTING 定义每个方法的"主源 -> 备源"优先级链。
业务层通过 SourceManager.get_xxx() 调用，完全不感知底层用了哪个源。
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd

from .protocol import DataSource

logger = logging.getLogger(__name__)

# 配置文件路径（项目根目录下的 config.json，已加入 .gitignore）
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


def _load_config() -> dict:
    """读取 config.json；文件不存在或解析失败时返回空 dict。"""
    if not _CONFIG_PATH.exists():
        return {}
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"读取配置文件 {_CONFIG_PATH} 失败: {e}")
        return {}


# 配置优先级：环境变量 > config.json > 默认值（None）
_cfg = _load_config()
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN") or _cfg.get("tushare_token")
TUSHARE_API_URL = os.getenv("TUSHARE_API_URL", _cfg.get("tushare_api_url", ""))

# 每个方法的数据源优先级链（排在前面的优先尝试，失败则降级到下一个）
# tushare 不支持/不适合的方法（期权 PCR、CFTC）只列 akshare
METHOD_ROUTING: dict[str, list[str]] = {
    "get_bond_10y":                  ["akshare"],              # 实测代理不支持债券接口
    "get_index_pe_monthly":          ["tushare", "akshare"],
    "get_index_pe_daily":            ["tushare", "akshare"],
    "get_index_pe_daily_csindex":    ["tushare", "akshare"],
    "get_index_daily":               ["tushare", "akshare"],
    "get_shibor":                    ["akshare", "tushare"],   # akshare 一次拿全历史更快
    "get_shibor_irs_curve":          ["akshare"],              # 中国货币网 chinamoney.com.cn，tushare 无此接口
    "get_option_pcr_history":        ["tushare", "akshare"],  # tushare opt_daily 单日全量，调用少一半
    "get_option_broker_holding_history": ["akshare"],          # 仅上交所公开持仓排名，tushare 无此接口
    "get_index_futures_basis_history": ["tushare", "akshare"],
    "get_citic_position_history":    ["akshare"],              # 代理 fut_holding 不支持 CFFEX
    "get_gold_curve_latest":         ["akshare", "tushare"],
    "get_gold_main_daily":           ["akshare", "tushare"],
    "get_cot_gold":                  ["akshare"],              # 只有 cftc.gov 直连
    # 离岸在岸息差系统
    "get_effr":                      ["akshare"],              # FRED API，tushare 无此接口
    "get_sofr_futures":              ["akshare"],              # yfinance / FRED，tushare 无
    "get_cnh_hibor":                 ["akshare"],              # 东方财富，tushare 无
    "get_usd_cny":                   ["akshare"],              # 央行中间价，tushare 无
    "get_usd_cnh":                   ["akshare"],              # 东方财富 push2his / 新浪，tushare 无
    "get_bond_yield_curve":          ["akshare"],              # 中国国债信息网
    "get_us_macro_events":           ["akshare"],              # FRED DFEDTARU/CPIAUCSL/PPIACO
    "get_us_treasury_yields":        ["akshare"],              # FRED DGS* 系列
}


class SourceManager:
    """数据源管理器：负责创建实例、按路由表调度、失败自动降级。"""

    def __init__(self, default_source: str = "akshare"):
        self._sources: dict[str, DataSource] = {}
        self._default = default_source
        self._last_source: dict[str, str] = {}  # 方法名 -> 实际命中的源名
        self._init_sources()

    def _init_sources(self) -> None:
        # akshare 始终可用（免费无门槛）
        try:
            from .akshare_source import AkShareSource
            self._sources["akshare"] = AkShareSource()
            logger.info("akshare 数据源已就绪")
        except Exception as e:
            logger.error(f"akshare 初始化失败: {e}")

        # tushare 需 token；无 token 或积分不足则跳过，对应方法自动走 akshare
        token = TUSHARE_TOKEN
        api_url = TUSHARE_API_URL
        # 空串表示用官方地址
        api_url = api_url if api_url else None
        if token:
            try:
                from .tushare_source import TushareSource
                src = TushareSource(token, api_url=api_url)
                # 简单连通性自检：查一次用户积分
                try:
                    user = src.pro.query("user")
                    points = user.get("points", 0) if isinstance(user, dict) else 0
                    logger.info(f"tushare 数据源已就绪（{src._api_url}，积分: {points}）")
                except Exception:
                    logger.info(f"tushare 数据源已就绪（{src._api_url}，无法查询积分）")
                self._sources["tushare"] = src
            except Exception as e:
                logger.warning(f"tushare 初始化失败，相关方法将降级到 akshare: {e}")
        else:
            logger.info("未配置 tushare token，仅使用 akshare")

    @property
    def available_sources(self) -> list[str]:
        return list(self._sources.keys())

    def _chain_for(self, method_name: str) -> list[str]:
        """取某方法的优先级链，未在路由表中的方法用默认源 + 全部可用源兜底。"""
        chain = METHOD_ROUTING.get(method_name, [self._default])
        # 确保链中的源都已初始化，跳过不可用的
        return [s for s in chain if s in self._sources]

    def call(self, method_name: str, *args, **kwargs) -> pd.DataFrame:
        """按优先级链调用方法，失败自动降级到下一个源。"""
        chain = self._chain_for(method_name)
        if not chain:
            raise RuntimeError(f"没有可用的数据源处理 {method_name}")

        errors: list[str] = []
        for src_name in chain:
            src = self._sources[src_name]
            method = getattr(src, method_name, None)
            if method is None:
                continue
            try:
                result = method(*args, **kwargs)
                self._last_source[method_name] = src_name
                return result
            except NotImplementedError:
                # 该源明确不支持此方法，直接跳到下一个
                logger.debug(f"{src_name}.{method_name} 不支持，降级")
                continue
            except Exception as e:
                errors.append(f"{src_name}: {type(e).__name__}: {e}")
                logger.warning(f"{src_name}.{method_name} 失败，尝试下一个源: {e}")
                continue
        raise RuntimeError(f"方法 {method_name} 在所有数据源均失败: {' | '.join(errors)}")

    def get_last_source(self, method_name: str) -> str | None:
        """返回某方法最近一次实际命中的数据源名（akshare / tushare）。"""
        return self._last_source.get(method_name)

    def __getattr__(self, method_name: str):
        """透传：让调用方可以直接 ds.get_bond_10y(...) 而不用 ds.call('get_bond_10y', ...)"""
        # 避免 __getattr__ 在 _sources 等内部属性未初始化时被递归调用
        if method_name.startswith("_"):
            raise AttributeError(method_name)

        def wrapper(*args, **kwargs):
            return self.call(method_name, *args, **kwargs)

        wrapper.__name__ = method_name
        return wrapper


# 模块级单例（懒加载）
_default_manager: SourceManager | None = None


def get_default_manager() -> SourceManager:
    """获取全局默认数据源管理器（单例）。"""
    global _default_manager
    if _default_manager is None:
        _default_manager = SourceManager(default_source=os.getenv("DATA_SOURCE", "akshare"))
    return _default_manager


def reload_manager(default_source: str = "akshare") -> SourceManager:
    """重建全局管理器（用于运行时切换默认数据源）。"""
    global _default_manager
    _default_manager = SourceManager(default_source=default_source)
    return _default_manager


# 回溯天数默认值（来自 config.json，缺失时回退硬编码默认）
_LB_DEFAULTS = None


def get_lookback_defaults() -> dict:
    """返回 {"pcr": int, "futures": int}，优先读 config.json。"""
    global _LB_DEFAULTS
    if _LB_DEFAULTS is None:
        cfg = _load_config()
        _LB_DEFAULTS = {
            "pcr": int(cfg.get("pcr_lookback_days", 30)),
            "futures": int(cfg.get("futures_lookback_days", 120)),
            "irs": int(cfg.get("irs_lookback_days", 90)),
        }
    return _LB_DEFAULTS
