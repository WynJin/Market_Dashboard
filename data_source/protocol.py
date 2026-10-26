# -*- coding: utf-8 -*-
"""
protocol.py — 数据源统一协议（抽象基类）

所有数据源（akshare / tushare / ...）必须实现本协议。
每个方法的返回 DataFrame 的"列名 + 类型"是契约，业务层只依赖契约，不依赖具体数据源。
设计模式：策略模式（Strategy）—— 不同数据源是可互换的策略。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class DataSource(ABC):
    """数据源抽象基类。子类需实现全部 @abstractmethod。"""

    name: str = "base"  # 数据源标识，如 "akshare" / "tushare"

    # ------------------------------------------------------------------
    # 1. 10 年期国债收益率
    # ------------------------------------------------------------------
    @abstractmethod
    def get_bond_10y(self, start_date: str = "20100101") -> pd.DataFrame:
        """返回列: date(datetime), bond_yield(float, 单位 %)"""

    # ------------------------------------------------------------------
    # 2. 指数市盈率（月频，长期轨道用）
    # ------------------------------------------------------------------
    @abstractmethod
    def get_index_pe_monthly(self, symbol: str = "沪深300") -> pd.DataFrame:
        """返回列: date(月末日期), pe_ttm(float)"""

    # ------------------------------------------------------------------
    # 3. 指数市盈率（日频，近约 20 个交易日补充精度）
    # ------------------------------------------------------------------
    @abstractmethod
    def get_index_pe_daily_csindex(self, csindex_code: str = "000300") -> pd.DataFrame:
        """返回列: date, pe_static, pe_ttm, div_yield"""

    # ------------------------------------------------------------------
    # 4. 大盘指数日线
    # ------------------------------------------------------------------
    @abstractmethod
    def get_index_daily(self, sina_code: str = "sh000300") -> pd.DataFrame:
        """返回列: date, open, high, low, close, volume"""

    # ------------------------------------------------------------------
    # 5. SHIBOR
    # ------------------------------------------------------------------
    @abstractmethod
    def get_shibor(self) -> pd.DataFrame:
        """返回列: date + 8 个期限（隔夜/1周/2周/1个月/3个月/6个月/9个月/1年），单位 %"""

    # ------------------------------------------------------------------
    # 5b. Shibor3M 利率互换（IRS）收盘曲线
    # ------------------------------------------------------------------
    @abstractmethod
    def get_shibor_irs_curve(self, lookback_days: int = 30) -> pd.DataFrame:
        """Shibor3M 利率互换收盘曲线（中国货币网）。

        返回长格式 DataFrame，列: date(datetime), tenor(str, 6M/9M/1Y/2Y/3Y/4Y/5Y/7Y/10Y),
        bid(float, 报买%), avg(float, 均值%), ask(float, 报卖%)。
        lookback_days: 历史回溯天数（中国货币网仅提供近一年数据）。
        """

    # ------------------------------------------------------------------
    # 6. 期权 PCR（上交所 + 深交所官方日统计）
    # ------------------------------------------------------------------
    @abstractmethod
    def get_option_pcr_history(self, lookback_calendar_days: int = 45,
                               start_date: date | None = None) -> pd.DataFrame:
        """返回列: date, underlying, exp_month, call_volume, put_volume, call_oi, put_oi,
        pcr_volume, pcr_oi（exp_month 为到期月份 YYMM，akshare 无细分时为"全部"）。

        若提供 start_date，则从该日期拉到今天（忽略 lookback_calendar_days），
        用于增量补足。
        """

    # ------------------------------------------------------------------
    # 6b. 期权券商持仓排名（上交所前 3 大标的 × 认购/认沽前 5 会员）
    # ------------------------------------------------------------------
    @abstractmethod
    def get_option_broker_holding_history(
        self, lookback_calendar_days: int = 45, _progress_callback=None,
        start_date: date | None = None,
    ) -> pd.DataFrame:
        """返回列: date, underlying, broker, call_oi, put_oi, pcr_oi
        （仅持仓量；上交所未公开券商级交易量排名）。

        若提供 start_date，则从该日期拉到今天（忽略 lookback_calendar_days）。
        """

    # ------------------------------------------------------------------
    # 7. 股指期货升贴水
    # ------------------------------------------------------------------
    @abstractmethod
    def get_index_futures_basis_history(self, lookback_calendar_days: int = 45) -> pd.DataFrame:
        """返回列: date, variety, variety_name, contract, futures_close, spot_close,
        basis, basis_pct"""

    # ------------------------------------------------------------------
    # 7b. 中信期货多空持仓
    # ------------------------------------------------------------------
    @abstractmethod
    def get_citic_position_history(
        self, lookback_calendar_days: int = 45, vars_list: list[str] | None = None,
        start_date: date | None = None,
    ) -> pd.DataFrame:
        """返回列: date, variety, contract, long_oi, long_oi_chg, long_rank,
        short_oi, short_oi_chg, short_rank, net_oi。

        若提供 start_date，则从该日期拉到今天（忽略 lookback_calendar_days）。
        """

    # ------------------------------------------------------------------
    # 8. 黄金期货期限结构（上期所 AU 系列）
    # ------------------------------------------------------------------
    @abstractmethod
    def get_gold_curve_latest(self) -> pd.DataFrame:
        """返回列: date, contract, expire_date, close, settle, volume, open_interest"""

    # ------------------------------------------------------------------
    # 9. 黄金主力连续日线
    # ------------------------------------------------------------------
    @abstractmethod
    def get_gold_main_daily(
        self, start_date: str = "20180101", end_date: str | None = None
    ) -> pd.DataFrame:
        """返回列: date, close, volume, open_interest"""

    # ------------------------------------------------------------------
    # 10. CFTC COT 持仓报告（COMEX 黄金）
    # ------------------------------------------------------------------
    @abstractmethod
    def get_cot_gold(self, years: int = 3) -> pd.DataFrame:
        """返回列: date, open_interest, mm_long, mm_short, mm_spread, mm_net"""
