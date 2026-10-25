# -*- coding: utf-8 -*-
"""
indicators.py — 指标计算层

核心公式（与视频“债券-股票息差系统”对应）：
  盈利收益率 E/P = 100 / 指数滚动市盈率(TTM)         单位：%
  股债风险溢价 ERP = E/P - 10年期国债收益率           单位：百分点
    · ERP 越高 → 股票相对债券越便宜 → 视频中蓝线触及“标准差上轨”=股市见底信号
    · ERP 越低 → 股票相对债券越贵 → 触及“标准差下轨”=股市见顶信号
  轨道 = ERP 滚动均值 ± k × 滚动标准差（默认 ±1σ / ±2σ 两组）

其余工具：PCR、期指基差、COT 净持仓的计算已在抓取层完成，本模块提供
信号判定与历史分位等纯函数，方便单元测试与后续迁移到回测框架。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def earnings_yield(pe: pd.Series) -> pd.Series:
    """市盈率 -> 盈利收益率(%)。"""
    pe = pd.to_numeric(pe, errors="coerce")
    return 100.0 / pe


def build_erp(pe_monthly: pd.DataFrame, bond_daily: pd.DataFrame) -> pd.DataFrame:
    """
    用月频 PE 与日频国债收益率构造月频股债风险溢价序列。

    输入:
      pe_monthly : columns [date, pe_ttm]
      bond_daily : columns [date, bond_yield]
    返回:
      columns [date, pe_ttm, earnings_yield, bond_yield, erp, erp_reverse]
      erp         = E/P - 10Y（上轨=股票便宜=底部信号，与视频语义一致）
      erp_reverse = 10Y - E/P（口播字面方向，仅作对照）
    """
    pe = pe_monthly.sort_values("date").copy()
    bond = bond_daily.sort_values("date").copy()
    # 对每个 PE 月末日期，取不晚于该日的最近一个国债收益率（merge_asof 要求两侧都排序）
    pe["ey"] = earnings_yield(pe["pe_ttm"])
    merged = pd.merge_asof(
        pe.rename(columns={"ey": "earnings_yield"}),
        bond, on="date", direction="backward",
    )
    merged["erp"] = merged["earnings_yield"] - merged["bond_yield"]
    merged["erp_reverse"] = merged["bond_yield"] - merged["earnings_yield"]
    return merged.dropna(subset=["erp"]).reset_index(drop=True)


def build_erp_daily(pe_daily: pd.DataFrame, bond_daily: pd.DataFrame) -> pd.DataFrame:
    """
    用日频 PE 与日频国债收益率构造日频股债风险溢价序列。

    输入:
      pe_daily  : columns [date, pe_ttm]（交易日）
      bond_daily: columns [date, bond_yield]（日频）
    返回:
      columns [date, pe_ttm, earnings_yield, bond_yield, erp, erp_reverse]
      erp = E/P - 10Y
    两者均为日频但交易日可能不完全对齐（如节假日差异），用 merge_asof backward
    取不晚于 PE 日期的最近一个国债收益率。
    """
    pe = pe_daily.sort_values("date").copy()
    bond = bond_daily.sort_values("date").copy()
    pe["ey"] = earnings_yield(pe["pe_ttm"])
    merged = pd.merge_asof(
        pe.rename(columns={"ey": "earnings_yield"}),
        bond, on="date", direction="backward",
    )
    merged["erp"] = merged["earnings_yield"] - merged["bond_yield"]
    merged["erp_reverse"] = merged["bond_yield"] - merged["earnings_yield"]
    return merged.dropna(subset=["erp"]).reset_index(drop=True)


def add_std_bands(df: pd.DataFrame, value_col: str = "erp",
                  window: int = 36, n_sigma: float = 2.0) -> pd.DataFrame:
    """
    给时间序列添加滚动均值与 ±1σ / ±2σ 标准差轨道。
    window: 滚动窗口（月频数据下 36 ≈ 3 年）。
    返回新增列: mean, std, upper1, lower1, upper2, lower2,
               band_upper(=±n_sigma), band_lower。
    """
    out = df.copy()
    s = out[value_col]
    # min_periods 不得超过 window，否则 pandas 抛 ValueError；
    # 长窗口下需要更多初始样本才能得到稳定的统计量，至少取窗口的一半
    minp = min(window, max(12, window // 2))
    out["mean"] = s.rolling(window, min_periods=minp).mean()
    out["std"] = s.rolling(window, min_periods=minp).std(ddof=1)  # 样本标准差，金融时序惯例
    out["upper1"] = out["mean"] + out["std"]
    out["lower1"] = out["mean"] - out["std"]
    out["upper2"] = out["mean"] + 2 * out["std"]
    out["lower2"] = out["mean"] - 2 * out["std"]
    out["band_upper"] = out["mean"] + n_sigma * out["std"]
    out["band_lower"] = out["mean"] - n_sigma * out["std"]
    return out


def rolling_percentile(series: pd.Series, window: int = 60) -> pd.Series:
    """滚动历史分位（0~100）：当前值在过去 window 期样本中的百分位。"""
    def _pct(x):
        return (x.iloc[:-1] <= x.iloc[-1]).mean() * 100 if len(x) > 1 else np.nan
    minp = min(window, max(12, window // 3))
    return series.rolling(window, min_periods=minp).apply(_pct, raw=False)


def erp_signal(latest_erp: float, upper: float, lower: float) -> tuple[str, str]:
    """
    根据 ERP 与轨道的关系返回 (信号, 说明)。
    视频规则: 触上轨=股市见底(股票极具吸引力); 触下轨=股市见顶。
    """
    if pd.isna(upper) or pd.isna(lower):
        return ("数据不足", "滚动窗口内样本不足，轨道尚未形成")
    if latest_erp >= upper:
        return ("底部区域（股票显著低估）",
                "ERP 触及标准差上轨，股票盈利收益率显著高于债券，视频体系中的股市见底信号")
    if latest_erp <= lower:
        return ("顶部区域（股票显著高估）",
                "ERP 跌破标准差下轨，股票相对债券失去吸引力，视频体系中的股市见顶信号")
    if latest_erp >= (upper + latest_erp) / 2:
        return ("偏股票区域", "ERP 位于均值上方，股票性价比占优")
    if latest_erp <= (lower + latest_erp) / 2:
        return ("偏债券区域", "ERP 位于均值下方，债券性价比占优")
    return ("中性震荡", "ERP 处于均值附近，股债性价比均衡")


def pcr_signal(pcr: float) -> str:
    """PCR 经验解读（仅情绪参考，阈值随品种/时期漂移）。"""
    if pd.isna(pcr):
        return "—"
    if pcr >= 1.1:
        return "认沽持仓/成交拥挤，情绪偏悲观（逆向看多）"
    if pcr <= 0.6:
        return "认购占优，情绪偏乐观（逆向谨慎）"
    return "情绪中性"


def basis_signal(basis_pct: float) -> str:
    """股指期货基差率文字解读（%）。升水为正、贴水为负。"""
    if pd.isna(basis_pct):
        return "—"
    if basis_pct > 0.3:
        return "深度升水，杠杆资金情绪亢奋（顶部警惕）"
    if basis_pct > 0:
        return "小幅升水，情绪偏乐观"
    if basis_pct > -0.5:
        return "小幅贴水，情绪中性偏谨慎"
    return "深度贴水，对冲需求强/情绪低迷（底部特征之一）"


def citic_net_signal(net_oi: float, long_oi: float, short_oi: float) -> str:
    """
    中信期货净持仓文字解读。
    net_oi = 多单 - 空单（手）；long_oi / short_oi 为多空持仓量。
    中信作为头部期货公司，其净持仓方向常被视为大资金情绪风向标。
    """
    if pd.isna(net_oi) or (long_oi == 0 and short_oi == 0):
        return "数据不足"
    if short_oi > 0:
        ratio = long_oi / short_oi
    else:
        ratio = float("inf") if long_oi > 0 else 1.0
    if net_oi > 0 and ratio >= 1.3:
        return "中信显著净多，大资金偏多（注意其为代客持仓，反映客户整体方向）"
    if net_oi > 0:
        return "中信小幅净多，方向偏多"
    if net_oi < 0 and ratio <= 0.7:
        return "中信显著净空，大资金偏空/对冲压力大"
    if net_oi < 0:
        return "中信小幅净空，方向偏空"
    return "多空基本均衡"


# ===========================================================================
# 离岸在岸息差系统
# ===========================================================================
def build_usd_offshore_onshore_spread(effr: pd.DataFrame,
                                      sr3: pd.DataFrame) -> pd.DataFrame:
    """美元离岸-在岸息差 = 隐含离岸利率 - EFFR。

    输入:
      effr : columns [date, effr]  有效联邦基金利率(%)
      sr3  : columns [date, sr3_close]  3个月SOFR期货收盘价
    返回:
      columns [date, effr, sr3_close, implied_offshore, spread_pct, spread_bp]
      implied_offshore = 100 - sr3_close
      spread_pct = implied_offshore - effr
      spread_bp  = spread_pct * 100
    """
    df = pd.merge(effr, sr3, on="date", how="inner").sort_values("date").reset_index(drop=True)
    df["implied_offshore"] = 100.0 - df["sr3_close"]
    df["spread_pct"] = df["implied_offshore"] - df["effr"]
    df["spread_bp"] = df["spread_pct"] * 100.0
    return df


def build_cny_rate_spread(cnh_hibor: pd.DataFrame,
                          shibor: pd.DataFrame) -> pd.DataFrame:
    """人民币离在岸息差 = 3M CNH Hibor - 3M Shibor。

    输入:
      cnh_hibor : columns [date, cnh_hibor_3m]（离岸人民币拆借利率 %）
      shibor    : columns [date, 3个月]（在岸 SHIBOR %）
    返回:
      columns [date, cnh_hibor_3m, shibor_3m, spread_pct, spread_bp]
    """
    sh = shibor[["date", "3个月"]].rename(columns={"3个月": "shibor_3m"}).copy()
    df = pd.merge(cnh_hibor, sh, on="date", how="inner").sort_values("date").reset_index(drop=True)
    df["spread_pct"] = df["cnh_hibor_3m"] - df["shibor_3m"]
    df["spread_bp"] = df["spread_pct"] * 100.0
    return df


def build_cny_fx_spread(usd_cnh: pd.DataFrame,
                        usd_cny: pd.DataFrame) -> pd.DataFrame:
    """人民币离在岸汇差 = (USD/CNH - USD/CNY) × 10000（单位 bp）。

    汇差为正 → 离岸贬值预期强于在岸（外资看空人民币资产）。
    """
    df = pd.merge(usd_cnh, usd_cny, on="date", how="inner").sort_values("date").reset_index(drop=True)
    df["spread_bp"] = (df["usd_cnh"] - df["usd_cny"]) * 10000.0
    return df


def build_bond_term_spread(bond_curve: pd.DataFrame) -> pd.DataFrame:
    """国债长短端利差 = 10Y - 2Y（单位 %）。

    输入: columns [date, bond_2y, bond_10y]
    返回: columns [date, bond_2y, bond_10y, spread_pct]
    """
    df = bond_curve.copy()
    df["spread_pct"] = df["bond_10y"] - df["bond_2y"]
    return df.dropna(subset=["spread_pct"]).reset_index(drop=True)


def build_erp_from_curve(pe_monthly: pd.DataFrame,
                         bond_curve: pd.DataFrame) -> pd.DataFrame:
    """A股股债收益差 = 沪深300盈利收益率 - 10Y国债收益率。

    复用 build_erp 的逻辑，但接受 bond_yield_curve 输出（含 10Y 列）。
    """
    bond = bond_curve[["date", "bond_10y"]].rename(columns={"bond_10y": "bond_yield"}).copy()
    return build_erp(pe_monthly, bond)


def resonance_signal(fx_spread_bp: float, rate_spread_bp: float,
                     erp_percentile: float, term_spread_pct: float
                     ) -> tuple[str, str]:
    """内外资金共振判断。

    外部资金流入信号：汇差收窄(fx_spread_bp < 0) 且 息差收窄(rate_spread_bp < 0)
    内部吸引力强：ERP 历史分位 > 50（股优于债）
    返回 (象限标签, 解读)
    """
    external_in = (pd.notna(fx_spread_bp) and fx_spread_bp < 0
                   and pd.notna(rate_spread_bp) and rate_spread_bp < 0)
    internal_strong = pd.notna(erp_percentile) and erp_percentile > 50

    if external_in and internal_strong:
        return ("内外资金共振做多", "外资流入 + 股债收益差高位 → 牛市底部/启动期，指数级行情概率高")
    if (not external_in) and (not internal_strong):
        return ("内外资金共振撤离", "外资流出 + 股债收益差低位 → 熊市风险高，整体持续承压")
    if external_in and (not internal_strong):
        return ("外资主导结构性行情", "外资流入但内部吸引力弱 → 指数震荡，外资重仓赛道相对强势")
    return ("内资主导修复行情", "外资流出但内部吸引力强 → 政策与内资驱动，外资重仓板块偏弱")
#（注：内容由AI生成）
