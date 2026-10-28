# -*- coding: utf-8 -*-
"""
akshare_source.py — AkShare 数据源适配器

把 akshare 的中文字段体系翻译成项目统一契约列名。
设计模式：适配器模式（Adapter）。

============================================================
数据源渠道说明（一级 = akshare 数据源库；二级 = 原始发布渠道 + 频率）
============================================================
  1. 10 年期国债收益率   akshare.bond_zh_us_rate        （中国国债信息网，日频）
  2. 指数市盈率（长期）   akshare.stock_index_pe_lg      （乐咕乐股，月频，TTM 滚动市盈率）
  3. 指数市盈率（近期）   akshare.stock_zh_index_value_csindex（中证指数官网，日频，近约 20 个交易日）
  4. 大盘指数行情         akshare.stock_zh_index_daily   （新浪财经，日频）
  5. SHIBOR              akshare.macro_china_shibor_all （全国银行间同业拆借中心 shibor.org，日频）
  6. 期权 PCR            akshare.option_daily_stats_sse/szse（上交所/深交所官方股票期权日统计，日频）
  7. 股指期货升贴水       akshare.get_cffex_daily        （中金所官方日行情，日频）
  7b.中信期货多空持仓     akshare.get_cffex_rank_table   （中金所前 20 会员持仓排名，日频）
  8. 黄金期货期限结构     akshare.get_shfe_daily         （上期所官方日行情，日频）
  9. 黄金主力连续         akshare.futures_main_sina      （新浪财经，日频）
 10. CFTC COT 持仓报告   cftc.gov 官方文件               （CFTC Disaggregated Futures-only，周频）

注：全部数据均为公开免费渠道，仅供学习研究，不构成投资建议。
"""
from __future__ import annotations

import io
import logging
import zipfile
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

import akshare as ak

from .protocol import DataSource
from .utils import iter_days_from, iter_recent_days, to_dt

logger = logging.getLogger(__name__)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
HEADERS = {"User-Agent": UA}

# 乐咕支持的常见宽基
PE_INDEX_CHOICES = ["沪深300", "上证50", "中证500", "创业板指", "科创50", "上证指数"]

# SHIBOR 期限 -> akshare 列名
_SHIBOR_MAP = {
    "隔夜": "O/N-定价", "1周": "1W-定价", "2周": "2W-定价",
    "1个月": "1M-定价", "3个月": "3M-定价", "6个月": "6M-定价",
    "9个月": "9M-定价", "1年": "1Y-定价",
}

# 中金所品种 -> (新浪现货代码, 中文名)
INDEX_FUTURES_MAP = {
    "IF": ("sh000300", "沪深300"),
    "IH": ("sh000016", "上证50"),
    "IC": ("sh000905", "中证500"),
    "IM": ("sh000852", "中证1000"),
}

CITIC_KEYWORD = "中信期货"

# COMEX 黄金 CFTC 代码（注意排除 MICRO GOLD / PAX GOLD）
GOLD_CFTC_CODE = "088691"

# ------------------------------------------------------------------
# 数据源渠道映射：方法名 -> 二级渠道说明（一级固定为 "akshare"）
# 用于 UI 展示每个数据项的来源与频率
# ------------------------------------------------------------------
SOURCE_CHANNELS = {
    "get_bond_10y": "中国国债信息网，日频",
    "get_index_pe_monthly": "乐咕乐股，月频（TTM 滚动市盈率）",
    "get_index_pe_daily": "乐咕乐股月频前向填充至日频（TTM 滚动市盈率），日频",
    "get_index_pe_daily_csindex": "中证指数官网，日频（近约 20 个交易日）",
    "get_index_daily": "新浪财经，日频",
    "get_shibor": "全国银行间同业拆借中心（shibor.org），日频",
    "get_option_pcr_history": "上交所 / 深交所官方股票期权日统计，日频",
    "get_index_futures_basis_history": "中金所官方日行情，日频（现货指数来自新浪财经）",
    "get_citic_position_history": "中金所前 20 会员持仓排名，日频",
    "get_gold_curve_latest": "上期所官方日行情，日频",
    "get_gold_main_daily": "新浪财经，日频",
    "get_cot_gold": "CFTC 官方 Disaggregated Futures-only 报告（cftc.gov），周频",
    "get_option_broker_holding_history": "上交所期权持仓排名统计（qqccpmtj1_online，前 3 大标的 × 认购/认沽前 5 名会员），日频，仅持仓量",
    "get_effr": "美联储 FRED 数据库（EFFR 序列），日频",
    "get_sofr_futures": "优先本地 CSV（investing.com CME 3M SOFR 期货主连），降级 yfinance SR3=F / FRED SOFR90DAYAVG，日频",
    "get_cnh_hibor": "东方财富-香港银行同业拆借利率（Hibor人民币=CNH），日频",
    "get_usd_cny": "中国银行人民币汇率（央行中间价）/ 新浪财经，日频",
    "get_usd_cnh": "东方财富 push2his（USDCNH 日 K），降级新浪财经实时报价，日频",
    "get_bond_yield_curve": "中国国债信息网（中美国债收益率），日频",
    "get_us_macro_events": "美联储 FRED（DFEDTARU/CPIAUCSL/PPIACO），FOMC/CPI/PPI 事件日",
    "get_us_treasury_yields": "美联储 FRED（DGS1MO~DGS30），美国国债各期限收益率，日频",
}


class AkShareSource(DataSource):
    name = "akshare"

    # ------------------------------------------------------------------
    # 1. 10 年期国债收益率（中国国债信息网）
    # ------------------------------------------------------------------
    def get_bond_10y(self, start_date: str = "20100101") -> pd.DataFrame:
        raw = ak.bond_zh_us_rate(start_date=start_date)
        df = raw[["日期", "中国国债收益率10年"]].copy()
        df.columns = ["date", "bond_yield"]
        df["date"] = to_dt(df["date"])
        df["bond_yield"] = pd.to_numeric(df["bond_yield"], errors="coerce")
        return df.dropna(subset=["date", "bond_yield"]).sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. 指数市盈率（乐咕乐股，月频）
    # ------------------------------------------------------------------
    def get_index_pe_monthly(self, symbol: str = "沪深300") -> pd.DataFrame:
        raw = ak.stock_index_pe_lg(symbol=symbol)
        df = raw[["日期", "滚动市盈率"]].copy()
        df.columns = ["date", "pe_ttm"]
        df["date"] = to_dt(df["date"])
        df["pe_ttm"] = pd.to_numeric(df["pe_ttm"], errors="coerce")
        return df.dropna(subset=["date", "pe_ttm"]).sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2b. 指数市盈率（日频，全历史）
    #     akshare 乐咕仅提供月频，此处将月频 PE 前向填充到日频
    # ------------------------------------------------------------------
    def get_index_pe_daily(self, symbol: str = "沪深300") -> pd.DataFrame:
        monthly = self.get_index_pe_monthly(symbol)
        if monthly.empty:
            return pd.DataFrame(columns=["date", "pe_ttm"])
        monthly = monthly.sort_values("date").set_index("date")
        # 重采样到日频，前向填充（月内 PE 视为不变）
        daily = monthly.resample("D").ffill().dropna().reset_index()
        return daily

    # ------------------------------------------------------------------
    # 3. 指数市盈率（中证指数官网，日频，近约 20 个交易日）
    # ------------------------------------------------------------------
    def get_index_pe_daily_csindex(self, csindex_code: str = "000300") -> pd.DataFrame:
        raw = ak.stock_zh_index_value_csindex(symbol=csindex_code)
        df = raw[["日期", "市盈率1", "市盈率2", "股息率1"]].copy()
        df.columns = ["date", "pe_static", "pe_ttm", "div_yield"]
        df["date"] = to_dt(df["date"])
        for c in ["pe_static", "pe_ttm", "div_yield"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 4. 大盘指数日线（新浪）
    # ------------------------------------------------------------------
    def get_index_daily(self, sina_code: str = "sh000300") -> pd.DataFrame:
        raw = ak.stock_zh_index_daily(symbol=sina_code)
        df = raw.copy()
        df["date"] = to_dt(df["date"])
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 5. SHIBOR
    # ------------------------------------------------------------------
    def get_shibor(self) -> pd.DataFrame:
        raw = ak.macro_china_shibor_all()
        df = pd.DataFrame({"date": to_dt(raw["日期"])})
        for cn, col in _SHIBOR_MAP.items():
            df[cn] = pd.to_numeric(raw[col], errors="coerce")
        return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 5b. Shibor3M 利率互换（IRS）收盘曲线（中国货币网 chinamoney.com.cn）
    # ------------------------------------------------------------------
    # Shibor3M 对应 cfgItemType=71，curveId=16
    _IRS_TENORS = ["6M", "9M", "1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y"]
    _IRS_CHINAMONEY_BASE = "https://www.chinamoney.com.cn/ags/ms/cm-u-bk-shibor"

    def _get_chinamoney_session(self) -> requests.Session:
        """获取中国货币网会话（首次访问需注册 IP，24h 内可直接访问）。"""
        # __bond_register_service 是 akshare 模块级私有函数；用 getattr 避免类内名称修饰
        import akshare.bond.bond_china_money as _bcm
        register_fn = getattr(_bcm, "__bond_register_service", None)
        if register_fn is None:
            # 兼容旧版 akshare：直接创建会话并访问注册页
            session = requests.Session()
            session.get("https://www.chinamoney.com.cn/chinese/bkcurvclosedyhis/"
                        "?bondType=CYCC000&reference=1",
                        headers={"User-Agent": UA}, timeout=15)
            return session
        return register_fn()

    def get_shibor_irs_curve(self, lookback_days: int = 30) -> pd.DataFrame:
        """Shibor3M 利率互换收盘曲线，返回长格式: date, tenor, bid, avg, ask。"""
        session = self._get_chinamoney_session()
        headers = {
            "User-Agent": UA,
            "Referer": "https://www.chinamoney.com.cn/chinese/bkcurvfx/",
            "X-Requested-With": "XMLHttpRequest",
            "Origin": "https://www.chinamoney.com.cn",
        }
        cfg_item = "71"  # Shibor3M
        curve_id = "16"   # interestRateType

        # ---- 1. 拉取当前日曲线（Ifcc 端点，含最新交易日）----
        today_df = pd.DataFrame()
        try:
            r = session.post(
                f"{self._IRS_CHINAMONEY_BASE}/Ifcc",
                params={"lang": "CN", "cfgItemType": cfg_item},
                headers=headers, timeout=15,
            )
            data = r.json()
            recs = data.get("records", [])
            show_date = data.get("data", {}).get("showDateCN")
            if recs and show_date:
                dt = pd.to_datetime(show_date)
                rows = []
                for rec in recs:
                    tenor = rec.get("tl")
                    if tenor not in self._IRS_TENORS:
                        continue
                    rows.append({
                        "date": dt,
                        "tenor": tenor,
                        "bid": pd.to_numeric(rec.get("optimalBid"), errors="coerce"),
                        "avg": pd.to_numeric(rec.get("optimalAvg"), errors="coerce"),
                        "ask": pd.to_numeric(rec.get("optimalAsk"), errors="coerce"),
                    })
                today_df = pd.DataFrame(rows)
        except Exception as e:
            logger.warning(f"Shibor3M IRS 当前日数据拉取失败: {e}")

        # ---- 2. 拉取历史曲线（IfccHis 端点，按月分段，每次最多 1 个月）----
        hist_dfs = []
        end = pd.Timestamp.now().normalize()
        start = end - pd.Timedelta(days=lookback_days)
        # 中国货币网只提供近一年数据
        one_year_ago = end - pd.Timedelta(days=365)
        if start < one_year_ago:
            start = one_year_ago

        cur = start
        while cur <= end:
            seg_end = min(cur + pd.Timedelta(days=28), end)
            params_post = {
                "interestRateType": curve_id,
                "startDate": cur.strftime("%Y-%m-%d"),
                "endDate": seg_end.strftime("%Y-%m-%d"),
                "bidAskType": "",
                "lang": "CN",
                "quoteTime": "全部",
                "pageSize": "5000",
                "pageNum": "1",
            }
            try:
                r = session.post(
                    f"{self._IRS_CHINAMONEY_BASE}/IfccHis",
                    params={"cfgItemType": cfg_item},
                    data=params_post, headers=headers, timeout=20,
                )
                data = r.json()
                recs = data.get("records", [])
                for rec in recs:
                    dt = pd.to_datetime(rec.get("date"))
                    prices = rec.get("configPrice") or []
                    if not prices or len(prices) < len(self._IRS_TENORS):
                        continue
                    ask_bid = rec.get("askBid", "")
                    row = {"date": dt}
                    for i, t in enumerate(self._IRS_TENORS):
                        if i >= len(prices):
                            break
                        val = pd.to_numeric(prices[i], errors="coerce")
                        if ask_bid == "报买":
                            row[t] = {"bid": val}
                        elif ask_bid == "报卖":
                            row[t] = {"ask": val}
                        elif ask_bid == "均值":
                            row[t] = {"avg": val}
                    # 将 {tenor: {bid/avg/ask: val}} 展开为长行
                    for t in self._IRS_TENORS:
                        if t in row and isinstance(row[t], dict):
                            d_row = {"date": dt, "tenor": t}
                            d_row.update(row[t])
                            hist_dfs.append(d_row)
            except Exception as e:
                logger.warning(f"Shibor3M IRS 历史数据 {cur.date()}~{seg_end.date()} 拉取失败: {e}")
            cur = seg_end + pd.Timedelta(days=1)

        hist_df = pd.DataFrame(hist_dfs) if hist_dfs else pd.DataFrame()

        # ---- 3. 合并当前日 + 历史，去重保留最新 ----
        parts = []
        if not hist_df.empty:
            parts.append(hist_df)
        if not today_df.empty:
            parts.append(today_df)
        if not parts:
            return pd.DataFrame(columns=["date", "tenor", "bid", "avg", "ask"])

        merged = pd.concat(parts, ignore_index=True)
        # 同一天同一期限可能来自当前日和历史，保留最后一条（当前日优先）
        merged = merged.drop_duplicates(subset=["date", "tenor"], keep="last")
        for c in ["bid", "avg", "ask"]:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
        merged = merged.sort_values(["date", "tenor"]).reset_index(drop=True)
        return merged


    # ------------------------------------------------------------------
    # 6. 期权 PCR（上交所 + 深交所）
    # ------------------------------------------------------------------
    def _option_stats_one_day(self, d: date) -> pd.DataFrame | None:
        ymd = d.strftime("%Y%m%d")
        rows = []
        for fetcher in (ak.option_daily_stats_sse, ak.option_daily_stats_szse):
            try:
                stat = fetcher(date=ymd)
            except Exception:
                continue
            if stat is None or len(stat) == 0:
                continue
            for _, r in stat.iterrows():
                rows.append({
                    "date": pd.Timestamp(d),
                    "underlying": str(r["合约标的名称"]),
                    "call_volume": float(r["认购成交量"]),
                    "put_volume": float(r["认沽成交量"]),
                    "call_oi": float(r["未平仓认购合约数"]),
                    "put_oi": float(r["未平仓认沽合约数"]),
                })
        if not rows:
            return None
        out = pd.DataFrame(rows)
        agg = out.groupby(["date", "underlying"], as_index=False).agg(
            call_volume=("call_volume", "sum"), put_volume=("put_volume", "sum"),
            call_oi=("call_oi", "sum"), put_oi=("put_oi", "sum"),
        )
        # akshare 日统计按标的聚合（无到期月份细分），统一标记为"全部"
        agg["exp_month"] = "全部"
        agg["pcr_volume"] = agg["put_volume"] / agg["call_volume"]
        agg["pcr_oi"] = agg["put_oi"] / agg["call_oi"]
        return agg

    def get_option_pcr_history(self, lookback_calendar_days: int = 45,
                               _progress_callback=None,
                               start_date: date | None = None) -> pd.DataFrame:
        if start_date is not None:
            days = list(iter_days_from(start_date))
        else:
            days = list(iter_recent_days(lookback_calendar_days))
        total = len(days)
        frames = []
        for i, d in enumerate(days):
            one = self._option_stats_one_day(d)
            if one is not None:
                frames.append(one)
            if _progress_callback:
                _progress_callback(i + 1, total)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        return df.sort_values(["date", "underlying"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 6b. 期权券商持仓排名（上交所 qqccpmtj1_online）
    #     仅前 3 大标的、认购/认沽各前 5 名会员的持仓量；可计算券商级 OI-PCR
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_sse_option_holding(xlsx_bytes: bytes) -> pd.DataFrame | None:
        """解析上交所期权持仓排名 xlsx，返回 long-format:
        underlying, broker, side('call'/'put'), oi
        """
        import io as _io
        try:
            df = pd.read_excel(_io.BytesIO(xlsx_bytes), header=None)
        except Exception:
            return None
        if df is None or len(df) < 3:
            return None
        # 第 0 行是大标题"认购"/"认沽"，第 1 行是列名，第 2 行起是数据
        # 左半（认购）: col0=标的, col2=名次, col3=会员, col4=持仓量
        # 右半（认沽）: col5=标的, col7=名次, col8=会员, col9=持仓量
        rows = []
        cur_call_ul = cur_put_ul = None
        for _, r in df.iloc[2:].iterrows():
            # ---- 认购侧 ----
            ul_call = r.iloc[0] if pd.notna(r.iloc[0]) else None
            if ul_call:
                cur_call_ul = str(ul_call).split("(")[0].strip()
            broker_call = r.iloc[3] if pd.notna(r.iloc[3]) else None
            oi_call = r.iloc[4] if pd.notna(r.iloc[4]) else None
            if broker_call and oi_call is not None and cur_call_ul:
                try:
                    rows.append({"underlying": cur_call_ul, "broker": str(broker_call).strip(),
                                 "side": "call", "oi": int(str(oi_call).replace(",", ""))})
                except (ValueError, TypeError):
                    pass
            # ---- 认沽侧 ----
            ul_put = r.iloc[5] if pd.notna(r.iloc[5]) else None
            if ul_put:
                cur_put_ul = str(ul_put).split("(")[0].strip()
            broker_put = r.iloc[8] if pd.notna(r.iloc[8]) else None
            oi_put = r.iloc[9] if pd.notna(r.iloc[9]) else None
            if broker_put and oi_put is not None and cur_put_ul:
                try:
                    rows.append({"underlying": cur_put_ul, "broker": str(broker_put).strip(),
                                 "side": "put", "oi": int(str(oi_put).replace(",", ""))})
                except (ValueError, TypeError):
                    pass
        if not rows:
            return None
        long_df = pd.DataFrame(rows)
        # 透视: 每个 (underlying, broker) 一行，call_oi / put_oi
        pivot = long_df.pivot_table(index=["underlying", "broker"], columns="side",
                                    values="oi", aggfunc="sum").reset_index()
        if "call" not in pivot.columns:
            pivot["call"] = 0
        if "put" not in pivot.columns:
            pivot["put"] = 0
        pivot = pivot.rename(columns={"call": "call_oi", "put": "put_oi"})
        pivot.columns.name = None  # 清除 pivot 残留的 "side" 列名
        # 某券商可能只上榜认购或认沽一侧，缺失侧填 0
        pivot["call_oi"] = pivot["call_oi"].fillna(0).astype(int)
        pivot["put_oi"] = pivot["put_oi"].fillna(0).astype(int)
        return pivot[["underlying", "broker", "call_oi", "put_oi"]]

    def get_option_broker_holding_history(self, lookback_calendar_days: int = 45,
                                          _progress_callback=None,
                                          start_date: date | None = None) -> pd.DataFrame:
        """上交所期权持仓排名（前 3 大标的 × 认购/认沽前 5 会员）。

        返回列: date, underlying, broker, call_oi, put_oi, pcr_oi
        注意：仅持仓量（上交所未公开券商级交易量排名）；仅覆盖上交所前 3 大标的。
        """
        url = ("https://www.sse.org.cn/api/report/ShowReport"
               "?SHOWTYPE=xlsx&CATALOGID=qqccpmtj1_online&TABKEY=tab1"
               "&txtSearchDate={date}&jy=qqcc_online")
        if start_date is not None:
            days = list(iter_days_from(start_date))
        else:
            days = list(iter_recent_days(lookback_calendar_days))
        total = len(days)
        frames = []
        for i, d in enumerate(days):
            try:
                resp = requests.get(url.format(date=d.strftime("%Y-%m-%d")),
                                    headers={**HEADERS, "Referer": "https://www.sse.org.cn/"},
                                    timeout=15)
                if resp.status_code != 200 or len(resp.content) < 500:
                    if _progress_callback:
                        _progress_callback(i + 1, total)
                    continue
                one = self._parse_sse_option_holding(resp.content)
                if one is not None and not one.empty:
                    one["date"] = pd.Timestamp(d)
                    frames.append(one)
            except Exception:
                pass
            if _progress_callback:
                _progress_callback(i + 1, total)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        df["pcr_oi"] = df["put_oi"] / df["call_oi"].mask(df["call_oi"] == 0)
        return df.sort_values(["date", "underlying", "broker"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 7. 股指期货升贴水
    # ------------------------------------------------------------------
    def _cffex_one_day(self, d: date) -> pd.DataFrame | None:
        try:
            df = ak.get_cffex_daily(date=d.strftime("%Y%m%d"))
        except Exception:
            return None
        if df is None or len(df) == 0:
            return None
        df = df.copy()
        for c in ["close", "volume", "open_interest"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["close"])

    def get_index_futures_basis_history(self, lookback_calendar_days: int = 45) -> pd.DataFrame:
        # 现货一次性拉全历史，避免逐日请求
        spot = {}
        for variety, (sina_code, _name) in INDEX_FUTURES_MAP.items():
            spot[variety] = self.get_index_daily(sina_code)[["date", "close"]].rename(
                columns={"close": "spot_close"})

        rows = []
        for d in iter_recent_days(lookback_calendar_days):
            cff = self._cffex_one_day(d)
            if cff is None:
                continue
            ts = pd.Timestamp(d)
            for variety, (_code, name) in INDEX_FUTURES_MAP.items():
                sub = cff[cff["variety"] == variety]
                if sub.empty:
                    continue
                main = sub.sort_values("volume", ascending=False).iloc[0]
                spot_row = spot[variety].loc[spot[variety]["date"] == ts]
                if spot_row.empty:
                    spot_row = spot[variety].loc[spot[variety]["date"] <= ts].tail(1)
                if spot_row.empty:
                    continue
                spot_close = float(spot_row["spot_close"].iloc[0])
                fut_close = float(main["close"])
                rows.append({
                    "date": ts, "variety": variety, "variety_name": name,
                    "contract": str(main["symbol"]), "futures_close": fut_close,
                    "spot_close": spot_close, "basis": fut_close - spot_close,
                    "basis_pct": (fut_close - spot_close) / spot_close * 100,
                })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(["date", "variety"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 7b. 中信期货多空持仓
    # ------------------------------------------------------------------
    def _extract_citic_one_day(self, d: date, vars_list=None) -> pd.DataFrame | None:
        if vars_list is None:
            vars_list = ["IF", "IH", "IC", "IM"]
        try:
            data = ak.get_cffex_rank_table(date=d.strftime("%Y%m%d"), vars_list=vars_list)
        except Exception:
            return None
        if not data or not isinstance(data, dict):
            return None
        ts = pd.Timestamp(d)
        rows = []
        for contract, df in data.items():
            if df is None or len(df) == 0:
                continue
            df = df[df["rank"] != 999].copy()
            if df.empty:
                continue
            variety = str(df["variety"].iloc[0]) if "variety" in df.columns else contract[:2]
            long_row = df[df["long_party_name"].astype(str).str.contains(CITIC_KEYWORD, na=False)]
            short_row = df[df["short_party_name"].astype(str).str.contains(CITIC_KEYWORD, na=False)]
            long_oi = float(long_row["long_open_interest"].iloc[0]) if not long_row.empty else 0.0
            long_oi_chg = float(long_row["long_open_interest_chg"].iloc[0]) if not long_row.empty else 0.0
            long_rank = int(long_row["rank"].iloc[0]) if not long_row.empty else None
            short_oi = float(short_row["short_open_interest"].iloc[0]) if not short_row.empty else 0.0
            short_oi_chg = float(short_row["short_open_interest_chg"].iloc[0]) if not short_row.empty else 0.0
            short_rank = int(short_row["rank"].iloc[0]) if not short_row.empty else None
            rows.append({
                "date": ts, "variety": variety, "contract": str(contract),
                "long_oi": long_oi, "long_oi_chg": long_oi_chg, "long_rank": long_rank,
                "short_oi": short_oi, "short_oi_chg": short_oi_chg, "short_rank": short_rank,
                "net_oi": long_oi - short_oi,
            })
        if not rows:
            return None
        return pd.DataFrame(rows)

    def get_citic_position_history(self, lookback_calendar_days: int = 45,
                                   vars_list=None, _progress_callback=None,
                                   start_date: date | None = None) -> pd.DataFrame:
        if start_date is not None:
            days = list(iter_days_from(start_date))
        else:
            days = list(iter_recent_days(lookback_calendar_days))
        total = len(days)
        frames = []
        for i, d in enumerate(days):
            one = self._extract_citic_one_day(d, vars_list=vars_list)
            if one is not None:
                frames.append(one)
            if _progress_callback:
                _progress_callback(i + 1, total)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        return df.sort_values(["date", "variety", "contract"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 8. 黄金期货期限结构（上期所 AU 系列）
    # ------------------------------------------------------------------
    def _shfe_one_day(self, d: date) -> pd.DataFrame | None:
        try:
            df = ak.get_shfe_daily(date=d.strftime("%Y%m%d"))
        except Exception:
            return None
        if df is None or len(df) == 0:
            return None
        return df

    def get_gold_curve_latest(self) -> pd.DataFrame:
        for d in iter_recent_days(20):
            df = self._shfe_one_day(d)
            if df is None:
                continue
            au = df[df["symbol"].astype(str).str.match(r"^AU\d{4}$")].copy()
            if au.empty:
                continue
            for c in ["close", "settle", "volume", "open_interest"]:
                au[c] = pd.to_numeric(au[c], errors="coerce")
            au = au.dropna(subset=["close"])
            if au.empty:
                continue
            au = au.rename(columns={"symbol": "contract"})
            au["expire_date"] = au["contract"].map(
                lambda s: pd.Timestamp(year=2000 + int(s[2:4]), month=int(s[4:6]), day=15))
            return au[["date", "contract", "expire_date", "close", "settle",
                       "volume", "open_interest"]].sort_values("expire_date").reset_index(drop=True)
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # 9. 黄金主力连续日线（新浪）
    # ------------------------------------------------------------------
    def get_gold_main_daily(self, start_date: str = "20180101",
                            end_date: str | None = None) -> pd.DataFrame:
        end_date = end_date or datetime.today().strftime("%Y%m%d")
        raw = ak.futures_main_sina(symbol="AU0", start_date=start_date, end_date=end_date)
        df = raw.rename(columns={"收盘价": "close", "成交量": "volume", "持仓量": "open_interest"})
        df["date"] = to_dt(df["日期"])
        for c in ["close", "volume", "open_interest"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df[["date", "close", "volume", "open_interest"]].dropna(
            subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 10. CFTC COT 持仓报告（COMEX 黄金，cftc.gov 官方文件）
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_cot_zip_text(text: str) -> pd.DataFrame:
        df = pd.read_csv(io.StringIO(text), low_memory=False).rename(columns={
            "Market_and_Exchange_Names": "market",
            "Report_Date_as_YYYY-MM-DD": "date",
            "CFTC_Contract_Market_Code": "cftc_code",
            "Open_Interest_All": "open_interest",
            "M_Money_Positions_Long_All": "mm_long",
            "M_Money_Positions_Short_All": "mm_short",
            "M_Money_Positions_Spread_All": "mm_spread",
        })
        gold = df[df["cftc_code"].astype(str).str.strip() == GOLD_CFTC_CODE].copy()
        if gold.empty:
            gold = df[df["market"].astype(str).str.strip().str.startswith(
                "GOLD - COMMODITY EXCHANGE", na=False)].copy()
        gold["date"] = to_dt(gold["date"])
        for c in ["open_interest", "mm_long", "mm_short", "mm_spread"]:
            gold[c] = pd.to_numeric(gold[c], errors="coerce")
        gold["mm_net"] = gold["mm_long"] - gold["mm_short"]
        return gold[["date", "open_interest", "mm_long", "mm_short", "mm_spread", "mm_net"]]

    def get_cot_gold(self, years: int = 3) -> pd.DataFrame:
        this_year = date.today().year
        frames = []
        last_err = None
        for yr in range(this_year, this_year - years, -1):
            url = f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{yr}.zip"
            try:
                # 注意：cftc.gov 对浏览器 UA 会返回 403，必须用默认 UA；
                # 超时设为 (连接10s, 读取60s)，避免网络卡住时长时间阻塞
                resp = requests.get(url, timeout=(10, 60))
            except requests.RequestException as e:
                last_err = e
                break  # 网络不通时后续年份大概率同样失败，直接放弃
            if resp.status_code != 200:
                continue
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                inner = zf.namelist()[0]
                with zf.open(inner) as f:
                    text = f.read().decode("utf-8", errors="ignore")
            frames.append(self._parse_cot_zip_text(text))
        if not frames:
            raise RuntimeError(
                f"cftc.gov 直连失败（国内网络通常无法访问，需代理）：{last_err or '无有效响应'}")
        out = pd.concat(frames, ignore_index=True)
        return out.dropna(subset=["date"]).drop_duplicates("date").sort_values(
            "date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # FRED 辅助方法（美联储圣路易斯分行公开 API，国内可直连，无需 VPN）
    # ------------------------------------------------------------------
    @staticmethod
    def _fred_key() -> str:
        """从 config.json 读取 FRED API key，缺失时返回空串（仍可走公共限流接口）。"""
        try:
            from .factory import _load_config
            return _load_config().get("fred_api_key", "") or ""
        except Exception:
            return ""

    def _fred_series(self, series_id: str, start_date: str = "20200101") -> pd.DataFrame:
        """拉取 FRED 单个序列，返回列 [date, value]。失败返回空 DataFrame。"""
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self._fred_key(),
            "file_type": "json",
            "observation_start": f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}",
            "sort_order": "asc",
        }
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                return pd.DataFrame()
            data = resp.json()
            rows = data.get("observations", [])
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)[["date", "value"]]
            df["date"] = to_dt(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # 11. 有效联邦基金利率 EFFR（FRED）
    # ------------------------------------------------------------------
    def get_effr(self, start_date: str = "20200101") -> pd.DataFrame:
        df = self._fred_series("EFFR", start_date)
        if df.empty:
            return pd.DataFrame(columns=["date", "effr"])
        return df.rename(columns={"value": "effr"})

    # ------------------------------------------------------------------
    # 12. 3 个月 SOFR 期货 / 90天平均SOFR（离岸美元利率代理）
    # ------------------------------------------------------------------
    # 本地 SR3 期货历史数据 CSV（investing.com 导出），优先于网络接口
    _SR3_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "THREE-MONTH Sofr Futures Historical Data.csv"

    def get_sofr_futures(self, start_date: str = "20200101") -> pd.DataFrame:
        """优先本地 CSV（investing.com 导出的 CME 3M SOFR 期货主连）；
        失败时降级 yfinance SR3=F；再失败降级 FRED SOFR90DAYAVG。

        SR3 期货报价 = 100 - 隐含利率，故 sr3_close = 期货价格（业务层再算隐含利率）。
        降级路径下，sr3_close = 100 - SOFR90DAYAVG，使公式一致。
        """
        # --- 主路径：本地 CSV ---
        if self._SR3_CSV_PATH.exists():
            try:
                raw = pd.read_csv(self._SR3_CSV_PATH)
                df = pd.DataFrame({
                    "date": pd.to_datetime(raw["Date"], format="%m/%d/%Y"),
                    "sr3_close": pd.to_numeric(raw["Price"], errors="coerce"),
                })
                sd = pd.to_datetime(f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}")
                df = df[df["date"] >= sd]
                df = df.dropna(subset=["date", "sr3_close"]).sort_values(
                    "date").reset_index(drop=True)
                if not df.empty:
                    return df
            except Exception:
                pass
        # --- 降级1：yfinance SR3=F ---
        start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
        try:
            import yfinance as yf
            t = yf.Ticker("SR3=F")
            h = t.history(start=start, end=datetime.today().strftime("%Y-%m-%d"))
            if h is not None and not h.empty:
                df = h[["Close"]].copy()
                df.index.name = "date"
                df = df.reset_index()
                df["date"] = to_dt(df["date"])
                df["sr3_close"] = pd.to_numeric(df["Close"], errors="coerce")
                return df[["date", "sr3_close"]].dropna(
                    subset=["date", "sr3_close"]).sort_values("date").reset_index(drop=True)
        except Exception:
            pass
        # --- 降级2：FRED SOFR90DAYAVG ---
        df = self._fred_series("SOFR90DAYAVG", start_date)
        if df.empty:
            return pd.DataFrame(columns=["date", "sr3_close"])
        df["sr3_close"] = 100.0 - df["value"]
        return df[["date", "sr3_close"]].dropna(subset=["date", "sr3_close"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 13. 离岸人民币 3 个月 Hibor（CNH Hibor）
    # ------------------------------------------------------------------
    def get_cnh_hibor(self) -> pd.DataFrame:
        try:
            raw = ak.rate_interbank(market="香港银行同业拆借市场",
                                    symbol="Hibor人民币", indicator="3月")
            df = pd.DataFrame({
                "date": to_dt(raw["报告日"]),
                "cnh_hibor_3m": pd.to_numeric(raw["利率"], errors="coerce"),
            })
            return df.dropna(subset=["date", "cnh_hibor_3m"]).sort_values(
                "date").reset_index(drop=True)
        except Exception:
            return pd.DataFrame(columns=["date", "cnh_hibor_3m"])

    # ------------------------------------------------------------------
    # 14. 在岸人民币汇率 USD/CNY（央行中间价，日频长历史）
    # ------------------------------------------------------------------
    def get_usd_cny(self) -> pd.DataFrame:
        """在岸人民币汇率 USD/CNY。优先 FRED DEXCHUS（快），降级央行中间价。"""
        # 主路径：FRED DEXCHUS（单次 API 调用，历史长，速度快）
        df = self._fred_series("DEXCHUS", "20100101")
        if not df.empty:
            return df.rename(columns={"value": "usd_cny"})
        # 降级：中国银行央行中间价（分页较慢）
        try:
            raw = ak.currency_boc_sina(symbol="美元", start_date="20100101",
                                       end_date=datetime.today().strftime("%Y%m%d"))
            df = pd.DataFrame({
                "date": to_dt(raw["日期"]),
                "usd_cny": pd.to_numeric(raw["央行中间价"], errors="coerce"),
            })
            return df.dropna(subset=["date", "usd_cny"]).sort_values(
                "date").reset_index(drop=True)
        except Exception:
            return pd.DataFrame(columns=["date", "usd_cny"])

    # ------------------------------------------------------------------
    # 15. 离岸人民币汇率 USD/CNH
    # ------------------------------------------------------------------
    def get_usd_cnh(self) -> pd.DataFrame:
        """离岸人民币汇率 USD/CNH。主路径东方财富 push2his（长历史），
        降级新浪实时（仅最新一日）。

        注：原主路径 yfinance CNH=X 因 Yahoo 服务端 429 限流已不可靠，
        改用东方财富直连（secid=133.USDCNH，日频，2010-08 至今约 4200 个交易日）。
        """
        # --- 主路径：东方财富 push2his 日 K（secid=133.USDCNH）---
        try:
            url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
            params = {
                "secid": "133.USDCNH",
                "klt": "101",          # 日 K
                "fqt": "1",            # 不复权
                "lmt": "50000",        # 拉取全部
                "end": "20500000",
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
            }
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                data = r.json().get("data") or {}
                klines = data.get("klines") or []
                if klines:
                    # 每行: date,open,close,high,low,volume,amount,...
                    rows = [k.split(",") for k in klines]
                    df = pd.DataFrame({
                        "date": to_dt([r[0] for r in rows]),
                        "usd_cnh": pd.to_numeric([r[2] for r in rows], errors="coerce"),
                    })
                    df = df.dropna(subset=["date", "usd_cnh"])
                    if not df.empty:
                        return df.sort_values("date").reset_index(drop=True)
        except Exception:
            pass
        # --- 降级 1：akshare forex_hist_em（同上接口，若直连异常可兜底）---
        try:
            raw = ak.forex_hist_em(symbol="USDCNH")
            if raw is not None and not raw.empty:
                df = pd.DataFrame({
                    "date": to_dt(raw["日期"]),
                    "usd_cnh": pd.to_numeric(raw["收盘"], errors="coerce"),
                })
                df = df.dropna(subset=["date", "usd_cnh"])
                if not df.empty:
                    return df.sort_values("date").reset_index(drop=True)
        except Exception:
            pass
        # --- 降级 2：新浪财经实时报价（仅最新一日）---
        try:
            r = requests.get("https://hq.sinajs.cn/list=fx_susdcnh",
                             headers={**HEADERS, "Referer": "https://finance.sina.com.cn"},
                             timeout=10)
            text = r.text
            # var hq_str_fx_susdcnh="time,open,prev_close, ... ,date";
            parts = text.split('"')[1].split(",")
            if len(parts) >= 9 and parts[-1]:
                price = float(parts[8])
                d = pd.to_datetime(parts[-1])
                return pd.DataFrame({"date": [d.normalize()], "usd_cnh": [price]})
        except Exception:
            pass
        return pd.DataFrame(columns=["date", "usd_cnh"])

    # ------------------------------------------------------------------
    # 16. 中国国债收益率曲线（2Y + 10Y）
    # ------------------------------------------------------------------
    def get_bond_yield_curve(self, start_date: str = "20100101") -> pd.DataFrame:
        raw = ak.bond_zh_us_rate(start_date=start_date)
        df = pd.DataFrame({
            "date": to_dt(raw["日期"]),
            "bond_2y": pd.to_numeric(raw.get("中国国债收益率2年"), errors="coerce"),
            "bond_10y": pd.to_numeric(raw.get("中国国债收益率10年"), errors="coerce"),
        })
        return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # FRED 辅助：带 realtime 的序列（用于还原宏观数据实际公布日）
    # ------------------------------------------------------------------
    def _fred_series_realtime(self, series_id: str,
                              start_date: str = "20200101") -> pd.DataFrame:
        """拉取 FRED 单个序列并保留 realtime_start（数据首次公布日）。
        返回列 [date, value, release_date]。失败返回空 DataFrame。
        """
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self._fred_key(),
            "file_type": "json",
            "observation_start": f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}",
            "sort_order": "asc",
            "realtime_start": "1776-07-04",  # 取每个观测值首次公布的 realtime
            "realtime_end": "9999-12-31",
        }
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                return pd.DataFrame()
            data = resp.json()
            rows = data.get("observations", [])
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            df["date"] = to_dt(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df["release_date"] = to_dt(df["realtime_start"])
            return df[["date", "value", "release_date"]].dropna(
                subset=["date", "value", "release_date"]).sort_values("date").reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # 17. 美国宏观事件日历（FOMC 利率决议 / CPI / PPI 公布日）
    # ------------------------------------------------------------------
    # FOMC 议息会议决议公布日（两日会议的第二日），含临时会议。
    # 数据来源：美联储公开会议日程 https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
    FOMC_MEETING_DATES = [
        # 2020
        "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
        "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
        # 2021
        "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
        "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
        # 2022
        "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
        "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
        # 2023
        "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
        "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
        # 2024
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
        "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
        # 2025
        "2025-01-29", "2025-03-19", "2025-04-30", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-11-06", "2025-12-10",
        # 2026
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
        "2026-07-29", "2026-09-16", "2026-11-05", "2026-12-16",
    ]

    def get_us_macro_events(self, start_date: str = "20200101") -> pd.DataFrame:
        """汇总美联储相关重大事件的实际发生日与数据。

        事件类型:
          - FOMC  : 全部议息会议决议日，按结果分 加息/降息/持平
          - CPI   : 美国 CPI 公布日（CPIAUCSL 的 realtime_start）
          - PPI   : 美国 PPI 公布日（PPIACO 的 realtime_start）

        返回列: [event_date, event_type, event_name, value, prev_value,
                 change, surprise_hint, detail]
        """
        events: list[dict] = []

        # ---- FOMC 利率决议：固定会议日程 + DFEDTARU 判定加息/降息/持平 ----
        dfu = self._fred_series("DFEDTARU", start_date)
        dfl = self._fred_series("DFEDTARL", start_date)
        sd = pd.to_datetime(f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}")
        if not dfu.empty:
            dfu = dfu.sort_values("date").reset_index(drop=True)
            for mdate_str in self.FOMC_MEETING_DATES:
                mdate = pd.to_datetime(mdate_str)
                if mdate < sd:
                    continue
                # 取会议日前 5 天的利率作为"会前"，会议日后 5 天作为"会后"
                before = dfu[dfu["date"] <= mdate]
                after = dfu[dfu["date"] >= mdate]
                if before.empty or after.empty:
                    continue
                v_before = before.iloc[-1]["value"]
                v_after = after.iloc[0]["value"]
                # 若会后仍等于会前，再向后多看几天（利率可能次日生效）
                if v_after == v_before:
                    later = dfu[dfu["date"] > mdate]
                    if not later.empty:
                        v_after = later.iloc[0]["value"]
                chg = v_after - v_before
                lo_before_row = dfl[dfl["date"] <= mdate]
                lo_after_row = dfl[dfl["date"] >= mdate]
                lo_v = lo_after_row.iloc[0]["value"] if not lo_after_row.empty else (v_after - 0.25)
                if chg > 0:
                    name, hint = "FOMC 加息", "鹰派"
                elif chg < 0:
                    name, hint = "FOMC 降息", "鸽派"
                else:
                    name, hint = "FOMC 持平", "中性"
                detail = (f"目标区间 {lo_v:.2f}%-{v_after:.2f}%")
                if chg != 0:
                    detail += f"，变动 {chg:+.2f}%（{chg*100:+.0f}bp）"
                else:
                    detail += "，按兵不动"
                events.append({
                    "event_date": mdate,
                    "event_type": "FOMC",
                    "event_name": name,
                    "value": v_after,
                    "prev_value": v_before,
                    "change": chg,
                    "surprise_hint": hint,
                    "detail": detail,
                })

        # ---- CPI 公布日：CPIAUCSL 的 realtime_start ----
        cpi = self._fred_series_realtime("CPIAUCSL", start_date)
        if not cpi.empty:
            # 同一 release_date 可能对应多个历史修订，取该 release_date 下最新的观测
            cpi = cpi.sort_values(["release_date", "date"]).drop_duplicates(
                subset=["release_date"], keep="last")
            cpi = cpi.sort_values("release_date").reset_index(drop=True)
            prev_v = None
            for _, r in cpi.iterrows():
                v = r["value"]
                chg = (v - prev_v) / prev_v * 100.0 if prev_v else None
                # 同比：与 12 个月前比较
                yoy_series = cpi[cpi["date"] <= r["date"] - pd.Timedelta(days=365)]
                yoy = None
                if not yoy_series.empty:
                    yoy_v = yoy_series.iloc[-1]["value"]
                    yoy = (v - yoy_v) / yoy_v * 100.0
                hint = ""
                if chg is not None:
                    if chg > 0.4:
                        hint = "环比偏强"
                    elif chg < 0:
                        hint = "环比回落"
                    else:
                        hint = "环比温和"
                detail = f"CPI 指数 {v:.1f}"
                if yoy is not None:
                    detail += f"，同比 {yoy:+.2f}%"
                if chg is not None:
                    detail += f"，环比 {chg:+.2f}%"
                events.append({
                    "event_date": r["release_date"],
                    "event_type": "CPI",
                    "event_name": "美国 CPI 公布",
                    "value": v,
                    "prev_value": prev_v,
                    "change": chg,
                    "surprise_hint": hint,
                    "detail": detail,
                })
                prev_v = v

        # ---- PPI 公布日：PPIACO 的 realtime_start ----
        ppi = self._fred_series_realtime("PPIACO", start_date)
        if not ppi.empty:
            ppi = ppi.sort_values(["release_date", "date"]).drop_duplicates(
                subset=["release_date"], keep="last")
            ppi = ppi.sort_values("release_date").reset_index(drop=True)
            prev_v = None
            for _, r in ppi.iterrows():
                v = r["value"]
                chg = (v - prev_v) / prev_v * 100.0 if prev_v else None
                yoy_series = ppi[ppi["date"] <= r["date"] - pd.Timedelta(days=365)]
                yoy = None
                if not yoy_series.empty:
                    yoy_v = yoy_series.iloc[-1]["value"]
                    yoy = (v - yoy_v) / yoy_v * 100.0
                hint = ""
                if chg is not None:
                    if chg > 0.3:
                        hint = "环比偏强"
                    elif chg < 0:
                        hint = "环比回落"
                    else:
                        hint = "环比温和"
                detail = f"PPI 指数 {v:.1f}"
                if yoy is not None:
                    detail += f"，同比 {yoy:+.2f}%"
                if chg is not None:
                    detail += f"，环比 {chg:+.2f}%"
                events.append({
                    "event_date": r["release_date"],
                    "event_type": "PPI",
                    "event_name": "美国 PPI 公布",
                    "value": v,
                    "prev_value": prev_v,
                    "change": chg,
                    "surprise_hint": hint,
                    "detail": detail,
                })
                prev_v = v

        if not events:
            return pd.DataFrame(columns=["event_date", "event_type", "event_name",
                                         "value", "prev_value", "change",
                                         "surprise_hint", "detail"])
        df = pd.DataFrame(events)
        df["event_date"] = pd.to_datetime(df["event_date"])
        return df.sort_values("event_date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 18. 美国国债收益率曲线（多期限，FRED DGS* 系列）
    # ------------------------------------------------------------------
    # FRED 期限代码 -> 显示名
    US_TREASURY_TENORS = {
        "DGS1MO": "1M", "DGS3MO": "3M", "DGS6MO": "6M",
        "DGS1": "1Y", "DGS2": "2Y", "DGS5": "5Y",
        "DGS7": "7Y", "DGS10": "10Y", "DGS20": "20Y", "DGS30": "30Y",
    }

    def get_us_treasury_yields(self, start_date: str = "20200101") -> pd.DataFrame:
        """拉取美国国债各期限收益率，返回长表 [date, tenor, yield_pct]。"""
        frames = []
        for series_id, tenor in self.US_TREASURY_TENORS.items():
            df = self._fred_series(series_id, start_date)
            if not df.empty:
                df = df.rename(columns={"value": "yield_pct"})
                df["tenor"] = tenor
                frames.append(df[["date", "tenor", "yield_pct"]])
        if not frames:
            return pd.DataFrame(columns=["date", "tenor", "yield_pct"])
        out = pd.concat(frames, ignore_index=True)
        return out.sort_values(["date", "tenor"]).reset_index(drop=True)
