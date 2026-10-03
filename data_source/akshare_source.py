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
import zipfile
from datetime import date, datetime

import pandas as pd
import requests

import akshare as ak

from .protocol import DataSource
from .utils import iter_recent_days, to_dt

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
                               _progress_callback=None) -> pd.DataFrame:
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
                                          _progress_callback=None) -> pd.DataFrame:
        """上交所期权持仓排名（前 3 大标的 × 认购/认沽前 5 会员）。

        返回列: date, underlying, broker, call_oi, put_oi, pcr_oi
        注意：仅持仓量（上交所未公开券商级交易量排名）；仅覆盖上交所前 3 大标的。
        """
        url = ("https://www.sse.org.cn/api/report/ShowReport"
               "?SHOWTYPE=xlsx&CATALOGID=qqccpmtj1_online&TABKEY=tab1"
               "&txtSearchDate={date}&jy=qqcc_online")
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
        df["pcr_oi"] = df["put_oi"] / df["call_oi"].replace(0, pd.NA)
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
                                   vars_list=None, _progress_callback=None) -> pd.DataFrame:
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
        for yr in range(this_year, this_year - years, -1):
            url = f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{yr}.zip"
            resp = requests.get(url, headers=HEADERS, timeout=120)
            if resp.status_code != 200:
                continue
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                inner = zf.namelist()[0]
                with zf.open(inner) as f:
                    text = f.read().decode("utf-8", errors="ignore")
            frames.append(self._parse_cot_zip_text(text))
        if not frames:
            return pd.DataFrame(columns=["date", "open_interest", "mm_long",
                                         "mm_short", "mm_spread", "mm_net"])
        out = pd.concat(frames, ignore_index=True)
        return out.dropna(subset=["date"]).drop_duplicates("date").sort_values(
            "date").reset_index(drop=True)
