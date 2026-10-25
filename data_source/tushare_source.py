# -*- coding: utf-8 -*-
"""
tushare_source.py — Tushare Pro 数据源适配器

把 tushare Pro 的字段体系翻译成项目统一契约列名。
注意：部分接口有积分门槛（index_dailybasic 需 5000 分、bond/shibor/fut 需 2000 分），
若积分不足会抛异常，由 factory 自动降级到 akshare。

设计模式：适配器模式（Adapter）。
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from .protocol import DataSource
from .utils import iter_recent_days, to_dt

# 乐咕中文名 -> tushare ts_code
PE_INDEX_TS = {
    "沪深300": "000300.SH",
    "上证50": "000016.SH",
    "中证500": "000905.SH",
    "创业板指": "399006.SZ",
    "科创50": "000688.SH",
    "上证指数": "000001.SH",
}

# 中金所品种 -> tushare 交易所后缀 + 现货 ts_code
FUTURES_VARIETY = {
    "IF": ("CFX", "000300.SH", "沪深300"),
    "IH": ("CFX", "000016.SH", "上证50"),
    "IC": ("CFX", "000905.SH", "中证500"),
    "IM": ("CFX", "000852.SH", "中证1000"),
}

# ------------------------------------------------------------------
# 数据源渠道映射：方法名 -> 二级渠道说明（一级固定为 "tushare"）
# 用于 UI 展示每个数据项的来源与频率
# ------------------------------------------------------------------
SOURCE_CHANNELS = {
    "get_index_pe_monthly": "Tushare Pro index_dailybasic（中证指数），日频聚合为月频",
    "get_index_pe_daily": "Tushare Pro index_dailybasic（中证指数），日频",
    "get_index_pe_daily_csindex": "Tushare Pro index_dailybasic（中证指数），日频",
    "get_index_daily": "Tushare Pro index_daily，日频",
    "get_shibor": "Tushare Pro shibor（全国银行间同业拆借中心），日频",
    "get_option_pcr_history": "Tushare Pro opt_basic + opt_daily（上交所/深交所期权，逐合约按标的+到期月聚合），日频",
    "get_index_futures_basis_history": "Tushare Pro fut_daily（中金所）+ index_daily，日频",
    "get_citic_position_history": "Tushare Pro fut_holding（中金所前 20 会员），日频",
    "get_gold_curve_latest": "Tushare Pro fut_daily（上期所），日频",
    "get_gold_main_daily": "Tushare Pro fut_mapping + fut_daily，日频",
}


class TushareSource(DataSource):
    name = "tushare"

    def __init__(self, token: str, api_url: str | None = None):
        """
        token    : tushare token
        api_url  : 自定义 API 地址（代理服务）。为 None 时使用 tushare 官方地址。
                   例如 "http://api.quicksync.cn" 走第三方代理。
        """
        import tushare as ts  # 延迟导入，避免未安装时影响 akshare 路径
        # 关键：必须在 pro_api() 实例化之前改写类属性 http_url，
        # 否则 DataApi 会用官方地址构造请求
        if api_url:
            import tushare.pro.client as _client
            _client.DataApi._DataApi__http_url = api_url
        ts.set_token(token)
        self.pro = ts.pro_api()
        self._token = token
        self._api_url = api_url or "https://api.tushare.pro"  # 仅用于日志展示

    # ------------------------------------------------------------------
    # 工具：sina_code (sh000300) -> tushare ts_code (000300.SH)
    # ------------------------------------------------------------------
    @staticmethod
    def _sina_to_ts(sina_code: str) -> str:
        code = sina_code.lower()
        if code.startswith("sh"):
            return code[2:] + ".SH"
        if code.startswith("sz"):
            return code[2:] + ".SZ"
        return code

    # ------------------------------------------------------------------
    # 1. 10 年期国债收益率
    #    实测：quicksync 代理不支持 bond_china_yield / bond_zh_hs_daily 等债券接口，
    #    显式抛出 NotImplementedError 让工厂直接降级到 akshare，避免无谓等待。
    # ------------------------------------------------------------------
    def get_bond_10y(self, start_date: str = "20100101") -> pd.DataFrame:
        raise NotImplementedError("当前 tushare 代理不支持国债收益率接口，走 akshare")

    # ------------------------------------------------------------------
    # 2. 指数市盈率（月频）：从日频 index_dailybasic 取月末值
    # ------------------------------------------------------------------
    def get_index_pe_monthly(self, symbol: str = "沪深300") -> pd.DataFrame:
        ts_code = PE_INDEX_TS.get(symbol, "000300.SH")
        raw = self.pro.index_dailybasic(ts_code=ts_code)
        if raw is None or len(raw) == 0:
            return pd.DataFrame(columns=["date", "pe_ttm"])
        df = raw[["trade_date", "pe_ttm"]].copy()
        df.columns = ["date", "pe_ttm"]
        df["date"] = to_dt(df["date"])
        df["pe_ttm"] = pd.to_numeric(df["pe_ttm"], errors="coerce")
        df = df.dropna().sort_values("date")
        # 取每月最后一个交易日作为月频样本
        df = df.set_index("date").resample("ME").last().dropna().reset_index()
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2b. 指数市盈率（日频，全历史）：直接取 index_dailybasic 的每日 pe_ttm
    # ------------------------------------------------------------------
    def get_index_pe_daily(self, symbol: str = "沪深300") -> pd.DataFrame:
        ts_code = PE_INDEX_TS.get(symbol, "000300.SH")
        raw = self.pro.index_dailybasic(ts_code=ts_code)
        if raw is None or len(raw) == 0:
            return pd.DataFrame(columns=["date", "pe_ttm"])
        df = raw[["trade_date", "pe_ttm"]].copy()
        df.columns = ["date", "pe_ttm"]
        df["date"] = to_dt(df["date"])
        df["pe_ttm"] = pd.to_numeric(df["pe_ttm"], errors="coerce")
        return df.dropna().sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. 指数市盈率（日频）
    #    实测：index_dailybasic 返回 pe / pe_ttm / pb，部分代理不返回 dv_ratio（股息率）
    # ------------------------------------------------------------------
    def get_index_pe_daily_csindex(self, csindex_code: str = "000300") -> pd.DataFrame:
        ts_code = csindex_code + (".SH" if csindex_code.startswith("0") else ".SZ")
        raw = self.pro.index_dailybasic(ts_code=ts_code)
        if raw is None or len(raw) == 0:
            return pd.DataFrame(columns=["date", "pe_static", "pe_ttm", "div_yield"])
        # 取 pe(静态) / pe_ttm；dv_ratio(股息率) 可能不存在，用 NaN 填充
        div = raw["dv_ratio"] if "dv_ratio" in raw.columns else pd.NA
        df = pd.DataFrame({
            "date": to_dt(raw["trade_date"]),
            "pe_static": pd.to_numeric(raw["pe"], errors="coerce"),
            "pe_ttm": pd.to_numeric(raw["pe_ttm"], errors="coerce"),
            "div_yield": pd.to_numeric(div, errors="coerce"),
        })
        return df.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 4. 大盘指数日线
    # ------------------------------------------------------------------
    def get_index_daily(self, sina_code: str = "sh000300") -> pd.DataFrame:
        ts_code = self._sina_to_ts(sina_code)
        raw = self.pro.index_daily(ts_code=ts_code)
        if raw is None or len(raw) == 0:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        df = raw.rename(columns={"trade_date": "date", "vol": "volume"})
        df["date"] = to_dt(df["date"])
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df[["date", "open", "high", "low", "close", "volume"]].dropna(
            subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 5. SHIBOR
    # ------------------------------------------------------------------
    def get_shibor(self) -> pd.DataFrame:
        # tushare shibor 接口按日查询；这里取近 3 年逐日查询（约 750 次，受频控影响可能较慢）
        # 若频控过严会抛异常，由 factory 降级到 akshare
        tenors = ["隔夜", "1周", "2周", "1个月", "3个月", "6个月", "9个月", "1年"]
        records = []
        start = date.today() - timedelta(days=365 * 3)
        for d in iter_recent_days(365 * 3):
            if d < start:
                break
            try:
                raw = self.pro.shibor(date=d.strftime("%Y%m%d"))
            except Exception:
                continue
            if raw is None or len(raw) == 0:
                continue
            row = {"date": pd.Timestamp(d)}
            # tushare shibor 字段: one_week, two_week, one_month, three_month,
            # six_month, nine_month, one_year, overnight
            col_map = {
                "隔夜": "overnight", "1周": "one_week", "2周": "two_week",
                "1个月": "one_month", "3个月": "three_month", "6个月": "six_month",
                "9个月": "nine_month", "1年": "one_year",
            }
            for cn, en in col_map.items():
                row[cn] = pd.to_numeric(raw[en].iloc[0], errors="coerce") if en in raw.columns else None
            records.append(row)
        if not records:
            return pd.DataFrame(columns=["date"] + tenors)
        df = pd.DataFrame(records)
        return df.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 6. 期权 PCR：opt_basic 建立合约->(标的, C/P) 映射，opt_daily 逐日全量拉取后聚合
    #    积分门槛：opt_basic 需 5000 分，opt_daily 需 2000 分；不足则 factory 降级到 akshare
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_underlying(name: str) -> str | None:
        """把合约名称里的标的名标准化为统一口径，便于和 akshare 结果对齐。"""
        if not name:
            return None
        n = str(name)
        if "50ETF" in n or ("上证50" in n and "300" not in n and "500" not in n):
            return "50ETF"
        if "300ETF" in n or "沪深300" in n:
            return "300ETF"
        if "500ETF" in n or "中证500" in n:
            return "500ETF"
        if "创业板" in n:
            return "创业板ETF"
        if "科创50" in n or "科创" in n:
            return "科创50ETF"
        return n

    def get_option_pcr_history(self, lookback_calendar_days: int = 45,
                               _progress_callback=None) -> pd.DataFrame:
        # 1. 建立 ts_code -> (underlying, call_put, exp_month) 映射（SSE + SZSE）
        basic_frames = []
        for ex in ("SSE", "SZSE"):
            try:
                b = self.pro.opt_basic(exchange=ex,
                                       fields="ts_code,name,call_put,delist_date")
                if b is not None and len(b) > 0:
                    basic_frames.append(b)
            except Exception:
                continue
        if not basic_frames:
            raise RuntimeError("opt_basic 拉取失败（可能积分不足 5000）")
        basic = pd.concat(basic_frames, ignore_index=True)
        # 从 name 提取标的：如 "50ETF购10月2700" -> "50ETF"
        basic["underlying"] = (basic["name"].str.extract(r"^(.+?)[购沽]", expand=False)
                               .map(self._normalize_underlying))
        # 从 delist_date 提取到期月份 YYMM（如 20261028 -> "2610"）
        basic["exp_month"] = basic["delist_date"].astype(str).str[2:6]
        code_map = basic.set_index("ts_code")[
            ["underlying", "call_put", "exp_month"]].to_dict("index")

        # 2. 逐日拉 opt_daily（单日返回全市场全部合约），按 (标的, 到期月) 聚合 C/P
        days = list(iter_recent_days(lookback_calendar_days))
        total = len(days)
        frames = []
        for i, d in enumerate(days):
            try:
                raw = self.pro.opt_daily(trade_date=d.strftime("%Y%m%d"))
            except Exception:
                if _progress_callback:
                    _progress_callback(i + 1, total)
                continue
            if raw is None or len(raw) == 0:
                if _progress_callback:
                    _progress_callback(i + 1, total)
                continue
            raw = raw.copy()
            raw["underlying"] = raw["ts_code"].map(lambda c: code_map.get(c, {}).get("underlying"))
            raw["call_put"] = raw["ts_code"].map(lambda c: code_map.get(c, {}).get("call_put"))
            raw["exp_month"] = raw["ts_code"].map(lambda c: code_map.get(c, {}).get("exp_month"))
            raw = raw.dropna(subset=["underlying", "call_put"])
            if raw.empty:
                if _progress_callback:
                    _progress_callback(i + 1, total)
                continue
            raw["vol"] = pd.to_numeric(raw["vol"], errors="coerce")
            raw["oi"] = pd.to_numeric(raw["oi"], errors="coerce")
            ts = pd.Timestamp(d)
            grp_keys = ["underlying", "exp_month"]
            call = raw[raw["call_put"] == "C"].groupby(grp_keys, as_index=False).agg(
                call_volume=("vol", "sum"), call_oi=("oi", "sum"))
            put = raw[raw["call_put"] == "P"].groupby(grp_keys, as_index=False).agg(
                put_volume=("vol", "sum"), put_oi=("oi", "sum"))
            day_df = call.merge(put, on=grp_keys, how="outer").fillna(0)
            day_df["date"] = ts
            day_df["pcr_volume"] = day_df["put_volume"] / day_df["call_volume"].mask(day_df["call_volume"] == 0)
            day_df["pcr_oi"] = day_df["put_oi"] / day_df["call_oi"].mask(day_df["call_oi"] == 0)
            frames.append(day_df)
            if _progress_callback:
                _progress_callback(i + 1, total)

        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        return df.sort_values(["date", "underlying", "exp_month"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 7. 股指期货升贴水：用 fut_daily 拉全合约 + index_daily 算现货基差
    # ------------------------------------------------------------------
    def get_index_futures_basis_history(self, lookback_calendar_days: int = 45) -> pd.DataFrame:
        end = datetime.today().strftime("%Y%m%d")
        start = (date.today() - timedelta(days=lookback_calendar_days)).strftime("%Y%m%d")
        # 现货指数日线一次性拉取
        spot = {}
        for variety, (_ex, spot_ts, _name) in FUTURES_VARIETY.items():
            s = self.pro.index_daily(ts_code=spot_ts, start_date=start, end_date=end)
            if s is not None and len(s) > 0:
                s = s.rename(columns={"trade_date": "date", "close": "spot_close"})
                s["date"] = to_dt(s["date"])
                spot[variety] = s[["date", "spot_close"]]
        rows = []
        for variety, (ex, _spot_ts, name) in FUTURES_VARIETY.items():
            # 取该品种所有合约日线
            raw = self.pro.fut_daily(ts_code=f"{variety}.{ex}", start_date=start, end_date=end)
            if raw is None or len(raw) == 0:
                continue
            raw = raw.copy()
            raw["date"] = to_dt(raw["trade_date"])
            for d_str, g in raw.groupby("date"):
                main = g.sort_values("vol", ascending=False).iloc[0]
                fut_close = float(main["close"])
                ts = pd.Timestamp(d_str)
                sp = spot.get(variety)
                if sp is None:
                    continue
                spot_row = sp.loc[sp["date"] == ts]
                if spot_row.empty:
                    spot_row = sp.loc[sp["date"] <= ts].tail(1)
                if spot_row.empty:
                    continue
                spot_close = float(spot_row["spot_close"].iloc[0])
                rows.append({
                    "date": ts, "variety": variety, "variety_name": name,
                    "contract": str(main["ts_code"]), "futures_close": fut_close,
                    "spot_close": spot_close, "basis": fut_close - spot_close,
                    "basis_pct": (fut_close - spot_close) / spot_close * 100,
                })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(["date", "variety"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 7b. 中信期货多空持仓：fut_holding 取前 20 会员排名，筛选中信期货
    # ------------------------------------------------------------------
    def get_citic_position_history(self, lookback_calendar_days: int = 45,
                                   vars_list=None) -> pd.DataFrame:
        # 实测：quicksync 代理的 fut_holding 不支持 CFFEX 股指期货（返回商品期货数据，
        # 且忽略 ts_code 参数），无法拿到 IF/IH/IC/IM 的持仓排名，降级到 akshare。
        raise NotImplementedError("当前代理 fut_holding 不支持 CFFEX 股指期货持仓，走 akshare")

    # ------------------------------------------------------------------
    # 6b. 期权券商持仓排名：tushare 无期权机构持仓接口，走 akshare（上交所）
    # ------------------------------------------------------------------
    def get_option_broker_holding_history(self, lookback_calendar_days: int = 45,
                                          _progress_callback=None) -> pd.DataFrame:
        raise NotImplementedError("tushare 无期权券商持仓接口，走 akshare")

    # ------------------------------------------------------------------
    # 8. 黄金期货期限结构：取最近交易日所有 AU 合约
    # ------------------------------------------------------------------
    def get_gold_curve_latest(self) -> pd.DataFrame:
        # 用 fut_basic 取上期所所有 AU 合约（交易所代码为 SHFE），再取最新日日线
        basic = self.pro.fut_basic(exchange="SHFE", fut_type="1")
        au_contracts = basic[basic["ts_code"].astype(str).str.startswith("AU")]["ts_code"].tolist()
        if not au_contracts:
            return pd.DataFrame()
        # 取近 10 个交易日的合约日线
        end = datetime.today().strftime("%Y%m%d")
        start = (date.today() - timedelta(days=20)).strftime("%Y%m%d")
        all_data = []
        for c in au_contracts:
            try:
                d = self.pro.fut_daily(ts_code=c, start_date=start, end_date=end)
            except Exception:
                continue
            if d is not None and len(d) > 0:
                all_data.append(d)
        if not all_data:
            return pd.DataFrame()
        df = pd.concat(all_data, ignore_index=True)
        df["date"] = to_dt(df["trade_date"])
        latest = df["date"].max()
        au = df[df["date"] == latest].copy()
        au = au.rename(columns={"ts_code": "contract", "settle": "settle",
                                "vol": "volume", "oi": "open_interest"})
        for c in ["close", "settle", "volume", "open_interest"]:
            au[c] = pd.to_numeric(au[c], errors="coerce")
        au = au.dropna(subset=["close"])
        if au.empty:
            return pd.DataFrame()
        # AU yymm -> 到期月份 15 日
        au["expire_date"] = au["contract"].map(
            lambda s: pd.Timestamp(year=2000 + int(s[2:4]), month=int(s[4:6]), day=15))
        return au[["date", "contract", "expire_date", "close", "settle",
                   "volume", "open_interest"]].sort_values("expire_date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 9. 黄金主力连续日线
    # ------------------------------------------------------------------
    def get_gold_main_daily(self, start_date: str = "20180101",
                            end_date: str | None = None) -> pd.DataFrame:
        end_date = end_date or datetime.today().strftime("%Y%m%d")
        # tushare 无直接主力连续，用 fut_mapping 确定每日主力合约再拼接
        mapping = self.pro.fut_mapping(ts_code="AU.SHF", start_date=start_date, end_date=end_date)
        if mapping is None or len(mapping) == 0:
            # 降级：直接拿 AU0 连续合约（若接口支持）
            raw = self.pro.fut_daily(ts_code="AU0.SHF", start_date=start_date, end_date=end_date)
            if raw is None or len(raw) == 0:
                return pd.DataFrame(columns=["date", "close", "volume", "open_interest"])
            df = raw.rename(columns={"trade_date": "date", "vol": "volume", "oi": "open_interest"})
            df["date"] = to_dt(df["date"])
            return df[["date", "close", "volume", "open_interest"]].dropna(
                subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        # 按每日主力合约拼接
        frames = []
        for ts_code, g in mapping.groupby("ts_code"):
            s = g["trade_date"].min()
            e = g["trade_date"].max()
            try:
                d = self.pro.fut_daily(ts_code=ts_code, start_date=s, end_date=e)
            except Exception:
                continue
            if d is not None and len(d) > 0:
                frames.append(d)
        if not frames:
            return pd.DataFrame(columns=["date", "close", "volume", "open_interest"])
        df = pd.concat(frames, ignore_index=True)
        df = df.rename(columns={"trade_date": "date", "vol": "volume", "oi": "open_interest"})
        df["date"] = to_dt(df["date"])
        df = df.drop_duplicates("date").sort_values("date")
        return df[["date", "close", "volume", "open_interest"]].dropna(
            subset=["date", "close"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 10. CFTC COT：tushare 不提供，降级到 akshare
    # ------------------------------------------------------------------
    def get_cot_gold(self, years: int = 3) -> pd.DataFrame:
        raise NotImplementedError("Tushare 不提供 CFTC COT 数据，走 cftc.gov 直连（akshare 适配器）")

    # ------------------------------------------------------------------
    # 11-16. 离岸在岸息差系统：tushare 无对应接口，全部降级到 akshare
    # ------------------------------------------------------------------
    def get_effr(self, start_date: str = "20200101") -> pd.DataFrame:
        raise NotImplementedError("Tushare 无 EFFR 接口，走 FRED（akshare 适配器）")

    def get_sofr_futures(self, start_date: str = "20200101") -> pd.DataFrame:
        raise NotImplementedError("Tushare 无 SOFR 期货接口，走 yfinance/FRED（akshare 适配器）")

    def get_cnh_hibor(self) -> pd.DataFrame:
        raise NotImplementedError("Tushare 无 CNH Hibor 接口，走东方财富（akshare 适配器）")

    def get_usd_cny(self) -> pd.DataFrame:
        raise NotImplementedError("Tushare 无 USD/CNY 汇率接口，走央行中间价（akshare 适配器）")

    def get_usd_cnh(self) -> pd.DataFrame:
        raise NotImplementedError("Tushare 无 USD/CNH 汇率接口，走 yfinance/新浪（akshare 适配器）")

    def get_bond_yield_curve(self, start_date: str = "20100101") -> pd.DataFrame:
        raise NotImplementedError("Tushare 无国债收益率曲线接口，走中国国债信息网（akshare 适配器）")
