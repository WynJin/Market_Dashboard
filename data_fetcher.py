# -*- coding: utf-8 -*-
"""
data_fetcher.py — 宏观交易看板数据抓取层

全部数据源均为公开免费渠道：
  1. 10 年期国债收益率   akshare.bond_zh_us_rate        （中国国债信息网，日频）
  2. 指数市盈率（长期）   akshare.stock_index_pe_lg      （乐咕乐股，月频，TTM 滚动市盈率）
  3. 指数市盈率（近期）   akshare.stock_zh_index_value_csindex（中证指数官网，日频，近约 20 个交易日）
  4. 大盘指数行情         akshare.stock_zh_index_daily   （新浪财经，日频）
  5. SHIBOR              akshare.macro_china_shibor_all （shibor.org，日频）
  6. 期权 PCR            akshare.option_daily_stats_sse/szse（上交所/深交所官方日统计，日频）
  7. 股指期货升贴水       akshare.get_cffex_daily        （中金所官方日行情，日频）
  7b.中信期货多空持仓     akshare.get_cffex_rank_table   （中金所前20会员持仓排名，日频）
  8. 黄金期货期限结构     akshare.get_shfe_daily         （上期所官方日行情，日频）
  9. 黄金主力连续         akshare.futures_main_sina      （新浪财经，日频）
 10. CFTC COT 持仓报告   cftc.gov 官方文件               （周频，当年 deafut.txt + 历年 zip）

设计说明：
  - 所有函数返回规范化 DataFrame，日期列统一为 datetime64、列名见各函数 docstring；
  - 抓取函数不吞异常，交由页面层决定降级展示；
  - 通过 streamlit.cache_data 缓存（无 streamlit 时自动退化为空装饰器），
    因此本模块也可被普通 Python 脚本 / 后续 FastAPI 服务直接复用。
"""
from __future__ import annotations

import io
import zipfile
import calendar
from datetime import date, datetime, timedelta

import requests
import pandas as pd
import akshare as ak

# ---------------------------------------------------------------------------
# 缓存装饰器：Streamlit 环境下自动缓存，命令行环境下退化为不缓存
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from streamlit import cache_data
except Exception:  # pragma: no cover
    def cache_data(ttl=None, **_kwargs):
        def deco(fn):
            return fn
        return deco

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
HEADERS = {"User-Agent": UA}


def _to_dt(s: pd.Series) -> pd.Series:
    """把各种格式的日期列统一转成 datetime64（不含时分秒）。"""
    return pd.to_datetime(s.astype(str), errors="coerce").dt.normalize()


# ---------------------------------------------------------------------------
# 交易日回溯工具：免费接口不提供交易日历，用“向前逐日试探”的方式找有效交易日
# ---------------------------------------------------------------------------
def iter_recent_days(n_max: int = 45):
    """从今天（不含）向前生成 n_max 个自然日，跳过周六周日。"""
    d = date.today() - timedelta(days=1)
    yielded = 0
    while yielded < n_max:
        if d.weekday() < 5:  # 0=周一 ... 4=周五
            yield d
            yielded += 1
        d -= timedelta(days=1)


# ---------------------------------------------------------------------------
# 1. 10 年期国债收益率
# ---------------------------------------------------------------------------
@cache_data(ttl=12 * 3600)
def get_bond_10y(start_date: str = "20100101") -> pd.DataFrame:
    """返回列: date(datetime), bond_yield(float, 单位 %)。"""
    raw = ak.bond_zh_us_rate(start_date=start_date)
    df = raw[["日期", "中国国债收益率10年"]].copy()
    df.columns = ["date", "bond_yield"]
    df["date"] = _to_dt(df["date"])
    df["bond_yield"] = pd.to_numeric(df["bond_yield"], errors="coerce")
    return df.dropna(subset=["date", "bond_yield"]).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. 指数市盈率（乐咕乐股，月频）
# ---------------------------------------------------------------------------
# 乐咕支持的常见宽基（接口实际可用范围以官网为准，失败时页面层会提示）
PE_INDEX_CHOICES = ["沪深300", "上证50", "中证500", "创业板指", "科创50", "上证指数"]


@cache_data(ttl=24 * 3600)
def get_index_pe_monthly(symbol: str = "沪深300") -> pd.DataFrame:
    """
    返回列: date(月末日期), pe_ttm(float)。
    取“滚动市盈率”（TTM）口径；乐咕免费版为月频（每月末一个点）。
    """
    raw = ak.stock_index_pe_lg(symbol=symbol)
    df = raw[["日期", "滚动市盈率"]].copy()
    df.columns = ["date", "pe_ttm"]
    df["date"] = _to_dt(df["date"])
    df["pe_ttm"] = pd.to_numeric(df["pe_ttm"], errors="coerce")
    return df.dropna(subset=["date", "pe_ttm"]).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. 指数市盈率（中证指数官网，日频，仅最近约 20 个交易日，用于近期精度补充）
# ---------------------------------------------------------------------------
@cache_data(ttl=12 * 3600)
def get_index_pe_daily_csindex(csindex_code: str = "000300") -> pd.DataFrame:
    """返回列: date, pe_static(市盈率1), pe_ttm(市盈率2, 滚动), div_yield。"""
    raw = ak.stock_zh_index_value_csindex(symbol=csindex_code)
    df = raw[["日期", "市盈率1", "市盈率2", "股息率1"]].copy()
    df.columns = ["date", "pe_static", "pe_ttm", "div_yield"]
    df["date"] = _to_dt(df["date"])
    for c in ["pe_static", "pe_ttm", "div_yield"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. 大盘指数日线（新浪）
# ---------------------------------------------------------------------------
@cache_data(ttl=12 * 3600)
def get_index_daily(sina_code: str = "sh000300") -> pd.DataFrame:
    """返回列: date, open, high, low, close, volume。sina_code 如 sh000300。"""
    raw = ak.stock_zh_index_daily(symbol=sina_code)
    df = raw.copy()
    df["date"] = _to_dt(df["date"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. SHIBOR（上海银行间同业拆放利率）
# ---------------------------------------------------------------------------
_SHIBOR_MAP = {
    "隔夜": "O/N-定价",
    "1周": "1W-定价",
    "2周": "2W-定价",
    "1个月": "1M-定价",
    "3个月": "3M-定价",
    "6个月": "6M-定价",
    "9个月": "9M-定价",
    "1年": "1Y-定价",
}


@cache_data(ttl=12 * 3600)
def get_shibor() -> pd.DataFrame:
    """返回列: date + 8 个期限（隔夜/1周/2周/1个月/3个月/6个月/9个月/1年），单位 %。"""
    raw = ak.macro_china_shibor_all()
    df = pd.DataFrame({"date": _to_dt(raw["日期"])})
    for cn, col in _SHIBOR_MAP.items():
        df[cn] = pd.to_numeric(raw[col], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. 期权 PCR（上交所 + 深交所官方日统计）
# ---------------------------------------------------------------------------
def _option_stats_one_day(d: date) -> pd.DataFrame | None:
    """取某天上交所+深交所 ETF 期权成交/持仓汇总；非交易日返回 None。"""
    ymd = d.strftime("%Y%m%d")
    rows = []
    # 上交所：认购/认沽成交量、未平仓数
    try:
        sse = ak.option_daily_stats_sse(date=ymd)
        if sse is not None and len(sse) > 0:
            for _, r in sse.iterrows():
                rows.append({
                    "date": pd.Timestamp(d),
                    "underlying": str(r["合约标的名称"]),
                    "call_volume": float(r["认购成交量"]),
                    "put_volume": float(r["认沽成交量"]),
                    "call_oi": float(r["未平仓认购合约数"]),
                    "put_oi": float(r["未平仓认沽合约数"]),
                })
    except Exception:
        pass
    # 深交所：字段口径相同（无总成交额列，不影响 PCR）
    try:
        szse = ak.option_daily_stats_szse(date=ymd)
        if szse is not None and len(szse) > 0:
            for _, r in szse.iterrows():
                rows.append({
                    "date": pd.Timestamp(d),
                    "underlying": str(r["合约标的名称"]),
                    "call_volume": float(r["认购成交量"]),
                    "put_volume": float(r["认沽成交量"]),
                    "call_oi": float(r["未平仓认购合约数"]),
                    "put_oi": float(r["未平仓认沽合约数"]),
                })
    except Exception:
        pass
    if not rows:
        return None
    out = pd.DataFrame(rows)
    # 同名标的（如沪深300ETF 在上交所/深交所各有一只）按成交量加权合并
    agg = out.groupby(["date", "underlying"], as_index=False).agg(
        call_volume=("call_volume", "sum"),
        put_volume=("put_volume", "sum"),
        call_oi=("call_oi", "sum"),
        put_oi=("put_oi", "sum"),
    )
    agg["pcr_volume"] = agg["put_volume"] / agg["call_volume"]
    agg["pcr_oi"] = agg["put_oi"] / agg["call_oi"]
    return agg


@cache_data(ttl=12 * 3600)
def get_option_pcr_history(lookback_calendar_days: int = 45) -> pd.DataFrame:
    """
    逐日回溯抓取期权 PCR 日度序列（交易所接口按天查询，无批量历史）。
    返回列: date, underlying, call_volume, put_volume, call_oi, put_oi, pcr_volume, pcr_oi。
    lookback_calendar_days 越大请求越多、越慢，默认 45 个自然日（约 30 个交易日）。
    """
    frames = []
    for d in iter_recent_days(lookback_calendar_days):
        one = _option_stats_one_day(d)
        if one is not None:
            frames.append(one)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["date", "underlying"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7. 股指期货升贴水（中金所日行情 + 现货指数）
# ---------------------------------------------------------------------------
# 中金所品种 -> (新浪现货代码, 中文名)
INDEX_FUTURES_MAP = {
    "IF": ("sh000300", "沪深300"),
    "IH": ("sh000016", "上证50"),
    "IC": ("sh000905", "中证500"),
    "IM": ("sh000852", "中证1000"),
}


def _cffex_one_day(d: date) -> pd.DataFrame | None:
    """取某天中金所全部合约日行情；非交易日返回 None。"""
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


@cache_data(ttl=12 * 3600)
def get_index_futures_basis_history(lookback_calendar_days: int = 45) -> pd.DataFrame:
    """
    返回列: date, variety(IF/IH/IC/IM), variety_name, contract(主力合约),
            futures_close, spot_close, basis(基差点), basis_pct(基差率%)。
    主力合约 = 当日该品种成交量最大的合约。
    基差 = 期货收盘 - 现货收盘；basis_pct = 基差 / 现货收盘 * 100。
    升水(期货>现货)为正，贴水为负。
    """
    # 现货一次性拉全历史，避免逐日请求
    spot = {}
    for variety, (sina_code, _name) in INDEX_FUTURES_MAP.items():
        spot[variety] = get_index_daily(sina_code)[["date", "close"]].rename(
            columns={"close": "spot_close"})

    rows = []
    for d in iter_recent_days(lookback_calendar_days):
        cff = _cffex_one_day(d)
        if cff is None:
            continue
        ts = pd.Timestamp(d)
        for variety, (_code, name) in INDEX_FUTURES_MAP.items():
            sub = cff[cff["variety"] == variety]
            if sub.empty:
                continue
            main = sub.sort_values("volume", ascending=False).iloc[0]  # 成交量最大=主力
            spot_row = spot[variety].loc[spot[variety]["date"] == ts]
            if spot_row.empty:  # 现货当日可能尚未更新，取不晚于该日的最后一个收盘
                spot_row = spot[variety].loc[spot[variety]["date"] <= ts].tail(1)
            if spot_row.empty:
                continue
            spot_close = float(spot_row["spot_close"].iloc[0])
            fut_close = float(main["close"])
            rows.append({
                "date": ts,
                "variety": variety,
                "variety_name": name,
                "contract": str(main["symbol"]),
                "futures_close": fut_close,
                "spot_close": spot_close,
                "basis": fut_close - spot_close,
                "basis_pct": (fut_close - spot_close) / spot_close * 100,
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["date", "variety"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7b. 中信期货多空持仓（中金所股指期货 IF/IH/IC/IM，前 20 会员持仓排名）
# ---------------------------------------------------------------------------
CITIC_KEYWORD = "中信期货"  # 注意区分“中信期货”与“中信建投”，二者是不同机构


def _extract_citic_one_day(d: date, vars_list: list[str] | None = None) -> pd.DataFrame | None:
    """
    取某交易日中信期货在中金所指定品种各合约上的多空持仓。
    返回列: date, variety, contract, long_oi, long_oi_chg, long_rank,
            short_oi, short_oi_chg, short_rank, net_oi。
    若中信未进入该合约前 20 名，则对应持仓按 0 计（实际小于第 20 名）。
    """
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
        df = df[df["rank"] != 999].copy()  # 排除合计行
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
            "date": ts,
            "variety": variety,
            "contract": str(contract),
            "long_oi": long_oi,
            "long_oi_chg": long_oi_chg,
            "long_rank": long_rank,
            "short_oi": short_oi,
            "short_oi_chg": short_oi_chg,
            "short_rank": short_rank,
            "net_oi": long_oi - short_oi,
        })
    if not rows:
        return None
    return pd.DataFrame(rows)


@cache_data(ttl=12 * 3600)
def get_citic_position_history(lookback_calendar_days: int = 45,
                               vars_list: list[str] | None = None) -> pd.DataFrame:
    """
    逐日回溯抓取中信期货在中金所股指期货（IF/IH/IC/IM）上的多空持仓排名。

    返回列: date, variety, contract, long_oi, long_oi_chg, long_rank,
            short_oi, short_oi_chg, short_rank, net_oi（单位：手）。
    net_oi = long_oi - short_oi，正值=净多，负值=净空。
    每个交易日每个合约一行；如需按品种汇总，调用方按 (date, variety) groupby sum。
    """
    frames = []
    for d in iter_recent_days(lookback_calendar_days):
        one = _extract_citic_one_day(d, vars_list=vars_list)
        if one is not None:
            frames.append(one)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["date", "variety", "contract"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 8. 黄金期货期限结构（上期所 AU 系列）
# ---------------------------------------------------------------------------
def _shfe_one_day(d: date) -> pd.DataFrame | None:
    try:
        df = ak.get_shfe_daily(date=d.strftime("%Y%m%d"))
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    return df


@cache_data(ttl=12 * 3600)
def get_gold_curve_latest() -> pd.DataFrame:
    """
    返回最近交易日上期所全部 AU 合约的期限结构:
    date, contract, expire_date(合约到期月, 每月15日近似), close, settle, volume, open_interest。
    """
    for d in iter_recent_days(20):
        df = _shfe_one_day(d)
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
        # AUyymm -> 到期月份的 15 日作为近似到期点（个人合约自然人不能进入交割月，仅用于排序画图）
        au["expire_date"] = au["contract"].map(
            lambda s: pd.Timestamp(year=2000 + int(s[2:4]), month=int(s[4:6]), day=15))
        return au[["date", "contract", "expire_date", "close", "settle",
                   "volume", "open_interest"]].sort_values("expire_date").reset_index(drop=True)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 9. 黄金主力连续日线（新浪）
# ---------------------------------------------------------------------------
@cache_data(ttl=12 * 3600)
def get_gold_main_daily(start_date: str = "20180101",
                        end_date: str | None = None) -> pd.DataFrame:
    """返回列: date, close, volume, open_interest（上期所 AU 主力连续，元/克）。"""
    end_date = end_date or datetime.today().strftime("%Y%m%d")
    raw = ak.futures_main_sina(symbol="AU0", start_date=start_date, end_date=end_date)
    df = raw.rename(columns={"收盘价": "close", "成交量": "volume", "持仓量": "open_interest"})
    df["date"] = _to_dt(df["日期"])
    for c in ["close", "volume", "open_interest"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["date", "close", "volume", "open_interest"]].dropna(
        subset=["date", "close"]).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 10. CFTC COT 持仓报告（COMEX 黄金，Managed Money 管理基金）
# ---------------------------------------------------------------------------
GOLD_CFTC_CODE = "088691"  # GOLD - COMMODITY EXCHANGE INC. (COMEX)；
#                           注意 088695=MICRO GOLD、180LM9=PAX GOLD，均不可混入


def _parse_cot_zip_text(text: str) -> pd.DataFrame:
    """解析 CFTC 年度归档 fut_disagg_txt_YYYY.zip 内 f_year.txt（带表头）。"""
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
    if gold.empty:  # 个别年份代码列带空格，退回精确名称（排除 MICRO GOLD / PAX GOLD）
        gold = df[df["market"].astype(str).str.strip().str.startswith(
            "GOLD - COMMODITY EXCHANGE", na=False)].copy()
    gold["date"] = _to_dt(gold["date"])
    for c in ["open_interest", "mm_long", "mm_short", "mm_spread"]:
        gold[c] = pd.to_numeric(gold[c], errors="coerce")
    gold["mm_net"] = gold["mm_long"] - gold["mm_short"]
    return gold[["date", "open_interest", "mm_long", "mm_short", "mm_spread", "mm_net"]]


@cache_data(ttl=24 * 3600)
def get_cot_gold(years: int = 3) -> pd.DataFrame:
    """
    拉取 COMEX 黄金 CFTC 持仓（周频，每周二截止、周五左右发布）。
    使用 CFTC 年度归档 fut_disagg_txt_YYYY.zip（Disaggregated Futures-only，
    当年 zip 含年初至今全部周度数据且更新及时；官网当年实时文件 deafut.txt 列布局
    不同，切勿混用）。
    years: 含当年在内向前取多少年。返回按日期升序:
           date, open_interest, mm_long, mm_short, mm_spread, mm_net（单位：张合约）。
    """
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
        frames.append(_parse_cot_zip_text(text))
    if not frames:
        return pd.DataFrame(columns=["date", "open_interest", "mm_long",
                                     "mm_short", "mm_spread", "mm_net"])
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["date"]).drop_duplicates("date").sort_values(
        "date").reset_index(drop=True)
#（注：内容由AI生成）
