# -*- coding: utf-8 -*-
"""
app.py — 宏观交易指标看板（Streamlit）

复现“布鲁斯的金融宝藏”视频中的观察体系：
  ① 股债息差 + 标准差轨道（核心，中长期大盘顶底判断）
  ② SHIBOR 银行间资金面
  ③ 期权 PCR（短期情绪）
  ④ 股指期货升贴水（杠杆资金情绪）
  ⑤ 黄金：上期所 AU 期限结构 + CFTC 管理基金持仓

运行：streamlit run app.py
数据均来自公开渠道，仅供学习研究，不构成投资建议。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data_fetcher as dfetch
import indicators as ind
from data_source.factory import get_lookback_defaults

# ---------------------------------------------------------------------------
# 页面基础配置
# ---------------------------------------------------------------------------
st.set_page_config(page_title="宏观交易指标看板", page_icon="📊", layout="wide")

# PE 指数名称 -> (新浪现货代码, 中证官网代码, 乐咕月频PE名称)
INDEX_MAP = {
    "沪深300": ("sh000300", "000300", "沪深300"),
    "上证50": ("sh000016", "000016", "上证50"),
    "中证500": ("sh000905", "000905", "中证500"),
    "创业板指": ("sz399006", "399006", "创业板指"),
    "科创50": ("sh000688", "000688", "科创50"),
    "上证指数": ("sh000001", "000001", "上证指数"),
}

C_BLUE, C_RED, C_GRAY, C_GREEN, C_ORANGE = "#1f5fbf", "#d6332a", "#9aa5b1", "#1a9e5d", "#e08a1e"

# 可叠加的常用指数（名称 -> 新浪代码）
COMMON_INDICES = ["上证指数", "深证成指", "沪深300", "上证50", "中证500",
                  "中证1000", "创业板指", "科创50"]
_COMMON_INDEX_CODES = {
    "上证指数": "sh000001", "深证成指": "sz399001", "沪深300": "sh000300",
    "上证50": "sh000016", "中证500": "sh000905", "中证1000": "sh000852",
    "创业板指": "sz399006", "科创50": "sh000688",
}
_INDEX_COLORS = [C_BLUE, C_RED, C_GREEN, C_ORANGE, "#7a3e9d", "#0f8b8d", "#b5651d", C_GRAY]


def _canonical_underlying(name: str) -> str:
    """把不同数据源的标的名统一为展示用标准名。

    tushare 归一为 "50ETF/300ETF/创业板ETF/科创50ETF"，
    akshare 直接用交易所原名（"上证50ETF/沪深300ETF/..."），
    此处统一映射到：上证50 / 沪深300 / 中证500 / 创业板 / 科创50。
    """
    n = str(name)
    if "科创" in n:
        return "科创50"
    if "创业板" in n:
        return "创业板"
    if "300" in n:
        return "沪深300"
    if "500" in n:
        return "中证500"
    if "50" in n:
        return "上证50"
    return n


# 期权 PCR 默认展示的 4 个标的（从上到下顺序）及其对应叠加指数
# (标准标的名, 叠加指数名称)
DEFAULT_PCR_UNDERLYINGS = [
    ("上证50", "上证50"),
    ("沪深300", "沪深300"),
    ("创业板", "创业板指"),
    ("科创50", "科创50"),
]


@st.cache_data(ttl=12 * 3600)
def _load_index_close(name: str) -> pd.DataFrame:
    """加载单个指数的日线收盘价，结果缓存 12 小时。"""
    df = dfetch.get_index_daily(_COMMON_INDEX_CODES[name])
    return df[["date", "close"]].rename(columns={"close": name})


def add_index_overlay(fig, selected, date_min, date_max):
    """复用：把选中的常用指数叠加到 fig 的右轴(y2)。

    - selected: 指数名称列表（为空则原样返回）
    - date_min/date_max: 叠加的日期范围，与主图 x 轴对齐
    - 自动创建/复用 yaxis2，标题为「指数点位」
    """
    if not selected:
        return fig
    for i, name in enumerate(selected):
        idx = _load_index_close(name)
        idx = idx[(idx["date"] >= date_min) & (idx["date"] <= date_max)]
        if idx.empty:
            continue
        fig.add_trace(go.Scatter(
            x=idx["date"], y=idx[name], name=f"{name}（点位·右轴）",
            line=dict(width=1.4, color=_INDEX_COLORS[i % len(_INDEX_COLORS)]),
            opacity=0.72, yaxis="y2",
        ))
    fig.update_layout(
        yaxis2=dict(title="指数点位", overlaying="y", side="right", showgrid=False),
    )
    return fig


def _fetch_with_progress(cache_key: str, label: str, fetch_fn, *args, **kwargs):
    """复用的逐日回溯加载器：全局缓存（跨用户共享）+ Streamlit 进度条。

    - 命中全局缓存 → 静默返回，不显示进度条
    - 未命中 → 显示进度条，回调实时更新
    fetch_fn 必须支持 _progress_callback(current, total) 关键字参数。
    """
    if dfetch.is_cached(cache_key):
        return fetch_fn(*args, **kwargs)
    prog = st.progress(0.0, text=label)

    def _cb(cur, total):
        prog.progress(cur / total, text=f"拉取进度 {cur}/{total} 个自然日…")

    kwargs["_progress_callback"] = _cb
    result = fetch_fn(*args, **kwargs)
    prog.empty()
    return result


def show_data_sources(items: list) -> None:
    """在可折叠面板中展示本页用到的数据源说明。

    items: 每项可为 (数据项名称, data_fetcher 方法名) 或 (名称, 方法名, 最新日期)。
    最新日期可为 pd.Timestamp / str / None。
    每条显示：一级源（akshare/tushare）+ 二级渠道（原始发布渠道 + 频率）+ 数据截至日期。
    """
    lines = []
    for item in items:
        if len(item) == 3:
            label, method, latest_date = item
        else:
            label, method = item
            latest_date = None
        l1, l2 = dfetch.get_source_info(method)
        date_str = ""
        if latest_date is not None:
            if isinstance(latest_date, pd.Timestamp):
                date_str = f" ｜ 数据截至 {latest_date.strftime('%Y-%m-%d')}"
            else:
                date_str = f" ｜ 数据截至 {latest_date}"
        lines.append(f"- **{label}** — 一级 `{l1}` ｜ 二级：{l2}{date_str}")
    with st.expander("📡 本页数据源说明（一级库 / 二级渠道·频率 / 数据截至）", expanded=False):
        st.markdown("\n".join(lines))
        st.caption("一级 = 实际命中的数据获取库（失败自动降级时会切换）；二级 = 原始数据发布渠道与频率。")


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """将十六进制颜色转换为 Plotly 可用的 rgba(...) 字符串。"""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


with st.sidebar:
    st.header("⚙️ 参数设置")
    # 数据源选择：多源架构，akshare 始终可用，tushare 需配置 TUSHARE_TOKEN
    avail_sources = dfetch.available_sources()
    if "data_source" not in st.session_state:
        st.session_state["data_source"] = avail_sources[0] if avail_sources else "akshare"
    cur_src = st.selectbox(
        "数据源（失败自动降级）", avail_sources,
        index=avail_sources.index(st.session_state["data_source"])
        if st.session_state["data_source"] in avail_sources else 0,
        help="tushare 需在环境变量 TUSHARE_TOKEN 中配置；部分接口有积分门槛，不足时自动降级到 akshare")
    if cur_src != st.session_state["data_source"]:
        st.session_state["data_source"] = cur_src
        dfetch.set_data_source(cur_src)
        st.cache_data.clear()
        st.toast(f"已切换数据源为 {cur_src}，缓存已清空")
    pe_name = st.selectbox("息差定价基准（指数）", list(INDEX_MAP.keys()), index=0)
    band_days = st.slider("轨道滚动窗口（天）", 365, 2920, 1095, step=30,
                          help="PE 为月频数据，内部按 30.44 天/月换算为滚动月数；建议 3~5 年（1095~1825 天），窗口过短会导致轨道剧烈波动失效")
    sigma_k = st.slider("高亮轨道倍数 σ", 1.0, 2.5, 2.0, step=0.5)
    _lb = get_lookback_defaults()
    pcr_lookback = st.slider("PCR 回溯自然日数", 30, 730, _lb["pcr"], step=10,
                             help="期权 PCR 逐日回溯；tushare 历史充足，默认 6 个月（180 天），可扩展至 2 年")
    futures_lookback = st.slider("期指 / 中信持仓回溯自然日数", 30, 365, _lb["futures"],
                                 help="股指期货基差与中信期货持仓逐日回溯；默认约 4 个月，越大覆盖已到期合约越多")
    cot_years = st.slider("COT 持仓回溯年数", 1, 5, 3)
    if st.button("🔄 强制刷新全部数据", width="stretch"):
        st.cache_data.clear()
        # 清除 PCR / 期权券商持仓 / 中信持仓的全局缓存
        dfetch.clear_global_cache()
        st.toast("缓存已清空，重新拉取数据")
    st.caption("数据默认缓存 12~24 小时；首次加载 PCR/期指历史需逐日回溯，请耐心等待。")

sina_code, csindex_code, legu_name = INDEX_MAP[pe_name]

st.title("📊 宏观交易指标看板")
st.caption("股债息差（核心） · SHIBOR · 期权PCR · 股指期货升贴水 · 黄金COT与期限结构 ｜ 公开数据，仅供研究，不构成投资建议")

tab_core, tab_offshore, tab_shibor, tab_pcr, tab_basis, tab_gold, tab_citic = st.tabs([
    "① 股债息差系统", "② 离岸在岸息差", "③ 资金面 SHIBOR", "④ 期权 PCR",
    "⑤ 股指期货升贴水", "⑥ 黄金观察", "⑦ 中信期货持仓"])

# ---------------------------------------------------------------------------
# Tab ① 股债息差系统（核心）
# ---------------------------------------------------------------------------
with tab_core:
    try:
        with st.spinner("拉取国债收益率、指数 PE 与大盘行情…"):
            bond = dfetch.get_bond_10y("20100101")
            pe_d = dfetch.get_index_pe_daily(legu_name)
            index_daily = dfetch.get_index_daily(sina_code)

        show_data_sources([
            ("10 年期国债收益率", "get_bond_10y",
             bond["date"].max() if not bond.empty else None),
            ("指数市盈率（日频）", "get_index_pe_daily",
             pe_d["date"].max() if not pe_d.empty else None),
            ("大盘指数行情", "get_index_daily",
             index_daily["date"].max() if not index_daily.empty else None),
            ("指数市盈率（日频·近20日）", "get_index_pe_daily_csindex", None),
        ])

        erp_df = ind.build_erp_daily(pe_d, bond)
        idx_recent = index_daily[index_daily["date"] >= erp_df["date"].min()]
        # 日频样本：将自然日换算为交易日（252 交易日/年），窗口至少 120 个交易日
        window = max(120, round(band_days * 252 / 365))
        erp_df = ind.add_std_bands(erp_df, "erp", window=window, n_sigma=sigma_k)
        erp_df["pct_rank"] = ind.rolling_percentile(erp_df["erp"], window=max(window, 252))
        last = erp_df.iloc[-1]
        sig, sig_desc = ind.erp_signal(last["erp"], last["band_upper"], last["band_lower"])

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("股债风险溢价 ERP", f"{last['erp']:.2f} pct",
                  f"盈利收益率 {last['earnings_yield']:.2f}% − 10Y国债 {last['bond_yield']:.2f}%")
        c2.metric("最新信号", sig)
        c3.metric("滚动历史分位", f"{last['pct_rank']:.0f}%" if pd.notna(last["pct_rank"]) else "—")
        c4.metric(f"{pe_name} 滚动PE(TTM)", f"{last['pe_ttm']:.2f}")
        c5.metric("数据截至", last["date"].strftime("%Y-%m-%d"))

        st.info(f"**信号解读：** {sig_desc}。口径：ERP = 100/PE(TTM) − 10年期国债收益率（百分点）；"
                f"轨道为近 {band_days} 天（约 {window} 个交易日）滚动均值 ± {sigma_k}σ。触上轨=股票见底，触下轨=见顶（与视频规则一致）。")

        # 日期范围筛选：支持只看某一段时期
        erp_min = erp_df["date"].min().date()
        erp_max = erp_df["date"].max().date()
        date_range = st.date_input(
            "📅 显示日期范围（可只看局部时段）",
            value=(erp_min, erp_max),
            min_value=erp_min, max_value=erp_max,
            key="erp_date_range",
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            _sd, _ed = date_range
            erp_show = erp_df[(erp_df["date"].dt.date >= _sd) & (erp_df["date"].dt.date <= _ed)]
            idx_show = idx_recent[(idx_recent["date"].dt.date >= _sd) & (idx_recent["date"].dt.date <= _ed)]
        else:
            erp_show, idx_show = erp_df, idx_recent
        st.caption(f"当前显示 {len(erp_show)} 个交易日数据 · 轨道基于全量历史滚动计算，仅显示范围被裁剪")
        ol_erp = st.multiselect("叠加指数走势（右轴）", COMMON_INDICES, key="overlay_erp")

        fig = go.Figure()
        # ±2σ / ±1σ 浅色背景轨
        fig.add_trace(go.Scatter(x=erp_show["date"], y=erp_show["upper2"], name="+2σ",
                                 line=dict(width=1, color=C_GRAY), opacity=0.5,
                                 line_dash="dot", showlegend=True))
        fig.add_trace(go.Scatter(x=erp_show["date"], y=erp_show["lower2"], name="-2σ",
                                 line=dict(width=1, color=C_GRAY), opacity=0.5,
                                 line_dash="dot", showlegend=True))
        fig.add_trace(go.Scatter(x=erp_show["date"], y=erp_show["upper1"], name="+1σ",
                                 line=dict(width=1, color=C_GRAY), opacity=0.8,
                                 line_dash="dash", showlegend=True))
        fig.add_trace(go.Scatter(x=erp_show["date"], y=erp_show["lower1"], name="-1σ",
                                 line=dict(width=1, color=C_GRAY), opacity=0.8,
                                 line_dash="dash", showlegend=True))
        fig.add_trace(go.Scatter(x=erp_show["date"], y=erp_show["mean"], name="滚动均值",
                                 line=dict(width=1.5, color=C_ORANGE)))
        fig.add_trace(go.Scatter(x=erp_show["date"], y=erp_show["erp"], name="股债息差 ERP（蓝线）",
                                 line=dict(width=2.6, color=C_BLUE)))
        # 高亮选定的 ±kσ 轨道
        fig.add_trace(go.Scatter(x=erp_show["date"], y=erp_show["band_upper"],
                                 name=f"上轨 +{sigma_k}σ",
                                 line=dict(width=2, color=C_GREEN), line_dash="dash"))
        fig.add_trace(go.Scatter(x=erp_show["date"], y=erp_show["band_lower"],
                                 name=f"下轨 −{sigma_k}σ",
                                 line=dict(width=2, color=C_RED), line_dash="dash"))
        # 大盘指数（次轴，红线，仅参考）
        fig.add_trace(go.Scatter(x=idx_show["date"], y=idx_show["close"],
                                 name=f"{pe_name}指数（右轴·仅参考）",
                                 line=dict(width=1.6, color=C_RED, dash="solid"),
                                 opacity=0.55, yaxis="y2"))
        fig.update_layout(
            height=560, hovermode="x unified",
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(
                title="日期",
                rangeslider=dict(visible=True, thickness=0.06),
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1年", step="year", stepmode="backward"),
                        dict(count=2, label="2年", step="year", stepmode="backward"),
                        dict(count=3, label="3年", step="year", stepmode="backward"),
                        dict(count=5, label="5年", step="year", stepmode="backward"),
                        dict(step="all", label="全部"),
                    ]),
                ),
            ),
            yaxis=dict(title="ERP（百分点）"),
            yaxis2=dict(title=f"{pe_name}点位", overlaying="y", side="right",
                        showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        add_index_overlay(fig, ol_erp, erp_show["date"].min(), erp_show["date"].max())
        st.plotly_chart(fig, width="stretch")

        # 近期日频 PE（中证官网，近约 20 个交易日，补充月频精度）
        try:
            pe_d = dfetch.get_index_pe_daily_csindex(csindex_code)
            if not pe_d.empty:
                d = pe_d.iloc[-1]
                with st.expander(f"📎 近期日频估值（中证指数官网，截至 {d['date'].strftime('%Y-%m-%d')}）"):
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric("市盈率1（官网口径）", f"{d['pe_static']:.2f}")
                    cc2.metric("市盈率2（滚动TTM口径）", f"{d['pe_ttm']:.2f}")
                    cc3.metric("股息率1", f"{d['div_yield']:.2f}%")
                    st.caption("官网两个市盈率口径与乐咕 TTM 口径存在差异，长期轨道以乐咕月频序列为准；"
                               "日频全历史 PE 需升级 Tushare Pro / Wind / iFinD，见 README。")
        except Exception as e:
            st.caption(f"中证官网日频估值暂不可用：{type(e).__name__}（不影响主图）")
    except Exception as e:
        st.error(f"股债息差模块加载失败：{type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# Tab ② 离岸在岸息差系统（美元加息确定性 + 跨境资金 + 本土资金 + 共振矩阵）
# ---------------------------------------------------------------------------
with tab_offshore:
    try:
        with st.spinner("拉取 EFFR / SOFR / CNH Hibor / 汇率 / 国债收益率 / 美联储事件…"):
            effr = dfetch.get_effr("20200101")
            sofr = dfetch.get_sofr_futures("20200101")
            cnh_hibor = dfetch.get_cnh_hibor()
            usd_cny = dfetch.get_usd_cny()
            usd_cnh = dfetch.get_usd_cnh()
            shibor_full = dfetch.get_shibor()
            bond_curve = dfetch.get_bond_yield_curve("20100101")
            pe_hs300 = dfetch.get_index_pe_monthly("沪深300")
            us_events = dfetch.get_us_macro_events("20200101")
            us_tys = dfetch.get_us_treasury_yields("20200101")

        show_data_sources([
            ("有效联邦基金利率 EFFR", "get_effr",
             effr["date"].max() if not effr.empty else None),
            ("3个月SOFR期货 / SOFR90天均值", "get_sofr_futures",
             sofr["date"].max() if not sofr.empty else None),
            ("离岸人民币3M Hibor", "get_cnh_hibor",
             cnh_hibor["date"].max() if not cnh_hibor.empty else None),
            ("在岸人民币汇率 USD/CNY", "get_usd_cny",
             usd_cny["date"].max() if not usd_cny.empty else None),
            ("离岸人民币汇率 USD/CNH", "get_usd_cnh",
             usd_cnh["date"].max() if not usd_cnh.empty else None),
            ("中国国债收益率曲线 2Y/10Y", "get_bond_yield_curve",
             bond_curve["date"].max() if not bond_curve.empty else None),
            ("美联储事件日历 FOMC/CPI/PPI", "get_us_macro_events",
             us_events["event_date"].max() if not us_events.empty else None),
            ("美国国债收益率曲线", "get_us_treasury_yields",
             us_tys["date"].max() if not us_tys.empty else None),
        ])

        # ---- 美联储事件规律教育提示 ----
        with st.expander("📚 美联储重大事件规律速查（鼠标悬浮查看）", expanded=False):
            # 计算各类事件的"上一次"与"下一次（估算）"日期
            _today = pd.Timestamp.now().normalize()
            _last_next = {}
            _typi = {"FOMC": 42, "CPI": 30, "PPI": 30}  # 典型间隔（天）
            for _et in ["FOMC", "CPI", "PPI"]:
                _sub = us_events[us_events["event_type"] == _et]
                if not _sub.empty:
                    _last = _sub["event_date"].max()
                    _last_next[_et] = (
                        _last.strftime("%Y-%m-%d"),
                        (_last + pd.Timedelta(days=_typi[_et])).strftime("%Y-%m-%d") + "（估）",
                    )
                else:
                    _last_next[_et] = ("—", "—")

            _fed_event_rules = [
                ("🏛️ FOMC 利率决议", "每年 8 次（约每 6 周一次），通常为周二-周三两日会议，"
                 "决议于美东时间周三 14:00 公布。加息/降息/按兵不动三种结果，"
                 "同时发布声明（Statement）。是影响美元利率与跨境资金的最核心事件。",
                 _last_next.get("FOMC", ("—", "—"))),
                ("📊 点阵图（Dot Plot）", "仅在 3 月 / 6 月 / 9 月 / 12 月的 FOMC 会议后发布，"
                 "反映 19 位委员对未来联邦基金利率的中位数预期。市场重点关注当年末与次年末的中位数。",
                 ("随对应季月 FOMC", "3/6/9/12 月")),
                ("📈 CPI（消费者物价指数）", "由 BLS 公布，通常在每月 10 日-15 日之间发布前月数据。"
                 "分整体 CPI 与核心 CPI（剔除食品能源），是美联储 2% 通胀目标的核心参考。"
                 "数据公布日美债与美元波动显著放大。",
                 _last_next.get("CPI", ("—", "—"))),
                ("🏭 PPI（生产者物价指数）", "由 BLS 公布，通常每月 11 日-16 日发布前月数据，"
                 "略早于或同步于 CPI。PPI 领先 CPI，反映上游价格压力，常被视为 CPI 的先行指标。",
                 _last_next.get("PPI", ("—", "—"))),
                ("💼 非农就业（NFP）", "由 BLS 公布，每月第一个周五发布前月数据。"
                 "包含新增非农就业、失业率、平均时薪。是美联储判断劳动力市场与通胀的关键数据。",
                 ("—", "每月首周五")),
                ("🏦 鲍威尔讲话 / 杰克逊霍尔年会", "FOMC 主席在会议间隙的讲话常释放政策信号；"
                 "8 月杰克逊霍尔央行年会为全球央行政策转向的重要窗口。",
                 ("—", "8 月下旬")),
            ]
            _ec1, _ec2 = st.columns(2)
            for _i, (_t, _d, (_last, _nxt)) in enumerate(_fed_event_rules):
                with (_ec1 if _i % 2 == 0 else _ec2):
                    st.markdown(
                        f'<div style="border:1px solid {C_GRAY};border-radius:6px;'
                        f'padding:10px;margin-bottom:8px;background:rgba(255,255,255,0.04);">'
                        f'<div style="font-weight:600;font-size:0.95rem;color:#ffffff;">{_t}</div>'
                        f'<div style="font-size:0.82rem;color:#e0e0e0;margin-top:4px;line-height:1.5;">{_d}</div>'
                        f'<div style="font-size:0.78rem;color:#b0c4de;margin-top:6px;">'
                        f'🕒 上次: {_last} ｜ 下次: {_nxt}</div>'
                        f'</div>', unsafe_allow_html=True)
            st.caption("数据来源：FOMC 会议日程为公开固定安排；CPI/PPI 实际公布日由 FRED "
                       "（CPIAUCSL / PPIACO）的 realtime_start 还原，含同比/环比与强弱提示。")

        # ---- 计算各息差 ----
        usd_sp = ind.build_usd_offshore_onshore_spread(effr, sofr)
        cny_rate_sp = ind.build_cny_rate_spread(cnh_hibor, shibor_full)
        cny_fx_sp = ind.build_cny_fx_spread(usd_cnh, usd_cny)
        bond_term = ind.build_bond_term_spread(bond_curve)
        erp = ind.build_erp_from_curve(pe_hs300, bond_curve)

        # ERP 滚动分位（用 60 个月窗口）
        erp["pct_rank"] = ind.rolling_percentile(erp["erp"], window=60)

        # ---- 顶部信号摘要条 ----
        def _safe_last(df, col):
            return df[col].iloc[-1] if not df.empty and col in df.columns else None

        usd_last = _safe_last(usd_sp, "spread_bp")
        rate_last = _safe_last(cny_rate_sp, "spread_bp")
        fx_last = _safe_last(cny_fx_sp, "spread_bp")
        term_last = _safe_last(bond_term, "spread_pct")
        erp_last = _safe_last(erp, "erp")
        erp_pct = _safe_last(erp, "pct_rank")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("美元离在岸息差",
                  f"{usd_last:+.0f} bp" if usd_last is not None else "—",
                  "隐含离岸 − EFFR")
        m2.metric("CNH 离在岸息差",
                  f"{rate_last:+.0f} bp" if rate_last is not None else "—",
                  "CNH Hibor − Shibor")
        m3.metric("CNH 离在岸汇差",
                  f"{fx_last:+.0f} bp" if fx_last is not None else "—",
                  "(USDCNH − USDCNY)×1e4")
        m4.metric("国债 10Y−2Y",
                  f"{term_last:+.2f}%" if term_last is not None else "—",
                  "长短端利差")
        res_label, res_desc = ind.resonance_signal(fx_last, rate_last, erp_pct, term_last)
        m5.metric("内外资金共振", res_label,
                  f"ERP 分位 {erp_pct:.0f}%" if erp_pct is not None else "")

        # ---- 日期范围选择器 ----
        all_dates = pd.concat([
            usd_sp["date"], cny_rate_sp["date"], cny_fx_sp["date"],
            bond_term["date"], erp["date"],
        ]).dropna()
        if not all_dates.empty:
            d_min, d_max = all_dates.min().date(), all_dates.max().date()
            dr = st.date_input("📅 显示日期范围", value=(d_min, d_max),
                               min_value=d_min, max_value=d_max, key="os_date_range")
            if isinstance(dr, (list, tuple)) and len(dr) == 2:
                _sd, _ed = dr
            else:
                _sd, _ed = d_min, d_max
        else:
            _sd = _ed = None

        def _crop(df, sd, ed):
            if df.empty or sd is None:
                return df
            return df[(df["date"].dt.date >= sd) & (df["date"].dt.date <= ed)]

        # ===== 模块A：美联储加息确定性（美元离在岸息差）=====
        st.subheader("🇺🇸 模块A｜美联储加息确定性：离岸−在岸美元息差")
        if not usd_sp.empty:
            usd_show = _crop(usd_sp, _sd, _ed)
            u1, u2, u3 = st.columns(3)
            u1.metric("隐含离岸利率",
                      f"{usd_show['implied_offshore'].iloc[-1]:.2f}%"
                      if not usd_show.empty else "—")
            u2.metric("EFFR（在岸）",
                      f"{usd_show['effr'].iloc[-1]:.2f}%"
                      if not usd_show.empty else "—")
            u3.metric("息差",
                      f"{usd_show['spread_bp'].iloc[-1]:+.0f} bp"
                      if not usd_show.empty else "—")

            with st.expander("📐 标准差轨道参数", expanded=False):
                bw_a, bs_a = st.columns(2)
                with bw_a:
                    win_a = st.slider("滚动窗口（交易日）", 30, 250, 120, step=10, key="os_win_a")
                with bs_a:
                    sig_a = st.slider("轨道倍数 σ", 1.0, 3.0, 2.0, step=0.1, key="os_sig_a")
            usd_band = ind.add_std_bands(usd_sp, "spread_bp", window=win_a, n_sigma=sig_a)
            usd_band_show = _crop(usd_band, _sd, _ed)

            fig_a = go.Figure()
            fig_a.add_hline(y=0, line_color=C_GRAY, line_dash="dash")
            for lb, ub, dash in [("lower1", "upper1", "dot"), ("lower2", "upper2", "longdash")]:
                fig_a.add_trace(go.Scatter(x=usd_band_show["date"], y=usd_band_show[ub],
                                           line=dict(color=C_GRAY, width=1, dash=dash),
                                           opacity=0.4, showlegend=False, hoverinfo="skip"))
                fig_a.add_trace(go.Scatter(x=usd_band_show["date"], y=usd_band_show[lb],
                                           line=dict(color=C_GRAY, width=1, dash=dash),
                                           opacity=0.4, showlegend=False, hoverinfo="skip",
                                           fill="tonexty",
                                           fillcolor=_hex_to_rgba(C_GRAY, 0.05)
                                           if lb == "lower2" else None))
            fig_a.add_trace(go.Scatter(x=usd_band_show["date"], y=usd_band_show["mean"],
                                       name="滚动均值", line=dict(color=C_ORANGE, width=1.2, dash="dash")))
            fig_a.add_trace(go.Scatter(x=usd_band_show["date"], y=usd_band_show["spread_bp"],
                                       name="美元离在岸息差(bp)", line=dict(color=C_BLUE, width=2.4)))

            # ---- 息差构成项叠加（右轴 %）：隐含离岸利率 / EFFR ----
            _comp_sel = st.multiselect(
                "叠加息差构成项（右轴 %）",
                ["隐含离岸利率 (100−SR3)", "EFFR 在岸"],
                default=[], key="os_comp_overlay")
            _comp_map = {
                "隐含离岸利率 (100−SR3)": ("implied_offshore", C_ORANGE, "dash"),
                "EFFR 在岸": ("effr", "#7a3e9d", "dot"),
            }
            for _label in _comp_sel:
                _col, _color, _dash = _comp_map[_label]
                if _col in usd_band_show.columns:
                    fig_a.add_trace(go.Scatter(
                        x=usd_band_show["date"], y=usd_band_show[_col],
                        name=_label, line=dict(color=_color, width=1.6, dash=_dash),
                        yaxis="y2", opacity=0.9,
                    ))

            # ---- 美债收益率叠加（右轴，多选，默认 30Y）----
            if not us_tys.empty:
                _tys_show = us_tys[(us_tys["date"].dt.date >= _sd) &
                                   (us_tys["date"].dt.date <= _ed)].copy()
                _ty_choices = sorted(_tys_show["tenor"].unique().tolist(),
                                     key=lambda x: ["1M", "3M", "6M", "1Y", "2Y", "5Y",
                                                    "7Y", "10Y", "20Y", "30Y"].index(x)
                                     if x in ["1M", "3M", "6M", "1Y", "2Y", "5Y",
                                              "7Y", "10Y", "20Y", "30Y"] else 99)
                _ty_sel = st.multiselect("叠加美债收益率（右轴 %）", _ty_choices,
                                         default=["30Y"] if "30Y" in _ty_choices else [],
                                         key="os_ty_overlay")
                _ty_palette = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd",
                               "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
                for _i, _tn in enumerate(_ty_sel):
                    _sub = _tys_show[_tys_show["tenor"] == _tn].sort_values("date")
                    if _sub.empty:
                        continue
                    fig_a.add_trace(go.Scatter(
                        x=_sub["date"], y=_sub["yield_pct"], name=f"美债{_tn}",
                        line=dict(color=_ty_palette[_i % len(_ty_palette)], width=1.5,
                                  dash="solid"),
                        yaxis="y2", opacity=0.85,
                    ))

            # ---- 美联储事件标记点（悬浮显示当日重大事件）----
            _show_events = st.checkbox("🔔 叠加美联储事件标记（FOMC/CPI/PPI）",
                                       value=True, key="os_show_events")
            if _show_events and not us_events.empty:
                _evt_show = us_events[(us_events["event_date"].dt.date >= _sd) &
                                      (us_events["event_date"].dt.date <= _ed)].copy()
                if not _evt_show.empty:
                    # 标记点放在图表上方（上轨之上 15% 位置）
                    _y_top = usd_band_show["upper2"].max() if not usd_band_show["upper2"].dropna().empty else 0
                    _y_marker = _y_top + abs(_y_top) * 0.15 + 5
                    # 按事件子类型区分颜色/符号/大小
                    # FOMC: 加息(深红⭐) / 降息(深绿⭐) / 持平(琥珀◆)
                    # CPI: 蓝圆 / PPI: 绿圆
                    _fomc_color = {"FOMC 加息": "#dc2626", "FOMC 降息": "#16a34a",
                                   "FOMC 持平": "#f59e0b"}
                    _fomc_symbol = {"FOMC 加息": "star", "FOMC 降息": "star",
                                    "FOMC 持平": "diamond"}
                    _fomc_size = {"FOMC 加息": 15, "FOMC 降息": 15, "FOMC 持平": 10}

                    # --- FOMC（按加息/降息/持平分组，各自独立 trace）---
                    for _fname in ["FOMC 加息", "FOMC 降息", "FOMC 持平"]:
                        _sub = _evt_show[_evt_show["event_name"] == _fname]
                        if _sub.empty:
                            continue
                        _hover = [
                            f"<b>{r['event_name']}</b><br>"
                            f"日期: {r['event_date'].strftime('%Y-%m-%d')}<br>"
                            f"{r['detail']}"
                            + (f"<br>立场: {r['surprise_hint']}" if r.get("surprise_hint") else "")
                            for _, r in _sub.iterrows()
                        ]
                        fig_a.add_trace(go.Scatter(
                            x=_sub["event_date"], y=[_y_marker] * len(_sub),
                            mode="markers", name=_fname,
                            marker=dict(size=_fomc_size[_fname],
                                        color=_fomc_color[_fname],
                                        symbol=_fomc_symbol[_fname],
                                        line=dict(width=1.5, color="white")),
                            text=_hover, hoverinfo="text",
                            hoverlabel=dict(bgcolor="white", font_size=12),
                            showlegend=True,
                        ))
                        # 竖线颜色跟随事件类型
                        for _, r in _sub.iterrows():
                            fig_a.add_vline(x=r["event_date"],
                                            line_color=_fomc_color[_fname],
                                            line_width=1, line_dash="dot", opacity=0.4)

                    # --- CPI / PPI ---
                    _other = {"CPI": (C_BLUE, "circle", 6),
                              "PPI": (C_GREEN, "circle", 6)}
                    for _etype, (_color, _sym, _sz) in _other.items():
                        _sub = _evt_show[_evt_show["event_type"] == _etype]
                        if _sub.empty:
                            continue
                        _hover = [
                            f"<b>{r['event_name']}</b><br>"
                            f"日期: {r['event_date'].strftime('%Y-%m-%d')}<br>"
                            f"{r['detail']}"
                            + (f"<br>强弱: {r['surprise_hint']}" if r.get("surprise_hint") else "")
                            for _, r in _sub.iterrows()
                        ]
                        fig_a.add_trace(go.Scatter(
                            x=_sub["event_date"], y=[_y_marker] * len(_sub),
                            mode="markers", name=f"{_etype} 公布",
                            marker=dict(size=_sz, color=_color, symbol=_sym,
                                        line=dict(width=1, color="white")),
                            text=_hover, hoverinfo="text",
                            hoverlabel=dict(bgcolor="white", font_size=12),
                            showlegend=True,
                        ))
            fig_a.update_layout(
                height=400, margin=dict(l=10, r=10, t=30, b=10),
                yaxis=dict(title="息差 bp"),
                yaxis2=dict(title="美债收益率 %", overlaying="y", side="right",
                            showgrid=False),
                hovermode="x unified",
                legend=dict(orientation="h", y=1.08, font_size=10),
                xaxis=dict(rangeslider=dict(visible=True, thickness=0.06),
                           rangeselector=dict(buttons=list([
                               dict(count=6, label="6月", step="month", stepmode="backward"),
                               dict(count=1, label="1年", step="year", stepmode="backward"),
                               dict(count=2, label="2年", step="year", stepmode="backward"),
                               dict(step="all", label="全部"),
                           ]))),
            )
            st.plotly_chart(fig_a, width="stretch")
            st.caption("息差 = (100 − SR3期货价) − EFFR。提前台阶式上行 ≈ 预期加息点数 → 加息板上钉钉；"
                       "走平/收窄 → 存在不确定性。轨道基于全量历史滚动计算。"
                       "🔔 顶部标记：⭐加息(深红)、⭐降息(深绿)、◆持平(琥珀)，均带竖线；"
                       "●CPI 公布(蓝)、●PPI 公布(绿)。悬浮查看事件详情与立场提示。美债收益率以右轴(%)叠加。")
        else:
            st.warning("美元息差数据不足（EFFR 或 SOFR 拉取失败）。")

        # ===== 模块B：跨境资金（人民币离在岸息差 + 汇差）=====
        st.subheader("🇨🇳 模块B｜跨境资金：人民币离在岸息差与汇差")
        b1, b2 = st.columns(2)
        with b1:
            st.markdown("**离在岸息差（CNH Hibor − Shibor，bp）**")
            if not cny_rate_sp.empty:
                rs = _crop(cny_rate_sp, _sd, _ed)
                fig_r = go.Figure()
                fig_r.add_hline(y=0, line_color=C_GRAY, line_dash="dash")
                fig_r.add_trace(go.Scatter(x=rs["date"], y=rs["spread_bp"],
                                           name="息差(bp)", line=dict(color=C_BLUE, width=2.2),
                                           fill="tozeroy", fillcolor=_hex_to_rgba(C_BLUE, 0.08)))
                fig_r.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                                    yaxis_title="bp", showlegend=False,
                                    xaxis=dict(rangeslider=dict(visible=True, thickness=0.05)))
                st.plotly_chart(fig_r, width="stretch")
                st.caption("息差走阔 → 离岸人民币紧、贬值预期强、外资倾向卖出人民币资产。")
            else:
                st.warning("CNH Hibor 或 Shibor 数据不足。")
        with b2:
            st.markdown("**离在岸汇差（(USDCNH − USDCNY)×1e4，bp）**")
            if not cny_fx_sp.empty:
                fs = _crop(cny_fx_sp, _sd, _ed)
                fig_f = go.Figure()
                fig_f.add_hline(y=0, line_color=C_GRAY, line_dash="dash")
                fig_f.add_trace(go.Scatter(x=fs["date"], y=fs["spread_bp"],
                                           name="汇差(bp)", line=dict(color=C_RED, width=2.2),
                                           fill="tozeroy", fillcolor=_hex_to_rgba(C_RED, 0.08)))
                fig_f.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                                    yaxis_title="bp", showlegend=False,
                                    xaxis=dict(rangeslider=dict(visible=True, thickness=0.05)))
                st.plotly_chart(fig_f, width="stretch")
                st.caption("汇差为正且扩大 → 离岸贬值预期强于境内，北向资金倾向流出。")
            else:
                st.warning("USD/CNH 或 USD/CNY 数据不足（yfinance 可能被网络限制，已降级实时报价）。")

        # ===== 模块C：本土资金（国债期限利差 + A股股债收益差）=====
        st.subheader("🏦 模块C｜本土资金：国债期限利差 + A股股债收益差")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**国债长短端利差（10Y − 2Y，%）**")
            if not bond_term.empty:
                ts = _crop(bond_term, _sd, _ed)
                fig_t = go.Figure()
                fig_t.add_hline(y=0, line_color=C_GRAY, line_dash="dash")
                fig_t.add_trace(go.Scatter(x=ts["date"], y=ts["spread_pct"],
                                           name="10Y−2Y(%)", line=dict(color=C_GREEN, width=2.2),
                                           fill="tozeroy", fillcolor=_hex_to_rgba(C_GREEN, 0.08)))
                fig_t.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                                    yaxis_title="%", showlegend=False,
                                    xaxis=dict(rangeslider=dict(visible=True, thickness=0.05)))
                st.plotly_chart(fig_t, width="stretch")
                st.caption("利差收窄/倒挂 → 经济走弱预期、资金面偏紧；走阔 → 经济预期改善。")
            else:
                st.warning("国债收益率曲线数据不足。")
        with c2:
            st.markdown("**A股股债收益差（ERP = 盈利收益率 − 10Y国债）**")
            if not erp.empty:
                es = _crop(erp, _sd, _ed)
                with st.expander("📐 ERP 标准差轨道", expanded=False):
                    bw_e, bs_e = st.columns(2)
                    with bw_e:
                        win_e = st.slider("滚动窗口（月）", 12, 120, 60, step=6, key="os_win_e")
                    with bs_e:
                        sig_e = st.slider("轨道倍数 σ", 1.0, 3.0, 2.0, step=0.1, key="os_sig_e")
                erp_band = ind.add_std_bands(erp, "erp", window=win_e, n_sigma=sig_e)
                es_b = _crop(erp_band, _sd, _ed)
                ol_e = st.multiselect("叠加指数（右轴）", COMMON_INDICES, key="overlay_os_erp")
                fig_e = go.Figure()
                for lb, ub, dash in [("lower1", "upper1", "dot"), ("lower2", "upper2", "longdash")]:
                    fig_e.add_trace(go.Scatter(x=es_b["date"], y=es_b[ub],
                                               line=dict(color=C_GRAY, width=1, dash=dash),
                                               opacity=0.4, showlegend=False, hoverinfo="skip"))
                    fig_e.add_trace(go.Scatter(x=es_b["date"], y=es_b[lb],
                                               line=dict(color=C_GRAY, width=1, dash=dash),
                                               opacity=0.4, showlegend=False, hoverinfo="skip",
                                               fill="tonexty",
                                               fillcolor=_hex_to_rgba(C_GRAY, 0.05)
                                               if lb == "lower2" else None))
                fig_e.add_trace(go.Scatter(x=es_b["date"], y=es_b["mean"],
                                           name="均值", line=dict(color=C_ORANGE, width=1.2, dash="dash")))
                fig_e.add_trace(go.Scatter(x=es_b["date"], y=es_b["erp"],
                                           name="ERP(%)", line=dict(color=C_BLUE, width=2.4)))
                fig_e.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                                    yaxis_title="百分点", hovermode="x unified",
                                    xaxis=dict(rangeslider=dict(visible=True, thickness=0.05)),
                                    legend=dict(orientation="h", y=1.02))
                add_index_overlay(fig_e, ol_e, es_b["date"].min(), es_b["date"].max())
                st.plotly_chart(fig_e, width="stretch")
                st.caption("ERP 高位 → 股票相对债券吸引力极强，本土大资金从债市流向股市（底部信号）。")
            else:
                st.warning("沪深300 PE 或国债数据不足。")

        # ===== 模块D：内外资金共振判断矩阵 =====
        st.subheader("🧭 模块D｜内外资金共振判断矩阵")
        external_in = (fx_last is not None and fx_last < 0
                       and rate_last is not None and rate_last < 0)
        internal_strong = erp_pct is not None and erp_pct > 50

        quadrants = [
            ("外部流入 + 内部强", "🟢 牛市底部/启动期", external_in and internal_strong,
             "内外资金共振做多，指数级行情概率高"),
            ("外部流出 + 内部强", "🟡 内资主导修复行情", (not external_in) and internal_strong,
             "政策与内资驱动，外资重仓板块偏弱"),
            ("外部流入 + 内部弱", "🟡 外资主导结构性行情", external_in and (not internal_strong),
             "指数震荡，外资重仓赛道相对强势"),
            ("外部流出 + 内部弱", "🔴 熊市风险高", (not external_in) and (not internal_strong),
             "内外资金共振撤离，整体持续承压"),
        ]
        q1, q2 = st.columns(2)
        for i, (title, label, active, desc) in enumerate(quadrants):
            with (q1 if i % 2 == 0 else q2):
                bg = _hex_to_rgba(C_GREEN if active else C_GRAY, 0.12 if active else 0.04)
                bd = C_GREEN if active else C_GRAY
                st.markdown(
                    f'<div style="background:{bg};border:1px solid {bd};border-radius:8px;'
                    f'padding:14px;margin-bottom:10px;">'
                    f'<div style="font-size:1.1rem;font-weight:600;">{label}</div>'
                    f'<div style="font-size:0.85rem;color:#555;margin-top:4px;">{title}</div>'
                    f'<div style="font-size:0.8rem;color:#777;margin-top:6px;">{desc}</div>'
                    f'</div>', unsafe_allow_html=True)

        st.info(f"**当前状态：{res_label}** — {res_desc}")
        st.caption("判定规则：外部流入 = 汇差<0 且 息差<0；内部强 = ERP 历史分位>50%。"
                   "两个维度组合形成四象限。")
    except Exception as e:
        st.error(f"离岸在岸息差模块加载失败：{type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# Tab ③ SHIBOR
# ---------------------------------------------------------------------------
with tab_shibor:
    try:
        with st.spinner("拉取 SHIBOR…"):
            shibor = dfetch.get_shibor()
        show_data_sources([("SHIBOR 期限结构", "get_shibor",
                            shibor["date"].max() if not shibor.empty else None)])
        latest_date = shibor["date"].max()
        latest = shibor[shibor["date"] == latest_date].iloc[0]
        tenors = [c for c in shibor.columns if c != "date"]
        st.subheader(f"SHIBOR 期限结构（{latest_date.strftime('%Y-%m-%d')}，单位 %）")
        cols = st.columns(len(tenors))
        for col, t in zip(cols, tenors):
            col.metric(t, f"{latest[t]:.3f}" if pd.notna(latest[t]) else "—")

        fig_s = go.Figure()
        # 期限结构柱线
        fig_s.add_trace(go.Scatter(x=tenors, y=[latest[t] for t in tenors],
                                   mode="lines+markers", name="最新期限结构",
                                   line=dict(color=C_BLUE, width=2.5)))
        fig_s.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                            yaxis_title="利率 %", showlegend=False)
        st.plotly_chart(fig_s, width="stretch")

        st.subheader("历史走势")
        picked = st.multiselect("选择期限", tenors, default=["1周", "1个月", "3个月", "1年"])
        ol_shibor = st.multiselect("叠加指数走势（右轴）", COMMON_INDICES, key="overlay_shibor")
        if picked:
            fig_h = go.Figure()
            palette = [C_BLUE, C_RED, C_GREEN, C_ORANGE, "#7a3e9d", C_GRAY, "#0f8b8d", "#b5651d"]
            for i, t in enumerate(picked):
                fig_h.add_trace(go.Scatter(x=shibor["date"], y=shibor[t], name=t,
                                           line=dict(width=1.8, color=palette[i % len(palette)])))
            fig_h.update_layout(
                height=420, margin=dict(l=10, r=10, t=30, b=10),
                yaxis_title="SHIBOR %", hovermode="x unified",
                xaxis=dict(
                    rangeslider=dict(visible=True, thickness=0.06),
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1月", step="month", stepmode="backward"),
                            dict(count=3, label="3月", step="month", stepmode="backward"),
                            dict(count=6, label="6月", step="month", stepmode="backward"),
                            dict(count=1, label="1年", step="year", stepmode="backward"),
                            dict(step="all", label="全部"),
                        ]),
                    ),
                ),
            )
            add_index_overlay(fig_h, ol_shibor, shibor["date"].min(), shibor["date"].max())
            st.plotly_chart(fig_h, width="stretch")
        st.caption("数据源：shibor.org（全国银行间同业拆借中心）。短端利率飙升=资金面紧张，"
                   "视频中用于观察大资金流动性。")
    except Exception as e:
        st.error(f"SHIBOR 模块加载失败：{type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# Tab ④ 期权 PCR
# ---------------------------------------------------------------------------
with tab_pcr:
    try:
        # ---- 全市场 PCR（按标的 + 到期月聚合）----
        pcr_cache_key = f"pcr_{pcr_lookback}"
        pcr = _fetch_with_progress(
            pcr_cache_key,
            "正在逐日回溯拉取全市场期权数据（首次较慢，请稍候）…",
            dfetch.get_option_pcr_history, pcr_lookback)
        # 统一标的名（兼容 tushare/akshare 不同命名）
        if not pcr.empty:
            pcr["underlying"] = pcr["underlying"].map(_canonical_underlying)

        # 机构筛选开关：默认关闭（全市场），开启后才懒加载券商持仓数据
        enable_broker = st.checkbox(
            "🏛️ 按机构筛选（默认关闭=全市场）", value=False,
            help="开启后可选择具体机构（如中信），仅计算该机构的期权持仓 PCR。"
                 "上交所仅公开前 3 大标的认购/认沽前 5 名会员持仓量。")

        broker_df = pd.DataFrame()
        sel_brokers: list[str] = []
        if enable_broker:
            broker_cache_key = f"opt_broker_{pcr_lookback}"
            broker_df = _fetch_with_progress(
                broker_cache_key,
                "正在逐日回溯拉取期权券商持仓排名…",
                dfetch.get_option_broker_holding_history, pcr_lookback)
            # 统一标的名
            if not broker_df.empty:
                broker_df["underlying"] = broker_df["underlying"].map(_canonical_underlying)
            all_brokers = sorted(broker_df["broker"].unique()) if not broker_df.empty else []
            sel_brokers = st.multiselect(
                "选择机构（可多选，留空=全市场）", all_brokers, default=[],
                help="留空=全市场模式；选中后仅计算所选机构持仓 PCR。",
                key="pcr_sel_brokers")

        broker_mode = enable_broker and bool(sel_brokers)

        # 数据源展示（仅展示实际用到的源）
        src_items = [("全市场期权 PCR（分标的+到期月）", "get_option_pcr_history",
                      pcr["date"].max() if not pcr.empty else None)]
        if enable_broker:
            src_items.append(("期权券商持仓排名", "get_option_broker_holding_history",
                              broker_df["date"].max() if not broker_df.empty else None))
        show_data_sources(src_items)

        if pcr.empty and (not enable_broker or broker_df.empty):
            st.warning("未获取到期权数据（可能是非交易时段或交易所接口暂不可用）。")
        else:
            # ===== 到期月份多选（仅全市场模式可用）=====
            # 到期月份多选：仅全市场模式可用（券商持仓数据按标的聚合，无到期月细分）
            all_months = sorted(pcr["exp_month"].unique()) if not pcr.empty else []
            cur_ym = pd.Timestamp.now().strftime("%y%m")
            ongoing_months = [m for m in all_months if m != "全部" and m >= cur_ym]
            default_months = ongoing_months[:3] if ongoing_months else all_months[:3]
            month_labels = {
                m: (f"20{m[:2]}-{m[2:]}" if m != "全部" else "全部月份")
                for m in all_months
            }
            sel_months = st.multiselect(
                "📅 到期月份（可多选）", all_months, default=all_months,
                format_func=lambda m: month_labels.get(m, m),
                disabled=broker_mode,
                help="仅全市场模式（未选机构）可用。默认全选所有到期月份。"
                     if not broker_mode else "券商持仓数据按标的聚合，无到期月份细分。",
                key="pcr_sel_months")

            # ===== 标准差上下轨参数（单一指标时启用）=====
            with st.expander("📐 标准差上下轨（单一 PCR 指标时显示）", expanded=False):
                bw_c, bs_c = st.columns(2)
                with bw_c:
                    band_window = st.slider(
                        "滚动窗口（交易日）", 20, 180, 60, step=5,
                        help="计算均值与标准差的滚动窗口；窗口越长轨道越平滑，"
                             "过短会因样本不足而失效",
                        key="pcr_band_window")
                with bs_c:
                    band_sigma = st.slider(
                        "轨道倍数 σ", 1.0, 3.0, 2.0, step=0.1,
                        help="填充区间 = 均值 ± nσ；同时以虚线显示 ±1σ 与 ±2σ",
                        key="pcr_band_sigma")

            # ===== 数据分支：机构模式 vs 全市场模式 =====
            if broker_mode:
                # ---- 机构模式：用券商持仓数据，仅 OI-PCR ----
                if broker_df.empty:
                    st.warning("未获取到券商持仓数据。")
                else:
                    bdf = broker_df[broker_df["broker"].isin(sel_brokers)].copy()
                    if bdf.empty:
                        st.warning("所选机构在持仓排名中无数据。")
                    else:
                        latest_d = bdf["date"].max()
                        all_underlyings = sorted(bdf["underlying"].unique())

                        # 筛选区 2：标的多选
                        sel_underlyings = st.multiselect(
                            "📊 选择标的（可多选）", all_underlyings,
                            default=all_underlyings, key="pcr_broker_underlyings")

                        if not sel_underlyings:
                            st.warning("请至少选择一个标的。")
                        else:
                            df = bdf[bdf["underlying"].isin(sel_underlyings)].copy()
                            # 按 (date, underlying) 聚合所选机构的 call/put OI
                            agg = df.groupby(["date", "underlying"], as_index=False).agg(
                                call_oi=("call_oi", "sum"), put_oi=("put_oi", "sum"))
                            agg["pcr_oi"] = agg["put_oi"] / agg["call_oi"].mask(agg["call_oi"] == 0)

                            today_df = agg[agg["date"] == latest_d].copy()
                            total_call_o = today_df["call_oi"].sum()
                            total_put_o = today_df["put_oi"].sum()
                            agg_pcr_o = total_put_o / total_call_o if total_call_o else None

                            broker_names = "、".join(sel_brokers)
                            st.subheader(f"期权 PCR 概览（{latest_d.strftime('%Y-%m-%d')}，"
                                         f"机构：{broker_names}）")
                            k1, k2, k3 = st.columns(3)
                            k1.metric("合计 持仓量 PCR",
                                      f"{agg_pcr_o:.3f}" if agg_pcr_o else "—")
                            k2.metric("认购持仓（所选机构）", f"{total_call_o/1e4:.0f} 万张")
                            k3.metric("认沽持仓（所选机构）", f"{total_put_o/1e4:.0f} 万张")
                            if agg_pcr_o is not None:
                                st.caption("持仓量 PCR = 所选机构认沽持仓 / 认购持仓，"
                                           "反映机构中期多空倾向。" + ind.pcr_signal(agg_pcr_o))
                            st.caption("⚠️ 数据来源：上交所期权持仓排名（前 3 大标的 × 认购/"
                                       "认沽前 5 会员）。仅持仓量，无成交量；按标的聚合无到期月。")

                            # 历史走势（仅 OI-PCR）
                            st.markdown("##### 持仓量 PCR 历史走势")
                            ol_pcr = st.multiselect("叠加指数走势（右轴）", COMMON_INDICES,
                                                    key="overlay_pcr_broker")
                            palette = ["#1f5fbf", "#d6332a", "#1a9e5d", "#e08a1e",
                                       "#7a3e9d", "#0f8b8d", "#b5651d", "#9aa5b1"]
                            fig_p = go.Figure()
                            # 有效窗口 = 用户窗口 与 最短标的数据长度 的较小值，避免轨道全 NaN
                            min_pts = min(len(agg[agg["underlying"] == ul])
                                          for ul in sel_underlyings) if sel_underlyings else 0
                            eff_window = min(band_window, min_pts)
                            for i, ul in enumerate(sel_underlyings):
                                g = agg[agg["underlying"] == ul].sort_values("date").copy()
                                color = palette[i % len(palette)]
                                # 单一指标（持仓量 PCR），叠加标准差上下轨
                                g = ind.add_std_bands(g, "pcr_oi", window=eff_window,
                                                      n_sigma=band_sigma)
                                fig_p.add_trace(go.Scatter(
                                    x=g["date"], y=g["pcr_oi"],
                                    name=f"{ul} · 持仓量",
                                    line=dict(color=color, width=2.2, dash="solid")))
                                fig_p.add_trace(go.Scatter(
                                    x=g["date"], y=g["mean"],
                                    name=f"{ul} · 均值",
                                    line=dict(color=color, width=1.2, dash="dash"),
                                    opacity=0.6, showlegend=False))
                                # ±1σ 与 ±2σ 参考线
                                for lb, ub, dash in [("lower1", "upper1", "dot"),
                                                     ("lower2", "upper2", "longdash")]:
                                    fig_p.add_trace(go.Scatter(
                                        x=g["date"], y=g[ub], line=dict(
                                            color=color, width=1, dash=dash),
                                        opacity=0.35, showlegend=False,
                                        hoverinfo="skip"))
                                    fig_p.add_trace(go.Scatter(
                                        x=g["date"], y=g[lb], line=dict(
                                            color=color, width=1, dash=dash),
                                        opacity=0.35, showlegend=False,
                                        fill="tonexty",
                                        fillcolor=_hex_to_rgba(color, 0.06)
                                        if lb == "lower2" else None,
                                        hoverinfo="skip"))
                            fig_p.add_hline(y=1.0, line_dash="dash", line_color=C_GRAY,
                                            annotation_text="PCR=1（多空均衡）",
                                            annotation_position="top right")
                            fig_p.update_layout(
                                height=440, margin=dict(l=10, r=10, t=30, b=10),
                                yaxis_title="Put / Call Ratio", hovermode="x unified",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                                xaxis=dict(
                                    rangeslider=dict(visible=True, thickness=0.06),
                                    rangeselector=dict(buttons=list([
                                        dict(count=1, label="1月", step="month", stepmode="backward"),
                                        dict(count=3, label="3月", step="month", stepmode="backward"),
                                        dict(count=6, label="6月", step="month", stepmode="backward"),
                                        dict(step="all", label="全部"),
                                    ])),
                                ),
                            )
                            add_index_overlay(fig_p, ol_pcr, agg["date"].min(), agg["date"].max())
                            st.plotly_chart(fig_p, width="stretch")
                            _cap = (f"机构模式仅含持仓量 PCR，已叠加 ±{band_sigma:.1f}σ 填充区间"
                                    f"（有效窗口 {eff_window} 个交易日）；点线=±1σ，长虚线=±2σ。")
                            if eff_window < band_window:
                                _cap += f" 数据不足，窗口已从 {band_window} 自动缩减。"
                            st.caption(_cap)

                            # 最新日分标的明细
                            with st.expander(f"� 最新日分标的明细（{latest_d.strftime('%Y-%m-%d')}）"):
                                show_df = today_df[["underlying", "call_oi", "put_oi", "pcr_oi"]].copy()
                                show_df["pcr_oi"] = show_df["pcr_oi"].round(3)
                                st.dataframe(show_df.rename(columns={
                                    "underlying": "标的", "call_oi": "认购持仓",
                                    "put_oi": "认沽持仓", "pcr_oi": "持仓量PCR"}),
                                    width="stretch", hide_index=True)

                            # 所选机构分标的持仓明细
                            with st.expander("🏛️ 所选机构分标的持仓明细"):
                                detail = df[df["date"] == latest_d][
                                    ["underlying", "broker", "call_oi", "put_oi", "pcr_oi"]].copy()
                                detail["pcr_oi"] = detail["pcr_oi"].round(3)
                                st.dataframe(detail.rename(columns={
                                    "underlying": "标的", "broker": "机构",
                                    "call_oi": "认购持仓", "put_oi": "认沽持仓",
                                    "pcr_oi": "持仓量PCR"}), width="stretch", hide_index=True)

            else:
                # ---- 全市场模式：用全市场 PCR 数据，支持到期月份筛选 ----
                if pcr.empty:
                    st.warning("未获取到全市场期权数据。")
                else:
                    latest_d = pcr["date"].max()
                    all_underlyings = sorted(pcr["underlying"].unique())

                    # 默认选中全部标的（上证50/沪深300/创业板/科创50 排最前）
                    default_ul = [u for u, _ in DEFAULT_PCR_UNDERLYINGS
                                  if u in all_underlyings]
                    default_ul += [u for u in all_underlyings if u not in default_ul]

                    # 按到期月份过滤
                    month_df = pcr[pcr["exp_month"].isin(sel_months)].copy() if sel_months else pcr.copy()
                    # 聚合到 (date, underlying) 级别（跨所选到期月求和）
                    df = month_df.groupby(["date", "underlying"], as_index=False).agg(
                        call_volume=("call_volume", "sum"),
                        put_volume=("put_volume", "sum"),
                        call_oi=("call_oi", "sum"), put_oi=("put_oi", "sum"))
                    df["pcr_volume"] = df["put_volume"] / df["call_volume"].mask(df["call_volume"] == 0)
                    df["pcr_oi"] = df["put_oi"] / df["call_oi"].mask(df["call_oi"] == 0)

                    # 筛选区 2：标的多选 + 指标类型
                    c1, c2 = st.columns([3, 2])
                    with c1:
                        sel_underlyings = st.multiselect(
                            "📊 选择标的（可多选）", all_underlyings,
                            default=default_ul, key="pcr_sel_underlyings")
                    with c2:
                        st.write("📈 PCR 指标类型")
                        show_vol = st.checkbox("成交量 PCR", value=False, key="pcr_show_vol")
                        show_oi = st.checkbox("持仓量 PCR", value=True, key="pcr_show_oi")
                        show_kline_vol = st.checkbox("K线显示成交量", value=False,
                                                     key="pcr_show_kline_vol")
                        show_index = st.checkbox("叠加标的指数", value=True,
                                                 key="pcr_show_index")

                    if not sel_underlyings or not (show_vol or show_oi):
                        st.warning("请至少选择一个标的和一种 PCR 指标。")
                    else:
                        df = df[df["underlying"].isin(sel_underlyings)].copy()
                        today_df = df[df["date"] == latest_d].copy()
                        total_call_v = today_df["call_volume"].sum()
                        total_put_v = today_df["put_volume"].sum()
                        total_call_o = today_df["call_oi"].sum()
                        total_put_o = today_df["put_oi"].sum()
                        agg_pcr_v = total_put_v / total_call_v if total_call_v else None
                        agg_pcr_o = total_put_o / total_call_o if total_call_o else None

                        month_note = f"，到期月：{', '.join(sel_months)}" if sel_months else ""
                        st.subheader(f"全市场期权 PCR 概览（{latest_d.strftime('%Y-%m-%d')}，"
                                     f"已选 {len(sel_underlyings)} 个标的{month_note}）")
                        k1, k2, k3, k4 = st.columns(4)
                        k1.metric("合计 成交量 PCR",
                                  f"{agg_pcr_v:.3f}" if agg_pcr_v else "—")
                        k2.metric("合计 持仓量 PCR",
                                  f"{agg_pcr_o:.3f}" if agg_pcr_o else "—")
                        k3.metric("当日认购成交", f"{total_call_v/1e4:.0f} 万张")
                        k4.metric("当日认沽成交", f"{total_put_v/1e4:.0f} 万张")
                        if agg_pcr_v is not None:
                            st.caption("成交量 PCR 反映当日对冲/投机情绪；持仓量 PCR 反映中期防御仓位。"
                                       + ind.pcr_signal(agg_pcr_v))

                        st.markdown("##### PCR 历史走势（按标的分图，各叠加对应指数）")
                        # 标的 -> 自动叠加的对应指数
                        ul_to_index = dict(DEFAULT_PCR_UNDERLYINGS)
                        # 按默认顺序（上证50→沪深300→创业板→科创50）排列，其余追加末尾
                        ordered_ul = [u for u, _ in DEFAULT_PCR_UNDERLYINGS
                                      if u in sel_underlyings]
                        ordered_ul += [u for u in sel_underlyings if u not in ordered_ul]

                        # 仅当只选一种 PCR 指标时叠加标准差上下轨
                        single_ind = show_vol ^ show_oi
                        active_col = "pcr_volume" if show_vol else "pcr_oi"
                        palette = ["#1f5fbf", "#d6332a", "#1a9e5d", "#e08a1e",
                                   "#7a3e9d", "#0f8b8d", "#b5651d", "#9aa5b1"]

                        for i, ul in enumerate(ordered_ul):
                            g = df[df["underlying"] == ul].sort_values("date").copy()
                            if g.empty:
                                continue
                            color = palette[i % len(palette)]
                            band2_color = "#e0533d"
                            auto_idx = ul_to_index.get(ul)
                            sina_code = _COMMON_INDEX_CODES.get(auto_idx) if auto_idx else None
                            kl = (dfetch.get_index_kline(sina_code, lookback_days=pcr_lookback)
                                  if (sina_code and show_index) else pd.DataFrame())

                            fig = go.Figure()
                            if single_ind:
                                eff_w = min(band_window, len(g))
                                g = ind.add_std_bands(g, active_col, window=eff_w,
                                                      n_sigma=band_sigma)
                            if show_vol:
                                fig.add_trace(go.Scatter(
                                    x=g["date"], y=g["pcr_volume"],
                                    name=f"{ul} · 成交量 PCR",
                                    line=dict(color=color, width=3, dash="solid")))
                            if show_oi:
                                fig.add_trace(go.Scatter(
                                    x=g["date"], y=g["pcr_oi"],
                                    name=f"{ul} · 持仓量 PCR",
                                    line=dict(color=color, width=3, dash="solid")))
                            if single_ind:
                                fig.add_trace(go.Scatter(
                                    x=g["date"], y=g["mean"], name="均值",
                                    line=dict(color="#b0b0b0", width=1.2, dash="dash"),
                                    opacity=0.8, showlegend=False))
                                for lb, ub in [("lower1", "upper1")]:
                                    fig.add_trace(go.Scatter(
                                        x=g["date"], y=g[ub], line=dict(
                                            color=color, width=1, dash="dot"),
                                        opacity=0.3, showlegend=False, hoverinfo="skip"))
                                    fig.add_trace(go.Scatter(
                                        x=g["date"], y=g[lb], line=dict(
                                            color=color, width=1, dash="dot"),
                                        opacity=0.3, showlegend=False, hoverinfo="skip"))
                                for lb, ub in [("lower2", "upper2")]:
                                    fig.add_trace(go.Scatter(
                                        x=g["date"], y=g[ub], line=dict(
                                            color=band2_color, width=1.5, dash="longdash"),
                                        opacity=0.85, showlegend=False, hoverinfo="skip"))
                                    fig.add_trace(go.Scatter(
                                        x=g["date"], y=g[lb], line=dict(
                                            color=band2_color, width=1.5, dash="longdash"),
                                        opacity=0.85, showlegend=False, hoverinfo="skip"))
                            fig.add_hline(y=1.0, line_dash="dash", line_color=C_GRAY,
                                          annotation_text="PCR=1", annotation_position="top right")

                            # 标的指数 K 线（副轴叠加，可选）
                            if not kl.empty:
                                fig.add_trace(go.Candlestick(
                                    x=kl["date"], open=kl["open"], high=kl["high"],
                                    low=kl["low"], close=kl["close"],
                                    name=f"{auto_idx} K线", yaxis="y2",
                                    increasing_line_color="#d6332a",
                                    decreasing_line_color="#1a9e5d",
                                    increasing_fillcolor="#d6332a",
                                    decreasing_fillcolor="#1a9e5d",
                                ))
                                if show_kline_vol:
                                    vcolors = ["#d6332a" if c >= o else "#1a9e5d"
                                               for c, o in zip(kl["close"], kl["open"])]
                                    fig.add_trace(go.Bar(
                                        x=kl["date"], y=kl["volume"], name="成交量",
                                        marker_color=vcolors, opacity=0.25, yaxis="y3"))

                            layout = dict(
                                height=420, margin=dict(l=10, r=10, t=40, b=10),
                                title=f"{ul} PCR",
                                hovermode="x unified",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                                xaxis=dict(
                                    autorange=True,
                                    rangeslider=dict(visible=True, thickness=0.06),
                                    rangeselector=dict(buttons=list([
                                        dict(count=1, label="1月", step="month", stepmode="backward"),
                                        dict(count=3, label="3月", step="month", stepmode="backward"),
                                        dict(count=6, label="6月", step="month", stepmode="backward"),
                                        dict(step="all", label="全部"),
                                    ])),
                                ),
                                yaxis=dict(title="PCR"),
                            )
                            if not kl.empty:
                                layout["yaxis2"] = dict(
                                    title=auto_idx, overlaying="y", side="right",
                                    showgrid=False)
                            if show_kline_vol and not kl.empty:
                                layout["yaxis3"] = dict(
                                    title="成交量", overlaying="y", side="right",
                                    position=0.95, showgrid=False)
                            fig.update_layout(**layout)
                            st.plotly_chart(fig, width="stretch")

                        cap_parts = []
                        if show_vol:
                            cap_parts.append("实线=成交量 PCR")
                        if show_oi:
                            cap_parts.append("实线=持仓量 PCR")
                        if single_ind:
                            cap_parts.append(
                                f"已叠加 ±{band_sigma:.1f}σ 轨道（窗口 {band_window} 个交易日）；"
                                f"灰色虚线=均值，淡色点线=±1σ，橙红长虚线=±2σ")
                        else:
                            cap_parts.append("双指标时不显示标准差轨道")
                        cap_parts.append("右侧副轴叠加标的指数 K 线（可勾选关闭）")
                        st.caption("；".join(cap_parts) + "。")

                        with st.expander(f"📋 最新日分标的明细（{latest_d.strftime('%Y-%m-%d')}）"):
                            show_df = today_df[["underlying", "call_volume", "put_volume", "pcr_volume",
                                                "call_oi", "put_oi", "pcr_oi"]].copy()
                            for c in ["pcr_volume", "pcr_oi"]:
                                show_df[c] = show_df[c].round(3)
                            st.dataframe(show_df.rename(columns={
                                "underlying": "标的", "call_volume": "认购成交量",
                                "put_volume": "认沽成交量", "pcr_volume": "成交量PCR",
                                "call_oi": "认购持仓", "put_oi": "认沽持仓",
                                "pcr_oi": "持仓量PCR"}), width="stretch", hide_index=True)
    except Exception as e:
        st.error(f"期权 PCR 模块加载失败：{type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# Tab ⑤ 股指期货升贴水
# ---------------------------------------------------------------------------
with tab_basis:
    try:
        with st.spinner(f"逐日回溯抓取近 {futures_lookback} 个自然日中金所行情…"):
            basis = dfetch.get_index_futures_basis_history(futures_lookback)
        show_data_sources([("股指期货升贴水", "get_index_futures_basis_history",
                            basis["date"].max() if not basis.empty else None)])
        if basis.empty:
            st.warning("未获取到股指期货数据。")
        else:
            latest_d = basis["date"].max()
            b_now = basis[basis["date"] == latest_d].set_index("variety")
            st.subheader(f"四大期指主力合约基差（{latest_d.strftime('%Y-%m-%d')}）")
            cards = st.columns(4)
            for col, v in zip(cards, ["IF", "IH", "IC", "IM"]):
                if v in b_now.index:
                    r = b_now.loc[v]
                    col.metric(f"{v} {r['variety_name']}｜{r['contract']}",
                               f"{r['basis_pct']:+.3f}%",
                               f"基差 {r['basis']:+.1f} 点")
            st.caption("基差率 =（期货主力收盘 − 现货收盘）/ 现货收盘。正值=升水（乐观），负值=贴水（对冲压力）。")
            for v in ["IF", "IH", "IC", "IM"]:
                if v in b_now.index:
                    st.caption(f"· {v}（{b_now.loc[v]['variety_name']}）：{ind.basis_signal(b_now.loc[v]['basis_pct'])}")

            ol_basis = st.multiselect("叠加指数走势（右轴）", COMMON_INDICES, key="overlay_basis")
            fig_b = go.Figure()
            palette = {"IF": C_BLUE, "IH": C_RED, "IC": C_GREEN, "IM": C_ORANGE}
            for v, g in basis.groupby("variety"):
                g = g.sort_values("date")
                fig_b.add_trace(go.Scatter(x=g["date"], y=g["basis_pct"], name=v,
                                           line=dict(color=palette.get(v, C_GRAY), width=2)))
            fig_b.add_hline(y=0, line_color=C_GRAY, line_dash="dash")
            fig_b.update_layout(height=440, margin=dict(l=10, r=10, t=30, b=10),
                                yaxis_title="主力基差率 %", hovermode="x unified",
                                legend=dict(orientation="h", y=1.08))
            add_index_overlay(fig_b, ol_basis, basis["date"].min(), basis["date"].max())
            st.plotly_chart(fig_b, width="stretch")
            st.caption("数据源：中金所官方日行情（成交量最大合约视为主力）＋新浪现货指数。")
    except Exception as e:
        st.error(f"股指期货基差模块加载失败：{type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# Tab ⑥ 黄金：上期所期限结构 + CFTC COT
# ---------------------------------------------------------------------------
with tab_gold:
    g1, g2 = st.columns([3, 2])
    with g1:
        try:
            with st.spinner("拉取上期所 AU 合约行情…"):
                au = dfetch.get_gold_curve_latest()
            show_data_sources([("黄金期货期限结构（上期所 AU）", "get_gold_curve_latest",
                                au["date"].max() if not au.empty else None)])
            if au.empty:
                st.warning("未获取到上期所黄金合约数据。")
            else:
                active = au[au["volume"] >= 1000]
                st.subheader(f"上期所黄金期限结构（{pd.to_datetime(au['date'].iloc[0]).strftime('%Y-%m-%d')}）")
                near = active.iloc[0]
                far = active.iloc[-1]
                spread = far["close"] - near["close"]
                ann = spread / near["close"] * 100
                m1, m2, m3 = st.columns(3)
                m1.metric("活跃近月", f"{near['contract']} {near['close']:.2f}")
                m2.metric("活跃远月", f"{far['contract']} {far['close']:.2f}")
                m3.metric("近-远月价差", f"{spread:+.2f} 元/克", f"{ann:+.2f}%（区间）")
                st.caption("远月高于近月为升水（Contango，常态）；若近月反超远月（Backwardation），"
                           "常对应现货紧张/挤仓，是视频中‘期货升贴水看资金布局’的国内品种映射。")

                fig_g = go.Figure()
                fig_g.add_trace(go.Scatter(
                    x=au["expire_date"], y=au["close"], mode="lines+markers+text",
                    name="AU 收盘价", line=dict(color=C_ORANGE, width=2.4),
                    marker=dict(size=np.clip(np.sqrt(au["volume"]) / 6, 6, 22),
                                color=C_BLUE, opacity=0.7),
                    text=au["contract"], textposition="top center",
                    hovertemplate="%{text}<br>到期月 %{x|%y-%m}<br>收盘 %{y:.2f}<extra></extra>"))
                fig_g.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10),
                                    yaxis_title="元 / 克", xaxis_title="合约到期月")
                st.plotly_chart(fig_g, width="stretch")
                st.caption("圆点大小≈该合约当日成交量；数据源：上期所官方日行情。")
        except Exception as e:
            st.error(f"黄金期限结构模块加载失败：{type(e).__name__}: {e}")

    with g2:
        try:
            with st.spinner(f"拉取近 {cot_years} 年 CFTC COT 报告（周频）…"):
                cot = dfetch.get_cot_gold(cot_years)
            show_data_sources([("COMEX 黄金 CFTC COT 持仓", "get_cot_gold",
                                cot["date"].max() if not cot.empty else None)])
            if cot.empty:
                st.warning("未获取到 CFTC 持仓数据。")
            else:
                c = cot.iloc[-1]
                st.subheader("COMEX 黄金 COT｜管理基金（Managed Money）")
                n1, n2, n3 = st.columns(3)
                n1.metric("多头（张）", f"{c['mm_long']:,.0f}")
                n2.metric("空头（张）", f"{c['mm_short']:,.0f}")
                n3.metric("净持仓（张）", f"{c['mm_net']:,.0f}")
                z = float(((cot["mm_net"] - cot["mm_net"].mean())
                           / cot["mm_net"].std()).iloc[-1])
                st.caption(f"截至 {c['date'].strftime('%Y-%m-%d')}（周频）；净持仓 z-score = {z:.2f}，"
                           f"{'投机净多处于极端高位（拥挤，警惕回调）' if z > 1.5 else '净多处于极端低位（恐慌，关注反转）' if z < -1.5 else '处于常态区间'}。")
                fig_c = go.Figure()
                colors = [C_GREEN if v >= 0 else C_RED for v in cot["mm_net"]]
                fig_c.add_trace(go.Bar(x=cot["date"], y=cot["mm_net"], name="净持仓",
                                       marker_color=colors, opacity=0.45, showlegend=True))
                fig_c.add_trace(go.Scatter(x=cot["date"], y=cot["mm_long"], name="多头",
                                           line=dict(color=C_RED, width=1.6)))
                fig_c.add_trace(go.Scatter(x=cot["date"], y=cot["mm_short"], name="空头",
                                           line=dict(color=C_GREEN, width=1.6)))
                fig_c.add_hline(y=0, line_color=C_GRAY)
                fig_c.update_layout(
                    height=430, margin=dict(l=10, r=10, t=30, b=10),
                    yaxis_title="合约张数", hovermode="x unified",
                    legend=dict(orientation="h", y=1.08),
                    xaxis=dict(
                        rangeslider=dict(visible=True, thickness=0.06),
                        rangeselector=dict(
                            buttons=list([
                                dict(count=6, label="6月", step="month", stepmode="backward"),
                                dict(count=1, label="1年", step="year", stepmode="backward"),
                                dict(count=2, label="2年", step="year", stepmode="backward"),
                                dict(step="all", label="全部"),
                            ]),
                        ),
                    ),
                )
                st.plotly_chart(fig_c, width="stretch")
                st.caption("数据源：CFTC Disaggregated Futures-only 报告（COMEX Gold 代码 088691）。")
        except Exception as e:
            st.error(f"CFTC COT 模块加载失败：{type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# Tab ⑦ 中信期货持仓（中金所股指期货 IF/IH/IC/IM 前 20 会员排名）
# ---------------------------------------------------------------------------
with tab_citic:
    try:
        citic = _fetch_with_progress(
            f"citic_{futures_lookback}",
            "正在逐日回溯拉取中信期货持仓（首次较慢，请稍候）…",
            dfetch.get_citic_position_history, futures_lookback)
        show_data_sources([("中信期货多空持仓", "get_citic_position_history",
                            citic["date"].max() if not citic.empty else None)])
        if citic.empty:
            st.warning("未获取到中信期货持仓数据（可能是非交易时段或中金所接口暂不可用）。")
        else:
            # 品种代码 -> 中文名称
            VARIETY_NAMES = {"IF": "沪深300", "IH": "上证50",
                             "IC": "中证500", "IM": "中证1000"}
            vpalette = {"IF": C_BLUE, "IH": C_RED, "IC": C_GREEN, "IM": C_ORANGE}

            latest_d = citic["date"].max()
            st.subheader(f"中信期货股指期货多空持仓（{latest_d.strftime('%Y-%m-%d')}，单位：手）")
            st.caption("品种对照：IF=沪深300 ｜ IH=上证50 ｜ IC=中证500 ｜ IM=中证1000")

            # ===== 筛选区：品种多选 + 到期月份多选 =====
            def _contract_month(c: str) -> str:
                return str(c)[2:6]  # IC2611 -> 2611

            citic = citic.copy()
            citic["exp_month"] = citic["contract"].map(_contract_month)

            all_varieties = sorted(citic["variety"].unique())
            all_months = sorted(citic["exp_month"].unique())
            cur_ym = pd.Timestamp.now().strftime("%y%m")  # 当前年月，标记已到期

            # 默认选最近 2 个存续中月份（已到期的不默认选中，避免下拉框全选后显示 no results）
            ongoing_months = [m for m in all_months if m >= cur_ym]
            default_months = ongoing_months[:2] if ongoing_months else all_months[:2]

            c1, c2 = st.columns(2)
            with c1:
                sel_varieties = st.multiselect(
                    "品种（可多选）", all_varieties, default=all_varieties,
                    format_func=lambda v: f"{v} {VARIETY_NAMES.get(v, '')}",
                    key="citic_sel_varieties")
            with c2:
                month_labels = {
                    m: f"20{m[:2]}-{m[2:]}（{'已到期' if m < cur_ym else '存续中'}）"
                    for m in all_months
                }
                sel_months = st.multiselect(
                    "到期月份（可多选）", all_months, default=default_months,
                    format_func=lambda m: month_labels[m],
                    help="默认展示最近 2 个存续中月份；已到期合约仍可勾选查看其历史",
                    key="citic_sel_months")

            if not sel_varieties or not sel_months:
                st.warning("请至少选择一个品种和一个到期月份。")
            else:
                df = citic[citic["variety"].isin(sel_varieties) &
                           citic["exp_month"].isin(sel_months)].copy()
                n_contracts = df["contract"].nunique()

                if n_contracts == 0:
                    st.info("当前筛选条件下无数据。")
                elif n_contracts == 1:
                    # ---------- 单合约视图 ----------
                    single = df.sort_values("date").copy()
                    contract_code = single["contract"].iloc[0]
                    variety = single["variety"].iloc[0]
                    vname = VARIETY_NAMES.get(variety, variety)
                    last = single.iloc[-1]
                    s_long = float(last["long_oi"])
                    s_short = float(last["short_oi"])
                    s_net = float(last["net_oi"])
                    is_expired = single["exp_month"].iloc[0] < cur_ym

                    st.markdown(f"##### {contract_code}（{variety} {vname}）"
                                f"{' · 已到期' if is_expired else ''}")
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("多单持仓", f"{s_long:,.0f}",
                              f"日增减 {last['long_oi_chg']:+,.0f}")
                    m2.metric("空单持仓", f"{s_short:,.0f}",
                              f"日增减 {last['short_oi_chg']:+,.0f}")
                    m3.metric("净持仓", f"{s_net:+,.0f}",
                              f"{'净多' if s_net >= 0 else '净空'}")
                    m4.metric("多空比", f"{s_long / s_short:.2f}" if s_short else "—")
                    m5.metric("多单/空单排名",
                              f"第{int(last['long_rank'])}名 / 第{int(last['short_rank'])}名"
                              if pd.notna(last["long_rank"]) and pd.notna(last["short_rank"]) else "—")
                    st.caption(ind.citic_net_signal(s_net, s_long, s_short))

                    # 单合约历史净持仓走势
                    st.markdown("##### 净持仓历史走势")
                    ol_citic = st.multiselect("叠加指数走势（右轴）", COMMON_INDICES, key="overlay_citic")
                    fig_sn = go.Figure()
                    fig_sn.add_trace(go.Scatter(
                        x=single["date"], y=single["net_oi"], name="净持仓",
                        mode="lines+markers",
                        line=dict(color=vpalette.get(variety, C_BLUE), width=2.4),
                        marker=dict(size=6)))
                    fig_sn.add_hline(y=0, line_color=C_GRAY, line_dash="dash")
                    fig_sn.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10),
                                         yaxis_title="净持仓（手）", hovermode="x unified",
                                         showlegend=False)
                    add_index_overlay(fig_sn, ol_citic, single["date"].min(), single["date"].max())
                    st.plotly_chart(fig_sn, width="stretch")

                    # 单合约多空对比
                    st.markdown("##### 多空持仓对比")
                    fig_sl = go.Figure()
                    fig_sl.add_trace(go.Scatter(x=single["date"], y=single["long_oi"],
                                                name="多单持仓",
                                                line=dict(color=C_RED, width=2.2)))
                    fig_sl.add_trace(go.Scatter(x=single["date"], y=single["short_oi"],
                                                name="空单持仓",
                                                line=dict(color=C_GREEN, width=2.2)))
                    fig_sl.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10),
                                         yaxis_title="持仓量（手）", hovermode="x unified",
                                         legend=dict(orientation="h", y=1.08))
                    st.plotly_chart(fig_sl, width="stretch")

                else:
                    # ---------- 多合约视图：汇总 + 按合约对比 ----------
                    agg = df.groupby(["date", "variety"], as_index=False).agg(
                        long_oi=("long_oi", "sum"),
                        long_oi_chg=("long_oi_chg", "sum"),
                        short_oi=("short_oi", "sum"),
                        short_oi_chg=("short_oi_chg", "sum"),
                        net_oi=("net_oi", "sum"),
                    )
                    now = agg[agg["date"] == latest_d].set_index("variety")

                    if now.empty:
                        st.info("所选合约均已到期，最新交易日无持仓数据，请查看下方历史走势。")
                    else:
                        total_long = now["long_oi"].sum()
                        total_short = now["short_oi"].sum()
                        total_net = total_long - total_short
                        h1, h2, h3, h4 = st.columns(4)
                        h1.metric("合计多单", f"{total_long:,.0f}",
                                  f"日增减 {now['long_oi_chg'].sum():+,.0f}")
                        h2.metric("合计空单", f"{total_short:,.0f}",
                                  f"日增减 {now['short_oi_chg'].sum():+,.0f}")
                        h3.metric("合计净持仓", f"{total_net:+,.0f}",
                                  f"{'净多' if total_net >= 0 else '净空'}")
                        h4.metric("多空比", f"{total_long / total_short:.2f}" if total_short else "—")
                        st.caption("净持仓 = 多单 − 空单；正值净多、负值净空。"
                                   + ind.citic_net_signal(total_net, total_long, total_short))

                    # 净持仓历史走势（每个合约一条线，同品种同色·不同线型）
                    st.markdown(f"##### 净持仓历史走势（{n_contracts} 个合约）")
                    ol_citic = st.multiselect("叠加指数走势（右轴）", COMMON_INDICES, key="overlay_citic")
                    fig_n = go.Figure()
                    dash_styles = ["solid", "dash", "dot", "dashdot",
                                   "longdash", "longdashdot"]
                    for i, (contract, g) in enumerate(
                            df.groupby("contract", sort=True)):
                        g = g.sort_values("date")
                        v = g["variety"].iloc[0]
                        fig_n.add_trace(go.Scatter(
                            x=g["date"], y=g["net_oi"],
                            name=f"{contract}（{VARIETY_NAMES.get(v, '')}）",
                            mode="lines",
                            line=dict(color=vpalette.get(v, C_GRAY), width=2.2,
                                      dash=dash_styles[i % len(dash_styles)])))
                    fig_n.add_hline(y=0, line_color=C_GRAY, line_dash="dash")
                    fig_n.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10),
                                        yaxis_title="净持仓（手）", hovermode="x unified",
                                        legend=dict(orientation="h", y=1.08))
                    add_index_overlay(fig_n, ol_citic, df["date"].min(), df["date"].max())
                    st.plotly_chart(fig_n, width="stretch")

                    # 多空持仓对比（按合约分别显示，跟着多选框变动）
                    st.markdown(f"##### 多空持仓对比（{n_contracts} 个合约）")
                    fig_ls = go.Figure()
                    for contract, g in df.groupby("contract", sort=True):
                        g = g.sort_values("date")
                        v = g["variety"].iloc[0]
                        color = vpalette.get(v, C_GRAY)
                        # 多单实线、空单虚线，同品种同色
                        fig_ls.add_trace(go.Scatter(
                            x=g["date"], y=g["long_oi"], name=f"{contract} 多单",
                            line=dict(color=color, width=2, dash="solid")))
                        fig_ls.add_trace(go.Scatter(
                            x=g["date"], y=g["short_oi"], name=f"{contract} 空单",
                            line=dict(color=color, width=2, dash="dash")))
                    fig_ls.update_layout(height=440, margin=dict(l=10, r=10, t=30, b=10),
                                         yaxis_title="持仓量（手）", hovermode="x unified",
                                         legend=dict(orientation="h", y=1.08))
                    st.plotly_chart(fig_ls, width="stretch")

                # 最新日分合约明细（仅展示筛选后的合约）
                with st.expander(f"📋 最新日分合约明细（{latest_d.strftime('%Y-%m-%d')}）"):
                    detail = df[df["date"] == latest_d][[
                        "variety", "contract", "long_oi", "long_oi_chg", "long_rank",
                        "short_oi", "short_oi_chg", "short_rank", "net_oi"]].copy()
                    if detail.empty:
                        st.caption("最新日无筛选合约的数据。")
                    else:
                        for c in ["long_oi", "long_oi_chg", "short_oi",
                                  "short_oi_chg", "net_oi"]:
                            detail[c] = detail[c].astype(int)
                        detail["品种"] = detail["variety"].map(
                            lambda x: f"{x} {VARIETY_NAMES.get(x, '')}")
                        st.dataframe(detail[[
                            "品种", "contract", "long_oi", "long_oi_chg", "long_rank",
                            "short_oi", "short_oi_chg", "short_rank", "net_oi"]].rename(
                            columns={
                                "contract": "合约",
                                "long_oi": "多单持仓", "long_oi_chg": "多单增减", "long_rank": "多单排名",
                                "short_oi": "空单持仓", "short_oi_chg": "空单增减", "short_rank": "空单排名",
                                "net_oi": "净持仓",
                            }), width="stretch", hide_index=True)
                        st.caption("多单/空单排名指中信在该合约多头/空头前 20 名中的名次；"
                                   "若未进入前 20，对应持仓按 0 计（实际小于第 20 名）。")

            st.caption("数据源：中金所每日前 20 会员持仓排名（成交量/持买单/持卖单）；"
                       "中信期货为国内最大期货经纪商，其代客持仓方向常被视为大资金情绪风向标。")
    except Exception as e:
        st.error(f"中信期货持仓模块加载失败：{type(e).__name__}: {e}")

st.divider()
st.caption("全部数据来自公开渠道（中国国债信息网、乐咕乐股、中证指数、新浪财经、shibor.org、"
           "上交所、深交所、中金所、上期所、CFTC），通过 AKShare / 官方文件获取。"
           "本看板仅为视频指标体系的学习复现，不构成任何投资建议。")
#（注：内容由AI生成）
