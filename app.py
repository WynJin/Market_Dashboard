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

with st.sidebar:
    st.header("⚙️ 参数设置")
    pe_name = st.selectbox("息差定价基准（指数）", list(INDEX_MAP.keys()), index=0)
    band_years = st.slider("轨道滚动窗口（年，月频样本）", 2, 8, 3)
    sigma_k = st.slider("高亮轨道倍数 σ", 1.0, 2.5, 2.0, step=0.5)
    lookback_days = st.slider("PCR / 期指回溯自然日数", 20, 90, 45,
                              help="交易所历史接口按天回溯，越大越慢（已缓存）")
    cot_years = st.slider("COT 持仓回溯年数", 1, 5, 3)
    if st.button("🔄 强制刷新全部数据", width="stretch"):
        st.cache_data.clear()
        st.toast("缓存已清空，重新拉取数据")
    st.caption("数据默认缓存 12~24 小时；首次加载 PCR/期指历史需逐日回溯，请耐心等待。")

sina_code, csindex_code, legu_name = INDEX_MAP[pe_name]

st.title("📊 宏观交易指标看板")
st.caption("股债息差（核心） · SHIBOR · 期权PCR · 股指期货升贴水 · 黄金COT与期限结构 ｜ 公开数据，仅供研究，不构成投资建议")

tab_core, tab_shibor, tab_pcr, tab_basis, tab_gold, tab_citic = st.tabs([
    "① 股债息差系统", "② 资金面 SHIBOR", "③ 期权 PCR", "④ 股指期货升贴水",
    "⑤ 黄金观察", "⑥ 中信期货持仓"])

# ---------------------------------------------------------------------------
# Tab ① 股债息差系统（核心）
# ---------------------------------------------------------------------------
with tab_core:
    try:
        with st.spinner("拉取国债收益率、指数 PE 与大盘行情…"):
            bond = dfetch.get_bond_10y("20100101")
            pe_m = dfetch.get_index_pe_monthly(legu_name)
            index_daily = dfetch.get_index_daily(sina_code)

        erp_df = ind.build_erp(pe_m, bond)
        window = band_years * 12
        erp_df = ind.add_std_bands(erp_df, "erp", window=window, n_sigma=sigma_k)
        erp_df["pct_rank"] = ind.rolling_percentile(erp_df["erp"], window=max(window, 60))
        last = erp_df.iloc[-1]
        sig, sig_desc = ind.erp_signal(last["erp"], last["band_upper"], last["band_lower"])

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("股债风险溢价 ERP", f"{last['erp']:.2f} pct",
                  f"盈利收益率 {last['earnings_yield']:.2f}% − 10Y国债 {last['bond_yield']:.2f}%")
        c2.metric("最新信号", sig)
        c3.metric("滚动历史分位", f"{last['pct_rank']:.0f}%" if pd.notna(last["pct_rank"]) else "—")
        c4.metric(f"{pe_name} 滚动PE(TTM)", f"{last['pe_ttm']:.2f}")
        c5.metric("数据截至", last["date"].strftime("%Y-%m"))

        st.info(f"**信号解读：** {sig_desc}。口径：ERP = 100/PE(TTM) − 10年期国债收益率（百分点）；"
                f"轨道为近 {band_years} 年滚动均值 ± {sigma_k}σ。触上轨=股票见底，触下轨=见顶（与视频规则一致）。")

        fig = go.Figure()
        # ±2σ / ±1σ 浅色背景轨
        fig.add_trace(go.Scatter(x=erp_df["date"], y=erp_df["upper2"], name="+2σ",
                                 line=dict(width=1, color=C_GRAY), opacity=0.5,
                                 line_dash="dot", showlegend=True))
        fig.add_trace(go.Scatter(x=erp_df["date"], y=erp_df["lower2"], name="-2σ",
                                 line=dict(width=1, color=C_GRAY), opacity=0.5,
                                 line_dash="dot", showlegend=True))
        fig.add_trace(go.Scatter(x=erp_df["date"], y=erp_df["upper1"], name="+1σ",
                                 line=dict(width=1, color=C_GRAY), opacity=0.8,
                                 line_dash="dash", showlegend=True))
        fig.add_trace(go.Scatter(x=erp_df["date"], y=erp_df["lower1"], name="-1σ",
                                 line=dict(width=1, color=C_GRAY), opacity=0.8,
                                 line_dash="dash", showlegend=True))
        fig.add_trace(go.Scatter(x=erp_df["date"], y=erp_df["mean"], name="滚动均值",
                                 line=dict(width=1.5, color=C_ORANGE)))
        fig.add_trace(go.Scatter(x=erp_df["date"], y=erp_df["erp"], name="股债息差 ERP（蓝线）",
                                 line=dict(width=2.6, color=C_BLUE)))
        # 高亮选定的 ±kσ 轨道
        fig.add_trace(go.Scatter(x=erp_df["date"], y=erp_df["band_upper"],
                                 name=f"上轨 +{sigma_k}σ",
                                 line=dict(width=2, color=C_GREEN), line_dash="dash"))
        fig.add_trace(go.Scatter(x=erp_df["date"], y=erp_df["band_lower"],
                                 name=f"下轨 −{sigma_k}σ",
                                 line=dict(width=2, color=C_RED), line_dash="dash"))
        # 大盘指数（次轴，红线，仅参考）
        idx_recent = index_daily[index_daily["date"] >= erp_df["date"].min()]
        fig.add_trace(go.Scatter(x=idx_recent["date"], y=idx_recent["close"],
                                 name=f"{pe_name}指数（右轴·仅参考）",
                                 line=dict(width=1.6, color=C_RED, dash="solid"),
                                 opacity=0.55, yaxis="y2"))
        fig.update_layout(
            height=560, hovermode="x unified",
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis=dict(title="ERP（百分点）"),
            yaxis2=dict(title=f"{pe_name}点位", overlaying="y", side="right",
                        showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
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
# Tab ② SHIBOR
# ---------------------------------------------------------------------------
with tab_shibor:
    try:
        with st.spinner("拉取 SHIBOR…"):
            shibor = dfetch.get_shibor()
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
        if picked:
            fig_h = go.Figure()
            palette = [C_BLUE, C_RED, C_GREEN, C_ORANGE, "#7a3e9d", C_GRAY, "#0f8b8d", "#b5651d"]
            for i, t in enumerate(picked):
                fig_h.add_trace(go.Scatter(x=shibor["date"], y=shibor[t], name=t,
                                           line=dict(width=1.8, color=palette[i % len(palette)])))
            fig_h.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                                yaxis_title="SHIBOR %", hovermode="x unified")
            st.plotly_chart(fig_h, width="stretch")
        st.caption("数据源：shibor.org（全国银行间同业拆借中心）。短端利率飙升=资金面紧张，"
                   "视频中用于观察大资金流动性。")
    except Exception as e:
        st.error(f"SHIBOR 模块加载失败：{type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# Tab ③ 期权 PCR
# ---------------------------------------------------------------------------
with tab_pcr:
    try:
        with st.spinner(f"逐日回溯抓取近 {lookback_days} 个自然日期权统计（交易所接口，首次较慢）…"):
            pcr = dfetch.get_option_pcr_history(lookback_days)
        if pcr.empty:
            st.warning("未获取到期权数据（可能是非交易时段或交易所接口暂不可用）。")
        else:
            latest_d = pcr["date"].max()
            today_df = pcr[pcr["date"] == latest_d].copy()
            # 全市场合成 PCR
            agg = pcr.groupby("date", as_index=False).agg(
                call_volume=("call_volume", "sum"), put_volume=("put_volume", "sum"),
                call_oi=("call_oi", "sum"), put_oi=("put_oi", "sum"))
            agg["pcr_volume"] = agg["put_volume"] / agg["call_volume"]
            agg["pcr_oi"] = agg["put_oi"] / agg["call_oi"]
            a = agg.iloc[-1]
            st.subheader(f"全市场 ETF 期权 PCR（{latest_d.strftime('%Y-%m-%d')}）")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("成交量 PCR", f"{a['pcr_volume']:.3f}")
            k2.metric("持仓量 PCR", f"{a['pcr_oi']:.3f}")
            k3.metric("当日认购成交", f"{a['call_volume']/1e4:.0f} 万张")
            k4.metric("当日认沽成交", f"{a['put_volume']/1e4:.0f} 万张")
            st.caption("成交量 PCR 反映当日对冲/投机情绪；持仓量 PCR 反映中期防御仓位。"
                       + ind.pcr_signal(a["pcr_volume"]))

            show_df = today_df[["underlying", "call_volume", "put_volume", "pcr_volume",
                                "call_oi", "put_oi", "pcr_oi"]].copy()
            for c in ["pcr_volume", "pcr_oi"]:
                show_df[c] = show_df[c].round(3)
            st.dataframe(show_df.rename(columns={
                "underlying": "标的", "call_volume": "认购成交量", "put_volume": "认沽成交量",
                "pcr_volume": "成交量PCR", "call_oi": "认购持仓", "put_oi": "认沽持仓",
                "pcr_oi": "持仓量PCR"}), width="stretch", hide_index=True)

            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=agg["date"], y=agg["pcr_volume"],
                                       name="成交量 PCR", line=dict(color=C_BLUE, width=2.4)))
            fig_p.add_trace(go.Scatter(x=agg["date"], y=agg["pcr_oi"],
                                       name="持仓量 PCR", line=dict(color=C_ORANGE, width=2)))
            fig_p.add_hline(y=1.0, line_dash="dash", line_color=C_GRAY)
            fig_p.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                                yaxis_title="Put / Call", hovermode="x unified",
                                legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig_p, width="stretch")
            st.caption("数据源：上交所 / 深交所官方股票期权日统计（50ETF、300ETF、500ETF、创业板、科创50 等）。")
    except Exception as e:
        st.error(f"期权 PCR 模块加载失败：{type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# Tab ④ 股指期货升贴水
# ---------------------------------------------------------------------------
with tab_basis:
    try:
        with st.spinner(f"逐日回溯抓取近 {lookback_days} 个自然日中金所行情…"):
            basis = dfetch.get_index_futures_basis_history(lookback_days)
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
            st.plotly_chart(fig_b, width="stretch")
            st.caption("数据源：中金所官方日行情（成交量最大合约视为主力）＋新浪现货指数。")
    except Exception as e:
        st.error(f"股指期货基差模块加载失败：{type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# Tab ⑤ 黄金：上期所期限结构 + CFTC COT
# ---------------------------------------------------------------------------
with tab_gold:
    g1, g2 = st.columns([3, 2])
    with g1:
        try:
            with st.spinner("拉取上期所 AU 合约行情…"):
                au = dfetch.get_gold_curve_latest()
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
                fig_c.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10),
                                    yaxis_title="合约张数", hovermode="x unified",
                                    legend=dict(orientation="h", y=1.08))
                st.plotly_chart(fig_c, width="stretch")
                st.caption("数据源：CFTC Disaggregated Futures-only 报告（COMEX Gold 代码 088691）。")
        except Exception as e:
            st.error(f"CFTC COT 模块加载失败：{type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# Tab ⑥ 中信期货持仓（中金所股指期货 IF/IH/IC/IM 前 20 会员排名）
# ---------------------------------------------------------------------------
with tab_citic:
    try:
        with st.spinner(f"逐日回溯抓取近 {lookback_days} 个自然日中信期货持仓排名（中金所，首次较慢）…"):
            citic = dfetch.get_citic_position_history(lookback_days)
        if citic.empty:
            st.warning("未获取到中信期货持仓数据（可能是非交易时段或中金所接口暂不可用）。")
        else:
            # 按品种汇总（同一品种下所有合约相加）
            agg = citic.groupby(["date", "variety"], as_index=False).agg(
                long_oi=("long_oi", "sum"),
                long_oi_chg=("long_oi_chg", "sum"),
                short_oi=("short_oi", "sum"),
                short_oi_chg=("short_oi_chg", "sum"),
                net_oi=("net_oi", "sum"),
            )
            latest_d = agg["date"].max()
            now = agg[agg["date"] == latest_d].set_index("variety")

            st.subheader(f"中信期货股指期货多空持仓（{latest_d.strftime('%Y-%m-%d')}，单位：手）")
            # 全品种合计
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

            # 分品种卡片
            st.markdown("##### 分品种明细")
            vpalette = {"IF": C_BLUE, "IH": C_RED, "IC": C_GREEN, "IM": C_ORANGE}
            vcards = st.columns(len(now))
            for col, v in zip(vcards, now.index):
                r = now.loc[v]
                col.metric(
                    f"{v}",
                    f"净 {r['net_oi']:+,.0f}",
                    f"多 {r['long_oi']:,.0f} / 空 {r['short_oi']:,.0f}",
                )
                col.caption(f"多单增减 {r['long_oi_chg']:+,.0f} ｜ 空单增减 {r['short_oi_chg']:+,.0f}")

            # 历史净持仓走势
            st.markdown("##### 净持仓历史走势（分品种）")
            fig_n = go.Figure()
            for v, g in agg.groupby("variety"):
                g = g.sort_values("date")
                fig_n.add_trace(go.Scatter(
                    x=g["date"], y=g["net_oi"], name=v, mode="lines",
                    line=dict(color=vpalette.get(v, C_GRAY), width=2.2)))
            fig_n.add_hline(y=0, line_color=C_GRAY, line_dash="dash")
            fig_n.update_layout(height=440, margin=dict(l=10, r=10, t=30, b=10),
                                yaxis_title="净持仓（手）", hovermode="x unified",
                                legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig_n, width="stretch")

            # 多空持仓双线对比（合计）
            st.markdown("##### 多空持仓对比（四品种合计）")
            tot = agg.groupby("date", as_index=False).agg(
                long_oi=("long_oi", "sum"), short_oi=("short_oi", "sum"))
            fig_ls = go.Figure()
            fig_ls.add_trace(go.Scatter(x=tot["date"], y=tot["long_oi"], name="多单持仓",
                                        line=dict(color=C_RED, width=2.2)))
            fig_ls.add_trace(go.Scatter(x=tot["date"], y=tot["short_oi"], name="空单持仓",
                                        line=dict(color=C_GREEN, width=2.2)))
            fig_ls.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                                 yaxis_title="持仓量（手）", hovermode="x unified",
                                 legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig_ls, width="stretch")

            # 最新日分合约明细
            with st.expander(f"📋 最新日分合约明细（{latest_d.strftime('%Y-%m-%d')}）"):
                detail = citic[citic["date"] == latest_d][[
                    "variety", "contract", "long_oi", "long_oi_chg", "long_rank",
                    "short_oi", "short_oi_chg", "short_rank", "net_oi"]].copy()
                for c in ["long_oi", "long_oi_chg", "short_oi", "short_oi_chg", "net_oi"]:
                    detail[c] = detail[c].astype(int)
                st.dataframe(detail.rename(columns={
                    "variety": "品种", "contract": "合约",
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
