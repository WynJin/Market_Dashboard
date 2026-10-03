# -*- coding: utf-8 -*-
"""
tests/test_tushare_interfaces.py — Tushare 接口可用性实测脚本

逐一验证 TushareSource 的 11 个方法是否能通过代理成功返回数据，
并对"不支持"的接口（期权 PCR、CFTC）直接调用 tushare 原始 API 验证。
运行：python tests/test_tushare_interfaces.py
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

# 确保能 import 项目根目录
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from data_source.factory import TUSHARE_TOKEN, TUSHARE_API_URL
from data_source.tushare_source import TushareSource


def banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_method(src: TushareSource, name: str, call):
    """测试单个方法，打印结果摘要 + 范例数据行。"""
    print(f"\n▶ {name}")
    t0 = time.time()
    try:
        df = call()
        elapsed = time.time() - t0
        if df is None or len(df) == 0:
            print(f"  ⚠️  返回空数据 ({elapsed:.2f}s)")
        else:
            print(f"  ✅ 成功，行数={len(df)}，列={list(df.columns)} ({elapsed:.2f}s)")
            print(f"     最新日期: {df['date'].max() if 'date' in df.columns else 'N/A'}")
            # 展示最后 2 行范例数据
            sample = df.tail(2).to_string(index=False)
            print(f"     范例数据(末2行):\n{sample}")
        return True
    except NotImplementedError as e:
        print(f"  ⏭️  NotImplementedError: {e}")
        return False
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ❌ 失败: {type(e).__name__}: {e} ({elapsed:.2f}s)")
        traceback.print_exc()
        return False


def test_raw_api(pro, name: str, call):
    """直接调用 tushare 原始 API，用于验证"不支持"的接口是否真的没有。"""
    print(f"\n▶ [原始API] {name}")
    t0 = time.time()
    try:
        result = call()
        elapsed = time.time() - t0
        if result is None:
            print(f"  ⚠️  返回 None ({elapsed:.2f}s)")
        elif isinstance(result, pd.DataFrame) and len(result) == 0:
            print(f"  ⚠️  返回空 DataFrame ({elapsed:.2f}s)")
        elif isinstance(result, pd.DataFrame):
            print(f"  ✅ 成功，行数={len(result)}，列={list(result.columns)} ({elapsed:.2f}s)")
            sample = result.head(2).to_string(index=False)
            print(f"     范例数据(前2行):\n{sample}")
        else:
            print(f"  ✅ 返回: {result} ({elapsed:.2f}s)")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ❌ 失败: {type(e).__name__}: {e} ({elapsed:.2f}s)")
        return False


def main():
    banner(f"Tushare 接口可用性实测（api_url={TUSHARE_API_URL}）")

    src = TushareSource(TUSHARE_TOKEN, api_url=TUSHARE_API_URL)
    pro = src.pro

    # ------------------------------------------------------------------
    # 第一部分：测试 TushareSource 的 11 个方法
    # ------------------------------------------------------------------
    banner("第一部分：TushareSource 方法实测")

    results = {}
    results["get_bond_10y"] = test_method(src, "1. 10年期国债收益率",
                                           lambda: src.get_bond_10y("20240101"))
    results["get_index_pe_monthly"] = test_method(src, "2. 指数PE月频",
                                                  lambda: src.get_index_pe_monthly("沪深300"))
    results["get_index_pe_daily_csindex"] = test_method(src, "3. 指数PE日频",
                                                        lambda: src.get_index_pe_daily_csindex("000300"))
    results["get_index_daily"] = test_method(src, "4. 指数日线",
                                             lambda: src.get_index_daily("sh000300"))
    # results["get_shibor"] = test_method(src, "5. SHIBOR",
    #                                     lambda: src.get_shibor())
    results["get_option_pcr_history"] = test_method(src, "6. 期权PCR",
                                                    lambda: src.get_option_pcr_history(10))
    results["get_index_futures_basis_history"] = test_method(src, "7. 股指期货基差",
                                                             lambda: src.get_index_futures_basis_history(10))
    results["get_citic_position_history"] = test_method(src, "7b. 中信期货持仓",
                                                        lambda: src.get_citic_position_history(10))
    results["get_gold_curve_latest"] = test_method(src, "8. 黄金期限结构",
                                                   lambda: src.get_gold_curve_latest())
    results["get_gold_main_daily"] = test_method(src, "9. 黄金主力日线",
                                                 lambda: src.get_gold_main_daily("20240101", "20241231"))
    # results["get_cot_gold"] = test_method(src, "10. CFTC COT黄金",
    #                                       lambda: src.get_cot_gold(1))

    # ------------------------------------------------------------------
    # 第二部分：对"不支持"的接口直接调原始 tushare API 验证
    # ------------------------------------------------------------------
    banner("第二部分：原始 Tushare API 验证（PCR / CFTC / 持仓）")

    # 期权：opt_daily + opt_basic 是否可用
    test_raw_api(pro, "opt_daily(50ETF期权)",
                 lambda: pro.opt_daily(ts_code="10003720.SH", start_date="20240101", end_date="20240110"))
    test_raw_api(pro, "opt_basic(期权合约列表)",
                 lambda: pro.opt_basic(exchange="SSE", fields="ts_code,name,option_type,call_put,exercise_type"))

    # CFTC：tushare 是否有相关接口
    test_raw_api(pro, "fut_holding(IF持仓排名)",
                 lambda: pro.fut_holding(ts_code="IF2506.CFX", trade_date="20250601"))

    # 期货基差：fut_reversion 是否可用
    test_raw_api(pro, "fut_reversion(期货基差)",
                 lambda: pro.fut_reversion(ts_code="IF2506.CFX", trade_date="20250601"))

    # 期货合约列表
    test_raw_api(pro, "fut_basic(上期所AU合约)",
                 lambda: pro.fut_basic(exchange="SHF", fut_type="1"))

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    banner("测试汇总")
    ok = sum(1 for v in results.values() if v)
    print(f"通过: {ok}/{len(results)}")
    for name, passed in results.items():
        print(f"  {'✅' if passed else '❌'} {name}")


if __name__ == "__main__":
    main()
