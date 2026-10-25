# -*- coding: utf-8 -*-
import sys, logging
logging.disable(logging.CRITICAL)
sys.path.insert(0, r'd:\Invest\Projects\Market_Dashboard(origin)')

from data_source.factory import get_default_manager
import pandas as pd

ds = get_default_manager()
pcr = ds.get_option_pcr_history(90)
print(f"总行数: {len(pcr)}")
if not pcr.empty:
    print(f"日期范围: {pcr['date'].min()} ~ {pcr['date'].max()}")
    print(f"唯一交易日数: {pcr['date'].nunique()}")
    print(f"\n各标的数据点数:")
    print(pcr.groupby('underlying')['date'].nunique().to_string())
else:
    print("无数据")
