# -*- coding: utf-8 -*-
"""共享工具函数：日期规范化、交易日回溯等（被各 Adapter 复用）"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


def to_dt(s: pd.Series) -> pd.Series:
    """把各种格式的日期列统一转成 datetime64（不含时分秒）。"""
    return pd.to_datetime(s.astype(str), errors="coerce").dt.normalize()


def iter_recent_days(n_max: int = 45):
    """从今天（不含）向前生成 n_max 个自然日，跳过周六周日。"""
    d = date.today() - timedelta(days=1)
    yielded = 0
    while yielded < n_max:
        if d.weekday() < 5:  # 0=周一 ... 4=周五
            yield d
            yielded += 1
        d -= timedelta(days=1)


def iter_days_from(start_date: date):
    """从 start_date 到昨天（含），跳过周六周日。start_date 晚于昨天则返回空。"""
    end = date.today() - timedelta(days=1)
    d = start_date
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


# 新浪指数代码前缀：沪市 sh / 深市 sz
SINA_PREFIX = {"SH": "sh", "SZ": "sz"}


def code_to_sina(code: str) -> str:
    """把纯数字指数代码转成新浪格式，如 000300 -> sh000300，399006 -> sz399006"""
    code = str(code)
    return ("sh" if code.startswith("0") else "sz") + code


def code_to_ts(code: str) -> str:
    """把纯数字指数代码转成 tushare 格式，如 000300 -> 000300.SH，399006 -> 399006.SZ"""
    code = str(code)
    return code + (".SH" if code.startswith("0") else ".SZ")
