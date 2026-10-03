# -*- coding: utf-8 -*-
"""data_source 包：多数据源适配层（策略 + 适配器 + 工厂 + 责任链）"""
from .protocol import DataSource
from .factory import SourceManager, get_default_manager

__all__ = ["DataSource", "SourceManager", "get_default_manager"]
