#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试文件，验证事件系统是否正常工作
"""

import logging
from recbole.utils.encodeUtils import (
    EventType, EventHandler, register_handler, emit_event
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 测试处理器
@EventHandler(EventType.USER_LOGIN)
def test_login_handler(event_data, **kwargs):
    print(f"处理登录事件: {event_data}")
    return {"status": "login_success"}

@EventHandler(EventType.ITEM_CLICK)
def test_click_handler(event_data, **kwargs):
    print(f"处理点击事件: {event_data}")
    return {"status": "click_recorded"}


def test_event_system():
    """测试事件系统"""
    print("=== 测试事件系统 ===")
    
    # 测试登录事件
    login_data = {'user_id': 'test_user', 'timestamp': '2024-01-01'}
    results = emit_event(EventType.USER_LOGIN, login_data)
    print(f"登录事件结果: {results}")
    
    # 测试点击事件
    click_data = {'user_id': 'test_user', 'item_id': 'test_item'}
    results = emit_event(EventType.ITEM_CLICK, click_data)
    print(f"点击事件结果: {results}")
    
    print("测试完成！")


if __name__ == "__main__":
    test_event_system()