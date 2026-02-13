#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事件驱动系统使用示例
演示如何在推荐系统中使用事件处理器模式
"""

import logging
from recbole.utils.encodeUtils import (
    EventType, EventHandler, register_handler, 
    emit_event, get_handlers, unregister_handler
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 用户行为事件处理器 ====================
@EventHandler(EventType.USER_LOGIN)
def handle_user_login(event_data, **kwargs):
    """处理用户登录事件"""
    user_id = event_data.get('user_id')
    timestamp = event_data.get('timestamp')
    logger.info(f"用户 {user_id} 登录系统 at {timestamp}")
    
    # 可以在这里做推荐系统的初始化工作
    # 比如加载用户历史记录、更新用户画像等
    return {"status": "success", "message": f"欢迎用户 {user_id}"}


@EventHandler(EventType.USER_LOGOUT)
def handle_user_logout(event_data, **kwargs):
    """处理用户登出事件"""
    user_id = event_data.get('user_id')
    session_duration = kwargs.get('session_duration', 0)
    logger.info(f"用户 {user_id} 登出，会话时长: {session_duration}秒")
    
    # 清理会话数据、保存用户行为等
    return {"status": "success", "session_time": session_duration}


@EventHandler(EventType.ITEM_CLICK)
def handle_item_click(event_data, **kwargs):
    """处理物品点击事件"""
    user_id = event_data.get('user_id')
    item_id = event_data.get('item_id')
    timestamp = event_data.get('timestamp')
    
    logger.info(f"用户 {user_id} 点击物品 {item_id} at {timestamp}")
    
    # 更新用户兴趣模型
    # 记录交互行为用于推荐算法
    return {
        "user_id": user_id,
        "item_id": item_id,
        "action": "click_recorded",
        "score": 1.0
    }


@EventHandler(EventType.ITEM_PURCHASE)
def handle_item_purchase(event_data, **kwargs):
    """处理物品购买事件"""
    user_id = event_data.get('user_id')
    item_id = event_data.get('item_id')
    price = event_data.get('price', 0)
    
    logger.info(f"用户 {user_id} 购买物品 {item_id}, 价格: ¥{price}")
    
    # 强正反馈信号
    # 更新用户购买历史
    # 触发相关推荐
    return {
        "feedback_strength": "strong_positive",
        "recommend_next": True,
        "related_items": [f"related_{item_id}_{i}" for i in range(3)]
    }


# ==================== 数据统计事件处理器 ====================
@EventHandler(EventType.ITEM_CLICK)
def click_statistics_handler(event_data, **kwargs):
    """点击统计处理器 - 同一个事件可以有多个处理器"""
    item_id = event_data.get('item_id')
    # 这里可以实现点击次数统计、热门度计算等
    logger.info(f"统计处理器记录物品 {item_id} 的点击")
    return {"statistic_updated": True}


@EventHandler(EventType.RATING_SUBMIT)
def handle_rating_submit(event_data, **kwargs):
    """处理评分提交事件"""
    user_id = event_data.get('user_id')
    item_id = event_data.get('item_id')
    rating = event_data.get('rating')
    
    logger.info(f"用户 {user_id} 对物品 {item_id} 评分: {rating}")
    
    # 根据评分调整推荐策略
    if rating >= 4:
        feedback_type = "positive"
    elif rating <= 2:
        feedback_type = "negative"
    else:
        feedback_type = "neutral"
        
    return {
        "feedback_type": feedback_type,
        "rating_value": rating,
        "update_model": True
    }


# ==================== 搜索相关事件处理器 ====================
@EventHandler(EventType.SEARCH_QUERY)
def handle_search_query(event_data, **kwargs):
    """处理搜索查询事件"""
    user_id = event_data.get('user_id')
    query = event_data.get('query')
    timestamp = event_data.get('timestamp')
    
    logger.info(f"用户 {user_id} 搜索: '{query}' at {timestamp}")
    
    # 分析搜索意图
    # 记录搜索关键词
    # 触发基于搜索的推荐
    return {
        "query_processed": True,
        "intent_analyzed": len(query.split()),
        "recommendations": [f"result_{i}" for i in range(5)]
    }


# ==================== 自定义处理器注册 ====================
def custom_notification_handler(event_data, **kwargs):
    """自定义通知处理器"""
    message = event_data.get('message', '默认消息')
    priority = kwargs.get('priority', 'normal')
    logger.info(f"发送通知: {message} (优先级: {priority})")
    return {"notification_sent": True, "priority": priority}

# 注册自定义处理器
register_handler(EventType.NOTICE_EVENT, custom_notification_handler)


# ==================== 使用示例 ====================
def demo_user_journey():
    """演示完整的用户行为流程"""
    print("=== 用户行为事件演示 ===\n")
    
    # 1. 用户登录
    login_data = {
        'user_id': 'user_12345',
        'timestamp': '2024-01-15 10:30:00',
        'device': 'mobile'
    }
    results = emit_event(EventType.USER_LOGIN, login_data)
    print(f"登录处理结果: {results}\n")
    
    # 2. 用户搜索
    search_data = {
        'user_id': 'user_12345',
        'query': '智能手机',
        'timestamp': '2024-01-15 10:31:00'
    }
    results = emit_event(EventType.SEARCH_QUERY, search_data)
    print(f"搜索处理结果: {results}\n")
    
    # 3. 用户点击商品
    click_data = {
        'user_id': 'user_12345',
        'item_id': 'item_67890',
        'timestamp': '2024-01-15 10:32:00'
    }
    results = emit_event(EventType.ITEM_CLICK, click_data)
    print(f"点击处理结果: {results}\n")
    
    # 4. 用户评分
    rating_data = {
        'user_id': 'user_12345',
        'item_id': 'item_67890',
        'rating': 5,
        'timestamp': '2024-01-15 10:35:00'
    }
    results = emit_event(EventType.RATING_SUBMIT, rating_data)
    print(f"评分处理结果: {results}\n")
    
    # 5. 用户购买
    purchase_data = {
        'user_id': 'user_12345',
        'item_id': 'item_67890',
        'price': 2999,
        'timestamp': '2024-01-15 10:40:00'
    }
    results = emit_event(EventType.ITEM_PURCHASE, purchase_data)
    print(f"购买处理结果: {results}\n")
    
    # 6. 用户登出
    logout_data = {
        'user_id': 'user_12345',
        'timestamp': '2024-01-15 11:00:00'
    }
    results = emit_event(EventType.USER_LOGOUT, logout_data, session_duration=1800)
    print(f"登出处理结果: {results}\n")


def demo_system_events():
    """演示系统级别事件"""
    print("=== 系统事件演示 ===\n")
    
    # 发送系统通知
    notice_data = {
        'message': '系统维护通知',
        'type': 'maintenance'
    }
    results = emit_event(EventType.NOTICE_EVENT, notice_data, priority='high')
    print(f"通知处理结果: {results}\n")


def demo_handler_management():
    """演示处理器管理功能"""
    print("=== 处理器管理演示 ===\n")
    
    # 查看某个事件的处理器
    click_handlers = get_handlers(EventType.ITEM_CLICK)
    print(f"物品点击事件的处理器数量: {len(click_handlers)}")
    for handler in click_handlers:
        print(f"  - {handler.__name__}")
    
    # 注销处理器
    print(f"\n注销前处理器数量: {len(get_handlers(EventType.ITEM_CLICK))}")
    unregister_handler(EventType.ITEM_CLICK, click_statistics_handler)
    print(f"注销后处理器数量: {len(get_handlers(EventType.ITEM_CLICK))}")


def main():
    """主函数"""
    print("推荐系统事件驱动架构演示")
    print("=" * 50)
    
    demo_user_journey()
    demo_system_events()
    demo_handler_management()
    
    print("=" * 50)
    print("演示完成！")


if __name__ == "__main__":
    main()