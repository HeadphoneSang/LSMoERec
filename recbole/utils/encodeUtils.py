from enum import Enum, IntEnum, auto, unique, Flag, IntFlag
from collections import defaultdict
import logging

# 配置日志
logger = logging.getLogger(__name__)

# 事件处理器注册表
_event_handlers = defaultdict(list)


class EventType(Enum):
    """事件类型枚举"""
    NOTICE_EVENT = auto()


class EventHandler:
    """事件处理器装饰器"""
    def __init__(self, event_type: EventType):
        self.event_type = event_type
    
    def __call__(self, func):
        register_handler(self.event_type, func)
        return func


def register_handler(event_type: EventType, handler_func):
    """注册事件处理器
    
    Args:
        event_type: 事件类型
        handler_func: 处理函数，应该接受(event_data, **kwargs)参数
    """
    if not callable(handler_func):
        raise ValueError("处理器必须是可调用的函数")
    
    _event_handlers[event_type].append(handler_func)
    logger.info(f"注册处理器 {handler_func.__name__} 到事件 {event_type.name}")


def unregister_handler(event_type: EventType, handler_func):
    """注销事件处理器"""
    if handler_func in _event_handlers[event_type]:
        _event_handlers[event_type].remove(handler_func)
        logger.info(f"注销处理器 {handler_func.__name__} 从事件 {event_type.name}")


def get_handlers(event_type: EventType):
    """获取指定事件的所有处理器"""
    return _event_handlers[event_type]


def emit_event(event_type: EventType, event_data=None, **kwargs):
    """触发事件并执行所有注册的处理器
    
    Args:
        event_type: 事件类型
        event_data: 事件数据
        **kwargs: 额外参数
    """
    handlers = _event_handlers[event_type]
    if not handlers:
        logger.debug(f"事件 {event_type.name} 没有注册的处理器")
        return []
    
    results = []
    # logger.info(f"触发事件: {event_type.name}, 处理器数量: {len(handlers)}")
    
    for handler in handlers:
        try:
            result = handler(event_data, **kwargs)
            results.append(result)
            logger.debug(f"处理器 {handler.__name__} 执行成功")
        except Exception as e:
            logger.error(f"处理器 {handler.__name__} 执行失败: {e}")
            results.append(None)
    
    return results


def clear_handlers(event_type: EventType = None):
    """清空处理器
    
    Args:
        event_type: 如果指定则只清空该事件的处理器，否则清空所有
    """
    if event_type:
        _event_handlers[event_type].clear()
        logger.info(f"清空事件 {event_type.name} 的所有处理器")
    else:
        _event_handlers.clear()
        logger.info("清空所有事件处理器")


def dict_to_table_str(d, indent=0):
    """
    将嵌套字典格式化为文本表格字符串。
    如果是顶层调用，在最上方加上当前时间。
    当遇到 'timestamp' 属性时，不再递归其值。

    :param d: dict
    :param indent: 缩进层级
    :param is_root: 是否是顶层调用（自动设置，不用手动传）
    :return: str
    """
    if not isinstance(d, dict):
        raise TypeError(f"Expected dict, got {type(d)}")

    lines = []

    space = '  ' * indent
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{space}{key}:")
            lines.append(dict_to_table_str(value, indent + 1))
        else:
            lines.append(f"{space}{key}: {value}")

    return '\n'.join(lines)
