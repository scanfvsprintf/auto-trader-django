# trade_manager/service/db_utils.py

import contextlib
import logging
import threading
from django.db import connections
from django.db.backends.signals import connection_created

logger = logging.getLogger(__name__)

# 使用线程局部存储来安全地在多线程环境中传递 schema 名称
_db_context = threading.local()

def backtest_schema_handler(sender, connection, **kwargs):
    """
    Django `connection_created` 信号的处理器。
    当一个新的数据库连接被创建时，此函数会被调用。
    它会检查当前线程是否在 `use_backtest_schema` 上下文中，
    如果是，则立即为这个新连接设置正确的 search_path。
    """
    if hasattr(_db_context, 'schema_name') and _db_context.schema_name:
        schema_name = _db_context.schema_name
        logger.debug(f"新数据库连接创建，为其设置 search_path -> {schema_name}, public")
        with connection.cursor() as cursor:
            # 使用参数化查询防止SQL注入
            cursor.execute("SET search_path TO %s, public;", [schema_name])

# 将信号处理器连接到 `connection_created` 信号
# dispatch_uid 确保即使代码被多次导入，信号处理器也只连接一次
connection_created.connect(backtest_schema_handler, dispatch_uid="set_backtest_search_path")

@contextlib.contextmanager
def use_backtest_schema(schema_name: str):
    """
    一个上下文管理器，用于在特定代码块内将所有数据库操作重定向到指定的 schema。

    用法:
    with use_backtest_schema('my_backtest_schema'):
        # 此处所有的 Django ORM 操作都会在 'my_backtest_schema' 中进行
        MyModel.objects.create(...)
    """
    # 进入 with 块时，设置线程局部变量
    _db_context.schema_name = schema_name
    # 强制关闭当前线程的现有连接，以确保下一个查询会创建一个新连接，从而触发信号处理器
    connections['default'].close()
    try:
        # 将控制权交还给 with 块内的代码
        yield
    finally:
        # 退出 with 块时（无论成功还是异常），清理线程局部变量
        if hasattr(_db_context, 'schema_name'):
            del _db_context.schema_name
        # 再次关闭连接，以便后续操作能恢复到默认的 search_path
        connections['default'].close()
        logger.debug("已退出回测 schema 上下文，恢复默认 search_path。")

