#!/bin/bash

# ==============================================================================
# 自动部署脚本
# 功能: 停止服务 -> 拉取最新代码 -> 启动服务
# ==============================================================================

# --- 配置区 ---
# 请根据你的项目路径修改
PROJECT_DIR="
/root/autoTrade/auto-trader-django/autoTrade" 
# uwsgi的pid文件路径，请与你的uwsgi.ini配置保持一致
UWSGI_PID_FILE="/tmp/autoTrade.pid"
# Python解释器路径
PYTHON_EXECUTABLE="python3.10"
# Scheduler进程的唯一标识，用于pkill
SCHEDULER_CMD_PATTERN="manage.py run_scheduler"
# Scheduler的日志文件
SCHEDULER_LOG="nohup.out"

# --- 脚本核心 ---

# 设置 -e: 命令执行失败时立即退出脚本
# 设置 -o pipefail: 管道中的任何一个命令失败，整个管道都算失败
set -eo pipefail

# 函数：打印带颜色的信息
# 用法: log_info "这是一条信息"
log_info() {
    echo -e "\033[32m[INFO] $1\033[0m"
}

log_error() {
    echo -e "\033[31m[ERROR] $1\033[0m"
}

# 切换到项目目录
cd "$PROJECT_DIR" || { log_error "项目目录 '$PROJECT_DIR' 不存在，脚本退出。"; exit 1; }

log_info "当前工作目录: $(pwd)"

# --- 1. 停止现有服务 ---

log_info "--- 步骤 1: 正在停止 Scheduler 服务 ---"
# 使用pkill -f 精确查找并停止进程，@表示静默，即使找不到进程也不会报错退出
if pgrep -f "$SCHEDULER_CMD_PATTERN" > /dev/null; then
    pkill -f "$SCHEDULER_CMD_PATTERN"
    log_info "Scheduler 服务已发送停止信号，等待进程退出..."
    # 等待几秒钟让进程优雅退出
    sleep 3
    if pgrep -f "$SCHEDULER_CMD_PATTERN" > /dev/null; then
        log_error "Scheduler 服务未能正常停止，将强制终止！"
        pkill -9 -f "$SCHEDULER_CMD_PATTERN"
    fi
    log_info "Scheduler 服务已停止。"
else
    log_info "Scheduler 服务未在运行。"
fi


log_info "--- 步骤 2: 正在停止 uWSGI 服务 ---"
if [ -f "$UWSGI_PID_FILE" ]; then
    uwsgi --stop "$UWSGI_PID_FILE"
    # 等待uwsgi退出并删除pid文件
    sleep 2
    log_info "uWSGI 服务已停止。"
    # 确保pid文件被删除
    rm -f "$UWSGI_PID_FILE"
else
    log_info "未找到 uWSGI PID 文件，服务可能未在运行。"
fi

# --- 2. 清理旧日志 ---

log_info "--- 步骤 3: 清理旧的 nohup.out (如果存在) ---"
rm -f nohup.out

# --- 3. 更新代码 ---

log_info "--- 步骤 4: 从 Git 拉取最新代码 ---"
git pull
log_info "代码更新成功。"

# --- 4. 启动新服务 ---

log_info "--- 步骤 5: 正在启动 uWSGI 服务 ---"
uwsgi -i uwsgi.ini
# 检查uwsgi是否成功启动（通过PID文件）
sleep 2
if [ ! -f "$UWSGI_PID_FILE" ]; then
    log_error "uWSGI 启动失败！请检查 uwsgi.log。脚本退出。"
    exit 1
fi
log_info "uWSGI 服务启动成功，PID: $(cat $UWSGI_PID_FILE)"


log_info "--- 步骤 6: 正在后台启动 Scheduler 服务 ---"
# 将日志输出到指定文件，而不是nohup.out
nohup $PYTHON_EXECUTABLE manage.py run_scheduler > "$SCHEDULER_LOG" 2>&1 &

# 检查scheduler是否启动
sleep 2
if ! pgrep -f "$SCHEDULER_CMD_PATTERN" > /dev/null; then
    log_error "Scheduler 启动失败！请检查 '$SCHEDULER_LOG'。脚本退出。"
    # 停止刚刚启动的 uwsgi
    uwsgi --stop "$UWSGI_PID_FILE"
    exit 1
fi
log_info "Scheduler 服务启动成功，日志请查看 '$SCHEDULER_LOG'。"

log_info "=================================================="
log_info "部署完成！所有服务已成功启动。"
log_info "=================================================="
