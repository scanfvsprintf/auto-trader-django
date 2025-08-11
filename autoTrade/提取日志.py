import re
import html
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# ==============================================================================
# 配置区域
# ==============================================================================
LOG_FILE_PATH = '回测简单日志.txt'
PLOT_OUTPUT_PATH = '资金变化图.png'
HTML_REPORT_PATH = '交易变化.html'


# ==============================================================================
# 1. 日志解析
# ==============================================================================
def parse_log_file(file_path):
    """
    解析日志文件，提取绘图和报告所需的数据。
    """
    # 用于存储每日资产
    asset_dates = []
    asset_values = []
    
    # 用于存储每日的日志块
    daily_logs = []
    
    # 用于计算每支股票的盈亏
    # 结构: {'sz.002364': {'spent': 1000, 'received': 1100, 'dividends': 10, 'name': '中恒电气'}}
    stock_profits = defaultdict(lambda: {'spent': 0, 'received': 0, 'dividends': 0, 'name': ''})

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            current_day_block = None
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 匹配新的一天
                day_match = re.search(r"模拟日: ([\d-]+)", line)
                if day_match:
                    if current_day_block:
                        daily_logs.append(current_day_block)
                    date_str = day_match.group(1)
                    current_day_block = {'date': date_str, 'logs': [line]}
                    continue
                
                if not current_day_block:
                    continue

                current_day_block['logs'].append(line)

                # 匹配总资产
                asset_match = re.search(r"总资产: ([\d.]+)", line)
                if asset_match:
                    asset_dates.append(current_day_block['date'])
                    asset_values.append(float(asset_match.group(1)))

                # 匹配买入操作
                buy_match = re.search(r"买入 (.+?)\((.+?)\).*?花费: ([\d.]+)", line)
                if buy_match:
                    name, code, cost = buy_match.groups()
                    stock_profits[code]['spent'] += float(cost)
                    if not stock_profits[code]['name']: # 首次记录股票名称
                        stock_profits[code]['name'] = name

                # 匹配卖出操作
                sell_match = re.search(r"卖出 (.+?) .*?收入: ([\d.]+)", line)
                if sell_match:
                    code, income = sell_match.groups()
                    stock_profits[code]['received'] += float(income)

                # 匹配分红事件
                dividend_match = re.search(r"持仓ID \d+ \((.+?)\) 获得分红 ([\d.]+)", line)
                if dividend_match:
                    code, dividend = dividend_match.groups()
                    stock_profits[code]['dividends'] += float(dividend)


            if current_day_block: # 添加最后一天的数据
                daily_logs.append(current_day_block)

    except FileNotFoundError:
        print(f"错误: 日志文件 '{file_path}' 未找到。")
        return None, None, None, None
    
    return asset_dates, asset_values, daily_logs, stock_profits

# ==============================================================================
# 2. 生成资金曲线图
# ==============================================================================
def generate_asset_plot(dates, assets, output_path):
    """
    使用matplotlib生成资金曲线图并保存。
    """
    if not dates or not assets:
        print("没有足够的资产数据来生成图表。")
        return

    print("正在生成资金变化图...")
    
    # 设置中文字体，以防乱码
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建一个较大尺寸的图形
    fig, ax = plt.subplots(figsize=(18, 9))

    ax.plot(dates, assets, marker='.', linestyle='-', color='b')

    # 设置图表标题和标签
    ax.set_title('策略回测资金曲线', fontsize=20)
    ax.set_xlabel('模拟日期', fontsize=14)
    ax.set_ylabel('总资产 (元)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)

    # 自动调整x轴标签以避免重叠
    fig.autofmt_xdate(rotation=45)
    
    # 格式化y轴为货币格式
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    # 确保布局紧凑，所有元素都可见
    plt.tight_layout()

    # 保存图表
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"资金变化图已成功保存到: {output_path}")

# ==============================================================================
# 3. 生成HTML报告
# ==============================================================================
def generate_html_report(daily_logs, stock_profits, output_path):
    """
    生成包含高亮日志和盈亏汇总的HTML报告。
    """
    if not daily_logs or not stock_profits:
        print("没有足够的数据来生成HTML报告。")
        return
        
    print("正在生成HTML报告...")

    # --- 计算并排序股票盈亏 ---
    profit_summary = []
    for code, data in stock_profits.items():
        total_profit = data['received'] + data['dividends'] - data['spent']
        profit_summary.append({
            'code': code,
            'name': data['name'] or '未知名称',
            'profit': total_profit
        })
    
    # 从大到小排序
    sorted_profits = sorted(profit_summary, key=lambda x: x['profit'], reverse=True)

    # --- 构建HTML内容 ---
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>回测交易日志报告</title>
        <style>
            body { font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; line-height: 1.6; margin: 20px; background-color: #f4f4f4; color: #333; }
            h1, h2 { color: #0056b3; border-bottom: 2px solid #0056b3; padding-bottom: 10px; }
            .container { max-width: 1200px; margin: auto; background: #fff; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            .summary-table { width: 100%; border-collapse: collapse; margin-bottom: 30px; }
            .summary-table th, .summary-table td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            .summary-table th { background-color: #007bff; color: white; }
            .summary-table tr:nth-child(even) { background-color: #f2f2f2; }
            .profit { color: #d9534f; } /* 红色 */
            .loss { color: #5cb85c; } /* 绿色 */
            .day-block { border: 1px solid #ccc; border-radius: 5px; margin-bottom: 20px; padding: 15px; background-color: #fafafa; }
            .day-block h3 { margin-top: 0; color: #555; }
            .log-entry { font-family: 'Courier New', Courier, monospace; white-space: pre-wrap; word-wrap: break-word; }
            .log-profit-sell { color: #d9534f; font-weight: bold; } /* 止盈卖出 - 红色 */
            .log-stop-loss { color: #5cb85c; font-weight: bold; } /* 止损卖出 - 绿色 */
        </style>
    </head>
    <body>
        <div class="container">
            <h1>回测交易日志报告</h1>
            
            <h2>各股盈亏汇总 (从高到低)</h2>
            <table class="summary-table">
                <thead>
                    <tr>
                        <th>排名</th>
                        <th>股票名称</th>
                        <th>股票代码</th>
                        <th>总盈亏 (元)</th>
                    </tr>
                </thead>
                <tbody>
    """

    # 填充盈亏汇总表格
    for i, item in enumerate(sorted_profits):
        profit_class = 'profit' if item['profit'] >= 0 else 'loss'
        html_content += f"""
                    <tr>
                        <td>{i + 1}</td>
                        <td>{html.escape(item['name'])}</td>
                        <td>{html.escape(item['code'])}</td>
                        <td class="{profit_class}">{item['profit']:.2f}</td>
                    </tr>
        """
    
    html_content += """
                </tbody>
            </table>

            <h2>详细日志记录</h2>
    """

    # 填充详细日志
    for day in daily_logs:
        html_content += f"""
            <div class="day-block">
                <h3>{html.escape(day['date'])}</h3>
                <div class="log-entry">
        """
        for log_line in day['logs']:
            escaped_line = html.escape(log_line)
            if '触发止盈' in log_line or '止盈卖出' in log_line:
                html_content += f'<span class="log-profit-sell">{escaped_line}</span>\n'
            elif '触发止损' in log_line or '止损卖出' in log_line:
                html_content += f'<span class="log-stop-loss">{escaped_line}</span>\n'
            else:
                html_content += f'{escaped_line}\n'
        html_content += """
                </div>
            </div>
        """

    html_content += """
        </div>
    </body>
    </html>
    """

    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"HTML报告已成功保存到: {output_path}")

# ==============================================================================
# 主执行函数
# ==============================================================================
def main():
    """主函数，协调所有操作。"""
    print("开始处理回测日志...")
    symbol=['==================== 模拟日','触发','[回测] 卖出',' [回测] 买入','总资产: ','获得分红','风控价格','M(t)','确定唯一买入标的']
    log_prefix_pattern = re.compile(r"^[A-Z]+\s+[\d\-\s:,]+\s+\w+\s+\d+\s+\d+\s+(.*)")
 
    clean_log_content = []
    
    # 使用 'with' 语句能更好地处理文件，并且更安全
    # 注意：如果你的原始日志文件不是gbk编码，请修改这里的 encoding
    try:
        with open('logs/django.log', "r", encoding="gbk") as f:
            for line in f:
                # 检查该行是否包含我们关心的关键字
                should_keep = False
                for keyword in symbol:
                    if keyword in line:
                        should_keep = True
                        break
                
                if should_keep:
                    # 去掉行首行尾的空白字符
                    stripped_line = line.strip()
                    
                    # 尝试用正则表达式匹配并去除前缀
                    match = log_prefix_pattern.match(stripped_line)
                    if match:
                        # 如果匹配成功，只取括号里捕获的内容 (核心日志)
                        clean_line = match.group(1)
                    else:
                        # 如果不匹配 (比如 "==== 模拟日..." 这种行)，就保留原样
                        clean_line = stripped_line
                    
                    clean_log_content.append(clean_line)
    except FileNotFoundError:
        print("错误: 原始日志文件 'logs/django.log' 未找到。")
        return
    except Exception as e:
        print(f"读取原始日志时发生错误: {e}")
        return
 
    # 将处理过的、干净的日志内容写入到简单日志文件中
    with open(LOG_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(clean_log_content))

    # 1. 解析日志
    asset_dates, asset_values, daily_logs, stock_profits = parse_log_file(LOG_FILE_PATH)

    if asset_dates is None: # 如果解析失败
        print("日志处理终止。")
        return

    # 2. 生成图表
    generate_asset_plot(asset_dates, asset_values, PLOT_OUTPUT_PATH)

    # 3. 生成HTML报告
    generate_html_report(daily_logs, stock_profits, HTML_REPORT_PATH)
    
    print("\n所有任务完成！")
    print(f" - 图表文件: {os.path.abspath(PLOT_OUTPUT_PATH)}")
    print(f" - 报告文件: {os.path.abspath(HTML_REPORT_PATH)}")


if __name__ == '__main__':
    main()
main()