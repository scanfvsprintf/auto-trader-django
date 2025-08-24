import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# --- 配置区 ---

# 设置中文字体，以防图表和报告中的中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是黑体，适用于Windows
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # 适用于 Mac
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# --- 功能函数区 ---

def load_data(file_path):
    """
    加载数据文件，并自动处理常见的编码问题 (UTF-8, GBK)。
    """
    try:
        # 优先尝试UTF-8编码
        df = pd.read_csv(file_path, parse_dates=['日期'])
        print("文件以 UTF-8 编码成功加载。")
    except UnicodeDecodeError:
        try:
            # 如果UTF-8失败，尝试GBK编码
            df = pd.read_csv(file_path, parse_dates=['日期'], encoding='gbk')
            print("文件以 GBK 编码成功加载。")
        except Exception as e:
            print(f"尝试多种编码后，读取文件失败: {e}")
            return None
    except Exception as e:
        print(f"读取或解析文件时出错: {e}")
        return None
        
    df = df.set_index('日期').sort_index()
    return df

def analyze_m_value_predictiveness(df, horizons=[1,5, 20, 60]):
    """
    分析M值对未来沪深300指数收益和波动性的预测能力。
    """
    # 1. 计算未来的N日涨跌幅
    for h in horizons:
        df[f'fwd_return_{h}d'] = (df['沪深300收盘指数'].shift(-h) / df['沪深300收盘指数']) - 1

    df.dropna(inplace=True)

    # 2. 对M值进行分箱
    bins = np.arange(-1.0, 1.10, 0.1)
    # 确保标签和bins的长度匹配
    labels = [f"[{i:.1f}, {i+0.1:.1f})" for i in bins[:-1]]
    df['m_bin'] = pd.cut(df['m值'], bins=bins, labels=labels, right=False, include_lowest=True)

    print("M值分箱完成，开始进行分组统计...")

    # 3. 按M值分箱进行分组统计
    results = {}
    for h in horizons:
        agg_funcs = {
            f'fwd_return_{h}d': [
                ('平均涨跌幅(%)', lambda x: x.mean() * 100),
                ('涨跌幅标准差(%)', lambda x: x.std() * 100),
                ('样本数', 'count')
            ]
        }
        grouped_stats = df.groupby('m_bin').agg(agg_funcs)
        grouped_stats.columns = grouped_stats.columns.droplevel(0)
        results[h] = grouped_stats
    
    return results

def generate_text_report(results, file_path, df_info):
    """
    生成结构化的文本分析报告。
    """
    report_content = []
    report_content.append("="*80)
    report_content.append(" M(t) 指标有效性分析报告")
    report_content.append("="*80)
    report_content.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(f"数据源文件: {file_path}")
    report_content.append(f"数据时间范围: {df_info['start_date']} 到 {df_info['end_date']}")
    report_content.append(f"总样本数: {df_info['total_samples']}")
    report_content.append("\n")

    for horizon, data in results.items():
        report_content.append("-" * 80)
        report_content.append(f" 分析维度：未来 {horizon} 交易日市场表现")
        report_content.append("-" * 80)
        
        # 添加数据表
        report_content.append("【详细数据统计】\n")
        report_content.append(data.to_string())
        report_content.append("\n")

        # 添加自动解读
        report_content.append("【核心洞察解读】\n")
        
        # 1. 趋势性分析
        high_m_returns = data[data.index.str.startswith('[0.')]['平均涨跌幅(%)']
        low_m_returns = data[data.index.str.startswith('[-')]['平均涨跌幅(%)']
        
        if not high_m_returns.empty and not low_m_returns.empty:
            avg_high_return = high_m_returns.mean()
            avg_low_return = low_m_returns.mean()
            if avg_high_return > 0 and avg_low_return < 0 and avg_high_return > avg_low_return:
                report_content.append(f"  - 趋势预测性: 表现良好。M值为正时平均收益为 {avg_high_return:.2f}%，M值为负时平均收益为 {avg_low_return:.2f}%。M值越高，未来收益期望越高。")
            else:
                report_content.append(f"  - 趋势预测性: 表现不明显或存在反常。M值为正时平均收益 {avg_high_return:.2f}%，M值为负时平均收益 {avg_low_return:.2f}%。")
        
        # 2. 波动性/风险分析
        min_std_bin = data['涨跌幅标准差(%)'].idxmin()
        min_std_val = data['涨跌幅标准差(%)'].min()
        report_content.append(f"  - 风险指示性: 市场不确定性最低（最可预测）的区间出现在 M值 {min_std_bin}，其未来涨跌幅标准差仅为 {min_std_val:.2f}%。")

        # 3. 策略建议
        report_content.append("\n【策略应用建议】\n")
        if avg_high_return > 0.5: # 阈值可调
             report_content.append("  - 当 M 值较高时 (如 > 0.3)，未来市场上涨概率较大，适合加大【趋势动能】和【强势突破】策略的权重。")
        if avg_low_return < -0.5: # 阈值可调
             report_content.append("  - 当 M 值较低时 (如 < -0.3)，未来市场下跌风险较高，适合加大【质量防御】策略的权重。")
        if '[-0.2, -0.1)' in min_std_bin or '[0.0, 0.1)' in min_std_bin or '[-0.1, 0.0)' in min_std_bin:
             report_content.append("  - 当 M 值接近 0 时，若波动率显著降低，表明市场进入震荡期，适合加大【均值回归】策略的权重。")
        
        report_content.append("\n\n")

    report_filename = "M值分析报告.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_content))
    
    print(f"分析报告已成功保存至: {report_filename}")


def plot_analysis_results(results):
    """
    将分析结果可视化。
    """
    for horizon, data in results.items():
        fig, ax1 = plt.subplots(figsize=(14, 7))

        x_labels = data.index.astype(str)
        mean_returns = data['平均涨跌幅(%)']
        std_devs = data['涨跌幅标准差(%)']
        counts = data['样本数']

        color = 'tab:blue'
        ax1.set_xlabel('M值区间', fontsize=12)
        ax1.set_ylabel(f'未来{horizon}日平均涨跌幅 (%)', color=color, fontsize=12)
        bars = ax1.bar(x_labels, mean_returns, color=color, alpha=0.7)
        ax1.tick_params(axis='y', labelcolor=color)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
        ax1.axhline(0, color='gray', linewidth=0.8, linestyle='--')

        for i, bar in enumerate(bars):
            yval = bar.get_height()
            # 使用整数索引 i 从 counts 中获取正确的样本数值
            count_val = counts.iloc[i]
            ax1.text(bar.get_x() + bar.get_width()/2.0, yval + np.sign(yval)*0.1, f"n={int(count_val)}", ha='center', va='bottom', fontsize=8)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel(f'未来{horizon}日涨跌幅标准差 (%)', color=color, fontsize=12)
        ax2.plot(x_labels, std_devs, color=color, marker='o', linestyle='-')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.title(f'M值区间与未来 {horizon} 交易日市场表现关系', fontsize=16)
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.show()

# --- 主程序入口 ---
if __name__ == "__main__":
    # 1. 定义文件路径和分析参数
    csv_file_path = 'm_value_csi300.csv'  # <--- 请确保这是您的文件名
    analysis_horizons = [1,5, 20, 60]

    # 2. 加载数据
    df_raw = load_data(csv_file_path)

    if df_raw is not None:
        # 3. 提取数据信息用于报告
        df_info = {
            'start_date': df_raw.index.min().strftime('%Y-%m-%d'),
            'end_date': df_raw.index.max().strftime('%Y-%m-%d'),
            'total_samples': len(df_raw)
        }
        
        # 4. 执行核心分析
        analysis_results = analyze_m_value_predictiveness(df_raw.copy(), horizons=analysis_horizons)
        
        # 5. 生成文本报告
        generate_text_report(analysis_results, csv_file_path, df_info)
        
        # 6. 生成可视化图表
        plot_analysis_results(analysis_results)

