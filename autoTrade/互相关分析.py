import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 设置matplotlib以正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是一个常用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# --- 1. 加载和准备数据 ---
def load_and_prepare_data(filepath):
    """加载CSV数据，并进行基本预处理"""
    df = pd.read_csv(filepath)
    
    # 将'日期'列转换为datetime对象，并设为索引
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.set_index('日期')
    
    # 按日期排序，确保时间序列是连续的
    df = df.sort_index()
    
    print("原始数据预览：")
    print(df.head())
    print("\n")
    
    return df

# --- 2. 计算日收益率 ---
def calculate_returns(df):
    """
    计算日收益率。在金融分析中，分析收益率比分析原始价格更常见。
    这有助于序列的平稳化，并关注相对变化。
    """
    # 使用 .pct_change() 计算百分比变化
    df_returns = df.pct_change()
    
    # m值本身可能就是变化率，如果m值已经是类似收益率的指标，可以跳过对m值的处理
    # 这里我们假设m值也需要处理，如果不需要，可以注释掉下面这行
    # df_returns['m值'] = df['m值'] # 如果m值本身就是变化率，直接使用原始值
    
    # 删除第一行，因为其没有前一天的值，计算结果为NaN
    df_returns = df_returns.dropna()
    
    print("日收益率数据预览：")
    print(df_returns.head())
    print("\n")
    
    return df_returns

# --- 3. 互相关分析核心函数 ---
def cross_correlation_analysis(series1, series2, max_lag):
    """
    计算两个时间序列在指定最大滞后范围内的互相关性。
    
    参数:
    series1 (pd.Series): 第一个时间序列 (例如 m值的变化)
    series2 (pd.Series): 第二个时间序列 (例如 沪深300的变化)
    max_lag (int): 要测试的最大正负滞后天数
    
    返回:
    lags (list): 滞后天数列表
    corrs (list): 对应的相关系数列表
    """
    lags = []
    corrs = []
    
    # 遍历从 -max_lag 到 +max_lag 的所有滞后期
    for lag in range(-max_lag, max_lag + 1):
        # 对 series1 进行移位操作
        # lag > 0: series1 向前移动 (用未来的值对齐现在的series2)，代表series2领先
        # lag < 0: series1 向后移动 (用过去的值对齐现在的series2)，代表series1领先
        shifted_series1 = series1.shift(lag)
        
        # 将两个序列对齐，并移除因移位产生的NaN值
        # 这会确保我们只在两个序列都有数据的日期上计算相关性
        temp_df = pd.concat([shifted_series1, series2], axis=1)
        temp_df.columns = ['s1_shifted', 's2']
        temp_df = temp_df.dropna()
        
        if not temp_df.empty:
            # 计算皮尔逊相关系数
            corr, _ = pearsonr(temp_df['s1_shifted'], temp_df['s2'])
            lags.append(lag)
            corrs.append(corr)
            
    return lags, corrs

def plot_and_interpret(lags, corrs, series1_name, series2_name):
    """绘制互相关图，解读结果，并生成txt报告"""
    
    # 找到最大相关性及其对应的滞后期
    max_corr = max(corrs)
    best_lag = lags[np.argmax(corrs)]
    
    # --- 生成解读文本 ---
    if best_lag > 0:
        interpretation_text = f"解读: {series2_name} 的变化趋势平均领先于 {series1_name} {best_lag} 天。"
    elif best_lag < 0:
        interpretation_text = f"解读: {series1_name} 的变化趋势平均领先于 {series2_name} {-best_lag} 天。"
    else:
        interpretation_text = f"解读: {series1_name} 和 {series2_name} 的变化趋势基本同步。"
    # --- 在控制台打印结果 ---
    print("--- 分析结果 ---")
    print(f"最大相关系数为: {max_corr:.4f}")
    print(f"对应的最佳滞后期为: {best_lag} 天")
    print(interpretation_text)
    print("----------------\n")
    # --- 生成并保存TXT报告 ---
    report_filename = '互相关分析.txt'
    report_content = f"""
互相关分析报告
================================
分析对象:
- 序列1: {series1_name}
- 序列2: {series2_name}
核心发现:
--------------------------------
- 最大相关系数: {max_corr:.4f}
- 最佳滞后期: {best_lag} 天
结论:
--------------------------------
{interpretation_text}
================================
报告生成于: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    try:
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"分析报告已成功保存为: {report_filename}")
    except Exception as e:
        print(f"保存报告失败: {e}")
    # --- 绘制互相关图 ---
    plt.figure(figsize=(12, 6))
    # 使用修改后的代码，移除 use_line_collection
    plt.stem(lags, corrs) 
    plt.axvline(best_lag, color='r', linestyle='--', label=f'最佳滞后期: {best_lag}天')
    plt.title(f'{series1_name} 与 {series2_name} 的互相关分析')
    plt.xlabel('滞后期 (天)')
    plt.ylabel('相关系数')
    plt.grid(True)
    plt.legend()
    plt.show()


# --- 主程序 ---
if __name__ == '__main__':

    # 1. 加载数据
    df_raw = load_and_prepare_data('m_value_csi300.csv')
    
    # 2. 计算收益率
    # 注意：如果你的'm值'本身就是一种变化率或指标，而不是价格，你可能不需要对它计算pct_change()
    # 在这种情况下，你需要调整 calculate_returns 函数
    df_returns = calculate_returns(df_raw)
    
    # 3. 执行互相关分析
    # 我们将分析 'm值' 与 '沪深300收盘指数' 的收益率之间的关系
    # 设置一个合理的最大滞后期，比如30个交易日
    max_lag_days = 30
    
    # 注意这里的参数顺序：
    # series1是m值，series2是沪深300。
    # 结果中的正滞后意味着沪深300领先，负滞后意味着m值领先。
    lags, corrs = cross_correlation_analysis(
        df_returns['m值'], 
        df_returns['沪深300收盘指数'], 
        max_lag_days
    )
    
    # 4. 可视化和解读
    plot_and_interpret(lags, corrs, 'm值', '沪深300指数')

