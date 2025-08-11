import akshare as ak
df_normal = ak.stock_zh_a_hist(symbol='sh.600662', period="daily", start_date='2025-01-01', end_date='2025-08-08', adjust="")
df_hfq = ak.stock_zh_a_hist(symbol='sh.600662', period="daily", start_date='2025-01-01', end_date='2025-08-08', adjust="hfq")
df_normal_2023 = ak.stock_zh_a_hist(symbol='sh.600662', period="daily", start_date='2023-01-01', end_date='2025-08-08', adjust="")
df_hfq_2023 = ak.stock_zh_a_hist(symbol='sh.600662', period="daily", start_date='2023-01-01', end_date='2025-08-08', adjust="hfq")
print(df_hfq)
print(df_hfq_2023)

# 使用 .loc 通过索引来定位 '2025-08-07' 这一行
price_hfq_from_2025 = df_hfq.loc['2025-08-07']['收盘']
price_hfq_from_2023 = df_hfq_2023.loc['2025-08-07']['收盘']

# 打印两种方式获取的后复权价格及其差异
print(f"以 2025-01-01 为起点，获取的 8月7日 后复权收盘价: {price_hfq_from_2025}")
print(f"以 2023-01-01 为起点，获取的 8月7日 后复权收盘价: {price_hfq_from_2023}")
print(f"两种获取方式的价格差异: {price_hfq_from_2025 - price_hfq_from_2023}")
