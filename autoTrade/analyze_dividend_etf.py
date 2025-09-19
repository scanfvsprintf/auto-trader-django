#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析中证红利ETF(515080)的历史数据，模拟组合回测
"""

import os
import sys
import django
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 设置Django环境
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'autoTrade.settings')
django.setup()

from common.models import FundDailyQuotes, FundInfo

def analyze_dividend_etf():
    """分析中证红利ETF的历史数据"""
    
    print("开始分析中证红利ETF...")
    
    # 查询中证红利ETF信息
    try:
        etf_info = FundInfo.objects.filter(fund_code='sh.515080').first()
        if not etf_info:
            print("未找到中证红利ETF(515080)的信息")
            # 尝试查询所有基金
            all_funds = FundInfo.objects.all()[:20]
            print("数据库中的基金列表:")
            for fund in all_funds:
                print(f"  {fund.fund_code}: {fund.fund_name} (类型: {fund.fund_type})")
            
            # 尝试查询包含"红利"的基金
            dividend_funds = FundInfo.objects.filter(fund_name__icontains='红利')[:10]
            print("\n包含'红利'的基金:")
            for fund in dividend_funds:
                print(f"  {fund.fund_code}: {fund.fund_name}")
            return
        
        print(f"ETF信息: {etf_info.fund_name} ({etf_info.fund_code})")
        print(f"基金类型: {etf_info.fund_type}")
        
    except Exception as e:
        print(f"查询ETF信息失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 查询最近5年的历史数据
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365*5)
    
    print(f"\n查询日期范围: {start_date} 到 {end_date}")
    
    try:
        # 查询不复权数据
        raw_quotes = FundDailyQuotes.objects.filter(
            fund_code_id='sh.515080',
            trade_date__gte=start_date,
            trade_date__lte=end_date
        ).order_by('trade_date').values('trade_date', 'open', 'close', 'hfq_close')
        
        print(f"查询到 {len(raw_quotes)} 条不复权数据")
        
        if len(raw_quotes) == 0:
            print("没有查询到历史数据")
            return
        
        # 转换为DataFrame
        df = pd.DataFrame(list(raw_quotes))
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        
        # 计算分红
        df['close_prev'] = df['close'].shift(1)
        df['hfq_close_prev'] = df['hfq_close'].shift(1)
        
        # 分红计算公式：前一日不复权价格 * (今日后复权/前日后复权) - 今日不复权价格
        df['dividend'] = df['close_prev'] * (df['hfq_close'] / df['hfq_close_prev']) - df['close']
        df['dividend'] = df['dividend'].clip(lower=0)  # 分红不能为负
        df['dividend'] = df['dividend'].where(df['dividend'] > 0.01, 0)  # 忽略极小的分红
        
        # 显示分红信息
        dividend_days = df[df['dividend'] > 0]
        print(f"\n检测到 {len(dividend_days)} 个分红日:")
        for date, row in dividend_days.iterrows():
            print(f"  {date.strftime('%Y-%m-%d')}: 每股分红 {row['dividend']:.4f}元")
        
        # 计算基本统计
        print(f"\n基本统计信息:")
        print(f"数据期间: {df.index.min().strftime('%Y-%m-%d')} 到 {df.index.max().strftime('%Y-%m-%d')}")
        print(f"总交易日数: {len(df)}")
        print(f"价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
        print(f"总分红: {df['dividend'].sum():.4f}元/股")
        
        # 模拟组合回测
        print(f"\n=== 模拟组合回测 ===")
        initial_capital = 100000
        etf_ratio = 0.5
        cash_ratio = 0.5
        
        # 初始建仓
        etf_value = initial_capital * etf_ratio
        cash_value = initial_capital * cash_ratio
        
        # 第一天买入ETF
        first_price = float(df['open'].iloc[0])
        etf_shares = int(etf_value / first_price / 100) * 100  # 按手买入
        actual_etf_cost = etf_shares * first_price
        remaining_cash = initial_capital - actual_etf_cost
        
        print(f"初始建仓:")
        print(f"  ETF价格: {first_price:.2f}")
        print(f"  买入ETF: {etf_shares}股, 成本: {actual_etf_cost:.2f}")
        print(f"  剩余现金: {remaining_cash:.2f}")
        
        # 模拟每日收益
        results = []
        total_dividend_received = 0
        
        for date, row in df.iterrows():
            etf_price = row['open']
            dividend_per_share = row['dividend']
            
            # 计算当日组合价值
            etf_value = etf_shares * etf_price
            total_assets = etf_value + remaining_cash
            
            # 处理分红
            if dividend_per_share > 0 and etf_shares > 0:
                dividend_amount = etf_shares * dividend_per_share
                remaining_cash += dividend_amount
                total_dividend_received += dividend_amount
                total_assets = etf_shares * etf_price + remaining_cash
            
            results.append({
                'date': date,
                'etf_price': etf_price,
                'etf_value': etf_value,
                'cash': remaining_cash,
                'total_assets': total_assets,
                'dividend': dividend_per_share,
                'dividend_received': dividend_amount if dividend_per_share > 0 else 0
            })
        
        # 计算最终结果
        final_assets = results[-1]['total_assets']
        total_return = (final_assets / initial_capital) - 1
        
        # 计算年化收益率
        duration_days = (df.index[-1] - df.index[0]).days
        duration_years = duration_days / 365.25
        annualized_return = (1 + total_return) ** (1 / duration_years) - 1 if duration_years > 0 else 0
        
        print(f"\n回测结果:")
        print(f"初始资金: {initial_capital:,.2f}元")
        print(f"最终资产: {final_assets:,.2f}元")
        print(f"总收益率: {total_return:.2%}")
        print(f"年化收益率: {annualized_return:.2%}")
        print(f"回测期间: {duration_years:.2f}年")
        print(f"总分红收入: {total_dividend_received:.2f}元")
        print(f"分红贡献收益率: {total_dividend_received/initial_capital:.2%}")
        
        # 分析分红对收益的贡献
        price_return = (results[-1]['etf_value'] / actual_etf_cost) - 1
        print(f"\n收益分解:")
        print(f"ETF价格收益: {price_return:.2%}")
        print(f"分红收益: {total_dividend_received/initial_capital:.2%}")
        print(f"现金部分收益: 0% (无风险收益)")
        
        # 检查是否有异常数据
        print(f"\n数据质量检查:")
        print(f"ETF价格最小值: {df['close'].min():.2f}")
        print(f"ETF价格最大值: {df['close'].max():.2f}")
        print(f"价格变化范围: {(df['close'].max()/df['close'].min()-1):.2%}")
        
        # 检查分红计算是否合理
        if len(dividend_days) > 0:
            avg_dividend = dividend_days['dividend'].mean()
            print(f"平均每股分红: {avg_dividend:.4f}元")
            print(f"最大每股分红: {dividend_days['dividend'].max():.4f}元")
        
        return results
        
    except Exception as e:
        print(f"分析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_dividend_etf()
