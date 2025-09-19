<template>
  <div class="portfolio-result-panel">
    <el-card class="result-card">
      <div slot="header" class="result-header">
        <div class="header-title">
          <i class="el-icon-data-analysis"></i>
          <span>回测结果</span>
        </div>
        <div class="header-actions">
          <el-button :size="isMobile ? 'mini' : 'small'" type="primary" plain @click="$emit('show-config')">
            <i class="el-icon-setting" style="margin-right:4px"></i> 配置
          </el-button>
        </div>
      </div>
      
      <!-- 无结果状态 -->
      <div v-if="!backtestResult" class="empty-state">
        <div class="empty-content">
          <i class="el-icon-data-analysis empty-icon"></i>
          <div class="empty-text">请配置组合参数并开始回测</div>
          <el-button 
            type="primary" 
            :size="isMobile ? 'medium' : 'large'"
            @click="$emit('show-config')"
            class="start-config-btn"
          >
            <i class="el-icon-setting" style="margin-right:8px"></i>
            开始配置组合
          </el-button>
        </div>
      </div>
      
      <!-- 有结果状态 -->
      <div v-else class="result-content">
        <!-- 统计信息 -->
        <div class="stats-panel">
          <el-row :gutter="isMobile ? 8 : 16">
            <el-col :xs="12" :sm="6">
              <div class="stat-item">
                <div class="stat-value" :class="getReturnClass(backtestResult.statistics.total_return)">
                  {{ (backtestResult.statistics.total_return * 100).toFixed(2) }}%
                </div>
                <div class="stat-label">总收益率</div>
              </div>
            </el-col>
            <el-col :xs="12" :sm="6">
              <div class="stat-item">
                <div class="stat-value" :class="getReturnClass(backtestResult.statistics.annualized_return)">
                  {{ (backtestResult.statistics.annualized_return * 100).toFixed(2) }}%
                </div>
                <div class="stat-label">年化收益率</div>
              </div>
            </el-col>
            <el-col :xs="12" :sm="6">
              <div class="stat-item">
                <div class="stat-value negative">
                  {{ (backtestResult.statistics.max_drawdown * 100).toFixed(2) }}%
                </div>
                <div class="stat-label">最大回撤</div>
              </div>
            </el-col>
            <el-col :xs="12" :sm="6">
              <div class="stat-item">
                <div class="stat-value">
                  {{ backtestResult.statistics.sharpe_ratio.toFixed(2) }}
                </div>
                <div class="stat-label">夏普比率</div>
              </div>
            </el-col>
          </el-row>
        </div>
        
        <!-- 图表展示 -->
        <div class="chart-container">
          <el-tabs v-model="activeTab" @tab-click="handleTabClick">
            <el-tab-pane label="回测结果" name="capital">
              <div ref="capitalChart" class="chart-wrapper"></div>
            </el-tab-pane>
            <el-tab-pane label="最大回撤" name="drawdown">
              <div ref="drawdownChart" class="chart-wrapper"></div>
            </el-tab-pane>
            <el-tab-pane label="持有期分析" name="holding">
              <div ref="holdingChart" class="chart-wrapper"></div>
            </el-tab-pane>
          </el-tabs>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script>
import * as echarts from 'echarts'

export default {
  name: 'PortfolioResultPanel',
  props: {
    backtestResult: {
      type: Object,
      default: null
    },
    activeTab: {
      type: String,
      default: 'capital'
    },
    isMobile: {
      type: Boolean,
      default: false
    }
  },
  data() {
    return {
      // 图表实例
      _capitalChart: null,
      _drawdownChart: null,
      _holdingChart: null
    }
  },
  watch: {
    backtestResult: {
      handler(newVal) {
        if (newVal) {
          this.$nextTick(() => {
            this.drawCharts()
          })
        }
      },
      immediate: true
    },
    activeTab(newVal) {
      this.$nextTick(() => {
        this.resizeCharts()
      })
    }
  },
  mounted() {
    window.addEventListener('resize', this.resizeCharts)
  },
  beforeDestroy() {
    window.removeEventListener('resize', this.resizeCharts)
    this.disposeCharts()
  },
  methods: {
    handleTabClick(tab) {
      this.$emit('update:active-tab', tab.name)
    },
    
    // 获取收益率样式类
    getReturnClass(returnValue) {
      if (returnValue > 0) return 'positive'
      if (returnValue < 0) return 'negative'
      return 'neutral'
    },
    
    // 绘制图表
    drawCharts() {
      this.drawCapitalChart()
      this.drawDrawdownChart()
      this.drawHoldingChart()
    },
    
    // 绘制资金曲线图
    drawCapitalChart() {
      if (!this.backtestResult || !this.backtestResult.chart_data.capital_curve) return
      
      const chartElement = this.$refs.capitalChart
      if (!chartElement) return
      
      if (this._capitalChart) {
        this._capitalChart.dispose()
      }
      
      this._capitalChart = echarts.init(chartElement)
      
      const chartData = this.backtestResult.chart_data.capital_curve
      const option = {
        tooltip: {
          trigger: 'axis',
          formatter: (params) => {
            if (!params || params.length === 0) return ''
            const date = params[0].axisValueLabel || params[0].axisValue
            const value = params[0].data && params[0].data[1]
            return `${date}<br/>总资产: ¥${value ? value.toLocaleString() : '-'}`
          }
        },
        grid: {
          left: '6%',
          right: '12%',
          top: 24,
          bottom: 24,
          containLabel: true
        },
        xAxis: {
          type: 'time',
          boundaryGap: false,
          axisLabel: {
            color: '#6b7280',
            fontSize: this.isMobile ? 10 : 12
          },
          axisLine: {
            show: true,
            lineStyle: {
              color: '#6b7280'
            }
          },
          splitLine: {
            show: true,
            lineStyle: {
              color: '#e5e7eb'
            }
          }
        },
        yAxis: {
          type: 'value',
          name: '总资产(元)',
          nameLocation: 'end',
          nameGap: 6,
          axisLabel: {
            color: '#6b7280',
            formatter: (value) => {
              return (value / 10000).toFixed(1) + '万'
            },
            fontSize: this.isMobile ? 10 : 12
          },
          axisLine: {
            show: true,
            lineStyle: {
              color: '#6b7280'
            }
          },
          splitLine: {
            show: true,
            lineStyle: {
              color: '#e5e7eb'
            }
          }
        },
        series: [{
          type: 'line',
          data: chartData,
          smooth: true,
          showSymbol: false,
          lineStyle: {
            width: 1.2,
            color: '#10b981'
          },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(16,185,129,0.16)' },
                { offset: 1, color: 'rgba(16,185,129,0.00)' }
              ]
            }
          }
        }]
      }
      
      this._capitalChart.setOption(option)
    },
    
    // 绘制最大回撤图
    drawDrawdownChart() {
      if (!this.backtestResult || !this.backtestResult.chart_data.max_drawdown_curve) return
      
      const chartElement = this.$refs.drawdownChart
      if (!chartElement) return
      
      if (this._drawdownChart) {
        this._drawdownChart.dispose()
      }
      
      this._drawdownChart = echarts.init(chartElement)
      
      const chartData = this.backtestResult.chart_data.max_drawdown_curve
      const option = {
        tooltip: {
          trigger: 'axis',
          formatter: (params) => {
            if (!params || params.length === 0) return ''
            const date = params[0].axisValueLabel || params[0].axisValue
            const value = params[0].data && params[0].data[1]
            return `${date}<br/>回撤: ${value ? (value * 100).toFixed(2) + '%' : '-'}`
          }
        },
        grid: {
          left: '6%',
          right: '12%',
          top: 24,
          bottom: 24,
          containLabel: true
        },
        xAxis: {
          type: 'time',
          boundaryGap: false,
          axisLabel: {
            color: '#6b7280',
            fontSize: this.isMobile ? 10 : 12
          },
          axisLine: {
            show: true,
            lineStyle: {
              color: '#6b7280'
            }
          },
          splitLine: {
            show: true,
            lineStyle: {
              color: '#e5e7eb'
            }
          }
        },
        yAxis: {
          type: 'value',
          name: '回撤比例',
          nameLocation: 'end',
          nameGap: 6,
          axisLabel: {
            color: '#6b7280',
            formatter: (value) => {
              return (value * 100).toFixed(1) + '%'
            },
            fontSize: this.isMobile ? 10 : 12
          },
          axisLine: {
            show: true,
            lineStyle: {
              color: '#6b7280'
            }
          },
          splitLine: {
            show: true,
            lineStyle: {
              color: '#e5e7eb'
            }
          }
        },
        series: [{
          type: 'line',
          data: chartData,
          smooth: true,
          showSymbol: false,
          lineStyle: {
            width: 1.2,
            color: '#ef4444'
          },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(239,68,68,0.16)' },
                { offset: 1, color: 'rgba(239,68,68,0.00)' }
              ]
            }
          }
        }]
      }
      
      this._drawdownChart.setOption(option)
    },
    
    // 绘制持有期分析图
    drawHoldingChart() {
      if (!this.backtestResult || !this.backtestResult.chart_data.holding_period_analysis) return
      
      const chartElement = this.$refs.holdingChart
      if (!chartElement) return
      
      if (this._holdingChart) {
        this._holdingChart.dispose()
      }
      
      this._holdingChart = echarts.init(chartElement)
      
      const analysisData = this.backtestResult.chart_data.holding_period_analysis
      console.log('持有期分析数据:', analysisData)
      
      // 转换为echarts需要的格式 [x, y]
      const maxReturns = analysisData.map(item => [item.holding_days, item.max_return])
      const minReturns = analysisData.map(item => [item.holding_days, item.min_return])
      const winRates = analysisData.map(item => [item.holding_days, item.win_rate])
      
      console.log('原始数据样本:', analysisData.slice(0, 3))
      console.log('最大收益数据:', maxReturns.slice(0, 3))
      console.log('胜率数据:', winRates.slice(0, 3))
      
      // 检查数据有效性
      const checkData = (data, name) => {
        const valid = data.every(item => !isNaN(item[1]) && item[1] !== null && item[1] !== undefined)
        console.log(`${name}数据有效性:`, valid, '样本:', data.slice(0, 3))
        return valid
      }
      
      checkData(maxReturns, '最大收益')
      checkData(minReturns, '最小收益')
      checkData(winRates, '胜率')
      
      const option = {
        tooltip: {
          trigger: 'axis'
        },
        legend: {
          top: 4,
          left: 8,
          data: ['最大收益', '最小收益', '胜率'],
          textStyle: {
            fontSize: this.isMobile ? 10 : 12
          }
        },
        grid: {
          left: '6%',
          right: '12%',
          top: 46,
          bottom: 24,
          containLabel: true
        },
        xAxis: {
          type: 'value',
          name: '持有天数',
          min: 1,
          axisLabel: {
            color: '#6b7280',
            fontSize: this.isMobile ? 10 : 12,
            formatter: (value) => Math.round(value) + '天'
          },
          axisLine: {
            show: true,
            lineStyle: {
              color: '#6b7280'
            }
          },
          splitLine: {
            show: true,
            lineStyle: {
              color: '#e5e7eb'
            }
          }
        },
        yAxis: [
          {
            type: 'value',
            name: '收益率',
            position: 'left',
            nameLocation: 'end',
            nameGap: 6,
            axisLabel: {
              color: '#6b7280',
              formatter: (value) => (value * 100).toFixed(0) + '%'
            },
            axisLine: {
              show: true,
              lineStyle: {
                color: '#6b7280'
              }
            },
            splitLine: {
              show: true,
              lineStyle: {
                color: '#e5e7eb'
              }
            }
          },
          {
            type: 'value',
            name: '胜率',
            position: 'right',
            nameLocation: 'end',
            nameGap: 6,
            axisLabel: {
              color: '#7c3aed',
              formatter: (value) => (value * 100).toFixed(0) + '%'
            },
            axisLine: {
              show: true,
              lineStyle: {
                color: '#7c3aed'
              }
            },
            splitLine: {
              show: false
            }
          }
        ],
        series: [
          {
            name: '最大收益',
            type: 'line',
            data: maxReturns,
            smooth: true,
            showSymbol: false,
            lineStyle: {
              color: '#10b981',
              width: 1.2
            },
            areaStyle: {
              color: {
                type: 'linear',
                x: 0, y: 0, x2: 0, y2: 1,
                colorStops: [
                  { offset: 0, color: 'rgba(16,185,129,0.16)' },
                  { offset: 1, color: 'rgba(16,185,129,0.00)' }
                ]
              }
            }
          },
          {
            name: '最小收益',
            type: 'line',
            data: minReturns,
            smooth: true,
            showSymbol: false,
            lineStyle: {
              color: '#ef4444',
              width: 1.2
            },
            areaStyle: {
              color: {
                type: 'linear',
                x: 0, y: 0, x2: 0, y2: 1,
                colorStops: [
                  { offset: 0, color: 'rgba(239,68,68,0.16)' },
                  { offset: 1, color: 'rgba(239,68,68,0.00)' }
                ]
              }
            }
          },
          {
            name: '胜率',
            type: 'line',
            yAxisIndex: 1,
            data: winRates,
            smooth: true,
            showSymbol: false,
            lineStyle: {
              color: '#7c3aed',
              width: 1.2,
              type: 'dashed'
            }
          }
        ]
      }
      
      this._holdingChart.setOption(option)
    },
    
    // 图表大小调整
    resizeCharts() {
      if (this._capitalChart) this._capitalChart.resize()
      if (this._drawdownChart) this._drawdownChart.resize()
      if (this._holdingChart) this._holdingChart.resize()
    },
    
    // 销毁图表
    disposeCharts() {
      if (this._capitalChart) {
        this._capitalChart.dispose()
        this._capitalChart = null
      }
      if (this._drawdownChart) {
        this._drawdownChart.dispose()
        this._drawdownChart = null
      }
      if (this._holdingChart) {
        this._holdingChart.dispose()
        this._holdingChart = null
      }
    }
  }
}
</script>

<style scoped>
.portfolio-result-panel {
  height: 100%;
}

.result-card {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.result-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.header-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
}

.header-actions {
  display: flex;
  gap: 8px;
  align-items: center;
}

/* 空状态 */
.empty-state {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #999;
}

.empty-content {
  text-align: center;
}

.empty-icon {
  font-size: 48px;
  margin-bottom: 16px;
  display: block;
  color: #d1d5db;
}

.empty-text {
  font-size: 14px;
  color: #6b7280;
  margin-bottom: 20px;
}

.start-config-btn {
  margin-top: 16px;
}

/* 结果内容 */
.result-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

/* 统计信息面板 */
.stats-panel {
  padding: 16px;
  background: #f8fafc;
  border-radius: 8px;
  margin-bottom: 16px;
}

.stat-item {
  text-align: center;
  padding: 8px;
}

.stat-value {
  font-size: 24px;
  font-weight: bold;
  margin-bottom: 4px;
}

.stat-value.positive {
  color: #10b981;
}

.stat-value.negative {
  color: #ef4444;
}

.stat-value.neutral {
  color: #6b7280;
}

.stat-label {
  font-size: 12px;
  color: #6b7280;
}

/* 图表容器 */
.chart-container {
  flex: 1;
  min-height: 400px;
  display: flex;
  flex-direction: column;
}

.chart-wrapper {
  flex: 1;
  min-height: 350px;
}

/* 移动端优化 */
@media (max-width: 768px) {
  .stats-panel {
    padding: 12px;
    margin-bottom: 12px;
  }
  
  .stat-value {
    font-size: 20px;
  }
  
  .stat-label {
    font-size: 11px;
  }
  
  .chart-container {
    min-height: 300px;
  }
  
  .chart-wrapper {
    min-height: 250px;
  }
  
  .empty-icon {
    font-size: 40px;
    margin-bottom: 12px;
  }
  
  .empty-text {
    font-size: 13px;
  }
}

@media (max-width: 480px) {
  .stats-panel {
    padding: 10px;
    margin-bottom: 10px;
  }
  
  .stat-value {
    font-size: 18px;
  }
  
  .stat-label {
    font-size: 10px;
  }
  
  .chart-container {
    min-height: 250px;
  }
  
  .chart-wrapper {
    min-height: 200px;
  }
  
  .empty-icon {
    font-size: 36px;
    margin-bottom: 10px;
  }
  
  .empty-text {
    font-size: 12px;
  }
}
</style>
