<template>
  <div v-loading="loading" element-loading-text="分析中..." style="height:100%">
    <!-- 移动端：按钮并入标题栏右侧 -->
    <el-row :gutter="12" style="height:100%">
      <el-col v-if="!isMobile" :xs="24" :sm="7" :md="6" style="height:100%">
        <el-card style="height:100%;display:flex;flex-direction:column">
          <div slot="header">标的比较设置</div>
          <el-form label-position="top" size="small">
            <!-- 标的1选择 -->
            <el-form-item label="标的1">
              <el-select 
                v-model="symbol1" 
                filterable 
                remote 
                reserve-keyword
                placeholder="请输入股票代码/名称" 
                :remote-method="searchSymbol1" 
                :loading="searching1" 
                @change="onSymbol1Change"
                style="width:100%"
              >
                <el-option
                  v-for="item in symbol1Options"
                  :key="item.code"
                  :label="`${item.name} (${item.code})`"
                  :value="item.code"
                >
                  <span style="float: left">{{ item.name }}</span>
                  <span style="float: right; color: #8492a6; font-size: 13px">{{ item.code }}</span>
                </el-option>
              </el-select>
            </el-form-item>
            
            <!-- 标的2选择 -->
            <el-form-item label="标的2">
              <el-select 
                v-model="symbol2" 
                filterable 
                remote 
                reserve-keyword
                placeholder="请输入股票代码/名称" 
                :remote-method="searchSymbol2" 
                :loading="searching2" 
                @change="onSymbol2Change"
                style="width:100%"
              >
                <el-option
                  v-for="item in symbol2Options"
                  :key="item.code"
                  :label="`${item.name} (${item.code})`"
                  :value="item.code"
                >
                  <span style="float: left">{{ item.name }}</span>
                  <span style="float: right; color: #8492a6; font-size: 13px">{{ item.code }}</span>
                </el-option>
              </el-select>
            </el-form-item>
            
            <!-- 分析参数 -->
            <el-form-item label="移动窗口大小">
              <el-select v-model="windowSize" placeholder="选择窗口大小" style="width:100%">
                <el-option label="30天" :value="30"></el-option>
                <el-option label="60天" :value="60"></el-option>
                <el-option label="90天" :value="90"></el-option>
                <el-option label="120天" :value="120"></el-option>
                <el-option label="252天" :value="252"></el-option>
              </el-select>
            </el-form-item>
            
            <el-form-item label="时间区间">
              <el-date-picker 
                v-model="dateRange" 
                type="daterange" 
                value-format="yyyy-MM-dd" 
                range-separator="至" 
                start-placeholder="开始日期" 
                end-placeholder="结束日期" 
                :picker-options="pickerOptions"
                style="width:100%"
              />
            </el-form-item>
            
            <el-form-item>
              <el-button type="primary" :loading="loading" @click="startAnalysis" :disabled="!symbol1 || !symbol2" style="width:100%">
                <i class="el-icon-data-analysis" style="margin-right:4px"></i>开始分析
              </el-button>
            </el-form-item>
          </el-form>
        </el-card>
      </el-col>
      
      <el-col :xs="24" :sm="17" :md="18" style="height:100%">
        <el-card style="height:100%;display:flex;flex-direction:column">
          <div slot="header" style="display:flex;align-items:center;justify-content:space-between">
            <div>
              相关性分析结果
              <span v-if="analysisResult" style="color:#6b7280;font-weight:normal">
                - {{ analysisResult.symbol1.name }} vs {{ analysisResult.symbol2.name }}
              </span>
            </div>
            <div style="display:flex; gap:8px; align-items:center">
              <template v-if="isMobile">
                <el-button size="mini" type="primary" plain class="toolbar-btn" @click="showSettingDrawer=true">
                  <i class="el-icon-setting" style="margin-right:4px"></i> 设置
                </el-button>
              </template>
            </div>
          </div>
          
          <!-- 分析结果展示 -->
          <div v-if="!analysisResult" style="flex:1;display:flex;align-items:center;justify-content:center;color:#999">
            <div style="text-align:center">
              <i class="el-icon-data-analysis" style="font-size:48px;margin-bottom:16px;display:block"></i>
              <div>请选择两个标的进行分析</div>
            </div>
          </div>
          
          <div v-else style="flex:1;display:flex;flex-direction:column">
            <!-- 统计信息 -->
            <div class="stats-panel">
              <el-row :gutter="16">
                <el-col :xs="12" :sm="6">
                  <div class="stat-item">
                    <div class="stat-value" :class="getCorrelationClass(analysisResult.statistics.current_correlation)">
                      {{ (analysisResult.statistics.current_correlation * 100).toFixed(2) }}%
                    </div>
                    <div class="stat-label">当前相关性</div>
                  </div>
                </el-col>
                <el-col :xs="12" :sm="6">
                  <div class="stat-item">
                    <div class="stat-value">
                      {{ (analysisResult.statistics.average_correlation * 100).toFixed(2) }}%
                    </div>
                    <div class="stat-label">平均相关性</div>
                  </div>
                </el-col>
                <el-col :xs="12" :sm="6">
                  <div class="stat-item">
                    <div class="stat-value positive">
                      {{ (analysisResult.statistics.max_correlation * 100).toFixed(2) }}%
                    </div>
                    <div class="stat-label">最大相关性</div>
                  </div>
                </el-col>
                <el-col :xs="12" :sm="6">
                  <div class="stat-item">
                    <div class="stat-value negative">
                      {{ (analysisResult.statistics.min_correlation * 100).toFixed(2) }}%
                    </div>
                    <div class="stat-label">最小相关性</div>
                  </div>
                </el-col>
              </el-row>
            </div>
            
            <!-- 图表展示 -->
            <div style="flex:1;min-height:300px;padding:16px 0">
              <div ref="correlationChart" style="width:100%;height:100%;min-height:300px"></div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 移动端抽屉：设置 -->
    <el-drawer :visible.sync="showSettingDrawer" title="分析设置" size="90%" :append-to-body="true" :destroy-on-close="true" custom-class="correlation-drawer">
      <div class="correlation-drawer-body">
        <div class="correlation-section">
          <div class="correlation-sec-title">标的选择</div>
          <el-form label-position="top" class="correlation-form">
            <!-- 标的1选择 -->
            <el-form-item label="标的1">
              <el-select 
                v-model="symbol1" 
                filterable 
                remote 
                reserve-keyword
                placeholder="请输入股票代码/名称" 
                :remote-method="searchSymbol1" 
                :loading="searching1" 
                @change="onSymbol1Change"
                style="width:100%"
              >
                <el-option
                  v-for="item in symbol1Options"
                  :key="item.code"
                  :label="`${item.name} (${item.code})`"
                  :value="item.code"
                >
                  <span style="float: left">{{ item.name }}</span>
                  <span style="float: right; color: #8492a6; font-size: 13px">{{ item.code }}</span>
                </el-option>
              </el-select>
            </el-form-item>
            
            <!-- 标的2选择 -->
            <el-form-item label="标的2">
              <el-select 
                v-model="symbol2" 
                filterable 
                remote 
                reserve-keyword
                placeholder="请输入股票代码/名称" 
                :remote-method="searchSymbol2" 
                :loading="searching2" 
                @change="onSymbol2Change"
                style="width:100%"
              >
                <el-option
                  v-for="item in symbol2Options"
                  :key="item.code"
                  :label="`${item.name} (${item.code})`"
                  :value="item.code"
                >
                  <span style="float: left">{{ item.name }}</span>
                  <span style="float: right; color: #8492a6; font-size: 13px">{{ item.code }}</span>
                </el-option>
              </el-select>
            </el-form-item>
            
            <el-form-item label="移动窗口大小">
              <el-select v-model="windowSize" placeholder="选择窗口大小" style="width:100%">
                <el-option label="30天" :value="30"></el-option>
                <el-option label="60天" :value="60"></el-option>
                <el-option label="90天" :value="90"></el-option>
                <el-option label="120天" :value="120"></el-option>
                <el-option label="252天" :value="252"></el-option>
              </el-select>
            </el-form-item>
            
            <el-form-item label="时间区间">
              <el-date-picker 
                v-model="dateRange" 
                type="daterange" 
                value-format="yyyy-MM-dd" 
                range-separator="至" 
                start-placeholder="开始日期" 
                end-placeholder="结束日期" 
                :picker-options="pickerOptions"
                style="width:100%"
              />
            </el-form-item>
            
            <el-form-item>
              <el-button type="primary" :loading="loading" @click="startAnalysis" :disabled="!symbol1 || !symbol2" style="width:100%">
                <i class="el-icon-data-analysis" style="margin-right:4px"></i>开始分析
              </el-button>
            </el-form-item>
          </el-form>
        </div>
      </div>
    </el-drawer>
  </div>
</template>

<script>
import smartViewportManager from '@/utils/smartViewportManager'
import axios from 'axios'
import * as echarts from 'echarts'

export default {
  name: 'CorrelationAnalysis',
  data() {
    return {
      // 设备信息
      isMobile: false,
      isPortrait: true,
      
      // 标的选择
      symbol1: '',
      symbol2: '',
      symbol1Options: [],
      symbol2Options: [],
      searching1: false,
      searching2: false,
      
      // 分析参数
      windowSize: 60,
      dateRange: null,
      
      // 状态
      loading: false,
      analysisResult: null,
      showSettingDrawer: false,
      _chart: null,
      
      // 日期选择器配置
      pickerOptions: {
        shortcuts: [{
          text: '最近一个月',
          onClick(picker) {
            const end = new Date()
            const start = new Date()
            start.setTime(start.getTime() - 3600 * 1000 * 24 * 30)
            picker.$emit('pick', [start, end])
          }
        }, {
          text: '最近三个月',
          onClick(picker) {
            const end = new Date()
            const start = new Date()
            start.setTime(start.getTime() - 3600 * 1000 * 24 * 90)
            picker.$emit('pick', [start, end])
          }
        }, {
          text: '最近一年',
          onClick(picker) {
            const end = new Date()
            const start = new Date()
            start.setTime(start.getTime() - 3600 * 1000 * 24 * 365)
            picker.$emit('pick', [start, end])
          }
        }]
      }
    }
  },
  created() {
    this.updateDeviceInfo()
    this.deviceInfoInterval = setInterval(this.updateDeviceInfo, 1000)
    
    // 设置默认日期范围为最近一年
    const end = new Date()
    const start = new Date()
    start.setTime(start.getTime() - 3600 * 1000 * 24 * 365)
    this.dateRange = [start, end]
  },
  beforeDestroy() {
    if (this.deviceInfoInterval) {
      clearInterval(this.deviceInfoInterval)
    }
    if (this._chart) {
      this._chart.dispose()
      this._chart = null
    }
    window.removeEventListener('resize', this.resizeChart)
  },
  methods: {
    updateDeviceInfo() {
      const viewportInfo = smartViewportManager.getViewportInfo()
      this.isMobile = viewportInfo.isMobile
      this.isPortrait = viewportInfo.isPortrait
    },
    
    // 搜索标的1
    async searchSymbol1(query) {
      if (!query) {
        this.symbol1Options = []
        return
      }
      this.searching1 = true
      try {
        // 搜索股票
        const stockRes = await axios.get(`/webManager/stock/search?q=${query}`)
        const stockOptions = (stockRes.data.data || []).map(item => ({
          code: item.stock_code,
          name: item.stock_name,
          type: 'stock'
        }))
        
        // 搜索ETF
        const etfRes = await axios.get(`/webManager/etf/search?q=${query}`)
        const etfOptions = (etfRes.data.data || []).map(item => ({
          code: item.fund_code,
          name: item.fund_name,
          type: 'fund'
        }))
        
        this.symbol1Options = [...stockOptions, ...etfOptions].slice(0, 20)
      } catch (error) {
        this.$message.error('搜索失败')
        this.symbol1Options = []
      } finally {
        this.searching1 = false
      }
    },
    
    // 搜索标的2
    async searchSymbol2(query) {
      if (!query) {
        this.symbol2Options = []
        return
      }
      this.searching2 = true
      try {
        // 搜索股票
        const stockRes = await axios.get(`/webManager/stock/search?q=${query}`)
        const stockOptions = (stockRes.data.data || []).map(item => ({
          code: item.stock_code,
          name: item.stock_name,
          type: 'stock'
        }))
        
        // 搜索ETF
        const etfRes = await axios.get(`/webManager/etf/search?q=${query}`)
        const etfOptions = (etfRes.data.data || []).map(item => ({
          code: item.fund_code,
          name: item.fund_name,
          type: 'fund'
        }))
        
        this.symbol2Options = [...stockOptions, ...etfOptions].slice(0, 20)
      } catch (error) {
        this.$message.error('搜索失败')
        this.symbol2Options = []
      } finally {
        this.searching2 = false
      }
    },
    
    // 标的1选择变化
    onSymbol1Change(value) {
      this.symbol1 = value
    },
    
    // 标的2选择变化
    onSymbol2Change(value) {
      this.symbol2 = value
    },
    
    // 开始分析
    async startAnalysis() {
      if (!this.symbol1 || !this.symbol2) {
        this.$message.warning('请选择两个标的进行分析')
        return
      }
      
      if (this.symbol1 === this.symbol2) {
        this.$message.warning('请选择不同的标的进行分析')
        return
      }
      
      this.loading = true
      try {
        const params = {
          symbol1: this.symbol1,
          symbol2: this.symbol2,
          window_size: this.windowSize
        }
        
        if (this.dateRange && this.dateRange.length === 2) {
          // 将Date对象转换为YYYY-MM-DD格式
          params.start_date = this.formatDate(this.dateRange[0])
          params.end_date = this.formatDate(this.dateRange[1])
        }
        
        const response = await axios.post('/webManager/analysis/correlation/compare', params)
        
        if (response.data.code === 0) {
          this.analysisResult = response.data.data
          this.$message.success('分析完成')
          
          // 绘制图表
          this.$nextTick(() => {
            this.drawChart()
          })
          
          // 移动端自动关闭抽屉
          if (this.isMobile) {
            this.showSettingDrawer = false
          }
        } else {
          this.$message.error(response.data.msg || '分析失败')
        }
      } catch (error) {
        console.error('相关性分析失败:', error)
        this.$message.error('分析失败，请稍后重试')
      } finally {
        this.loading = false
      }
    },
    
    // 获取相关性数值的样式类
    getCorrelationClass(correlation) {
      if (correlation > 0.5) return 'positive'
      if (correlation < -0.5) return 'negative'
      return 'neutral'
    },
    
    // 绘制ECharts图表
    drawChart() {
      if (!this.analysisResult || !this.analysisResult.chart_data) {
        return
      }
      
      try {
        const chartElement = this.$refs.correlationChart
        if (!chartElement) {
          return
        }
        
        // 销毁现有图表
        if (this._chart) {
          this._chart.dispose()
          this._chart = null
        }
        
        // 初始化图表
        this._chart = echarts.init(chartElement, null, {
          renderer: 'canvas',
          devicePixelRatio: window.devicePixelRatio || 1
        })
        
        const config = this.analysisResult.chart_config || {}
        const chartData = this.analysisResult.chart_data || []
        
        // 准备数据 - 将日期字符串转换为Date对象
        const chartDataWithDates = chartData.map(item => [new Date(item[0]), item[1]])
        const dates = chartDataWithDates.map(item => item[0])
        const correlations = chartDataWithDates.map(item => item[1])
        
        // 计算参考线数据
        const zeroLine = chartDataWithDates.map(item => [item[0], 0])
        const positiveLine = chartDataWithDates.map(item => [item[0], 0.5])
        const negativeLine = chartDataWithDates.map(item => [item[0], -0.5])
        
        // 配置图表选项
        const option = {
          title: {
            text: config.title || '相关性分析',
            subtext: config.subtitle || '',
            left: 'center',
            top: 10,
            textStyle: {
              fontSize: 16,
              fontWeight: 'bold',
              color: '#1f2937'
            },
            subtextStyle: {
              fontSize: 12,
              color: '#6b7280'
            }
          },
          tooltip: {
            trigger: 'axis',
            formatter: (params) => {
              if (!params || params.length === 0) return ''
              const date = params[0].axisValueLabel || params[0].axisValue
              const lines = [date]
              params.forEach(param => {
                const name = param.seriesName
                const value = param.data && param.data[1]
                let displayValue = '-'
                if (value !== null && value !== undefined) {
                  displayValue = (value * 100).toFixed(2) + '%'
                }
                lines.push(`${param.marker} ${name}: ${displayValue}`)
              })
              return lines.join('<br/>')
            }
          },
          legend: {
            top: 45,
            left: 8,
            data: ['相关系数', '零线', '正相关线', '负相关线'],
            selected: {
              '零线': false,
              '正相关线': false,
              '负相关线': false
            }
          },
          dataZoom: [
            { type: 'inside', xAxisIndex: [0, 0] },
            { type: 'slider', xAxisIndex: [0, 0], height: 14, bottom: 6 }
          ],
          grid: {
            left: '6%',
            right: '6%',
            top: 80,
            bottom: 56,
            containLabel: true
          },
          xAxis: {
            type: 'time',
            boundaryGap: false,
            axisLabel: {
              color: '#6b7280',
              hideOverlap: true
            },
            axisLine: {
              lineStyle: { color: '#e5e7eb' }
            },
            splitLine: {
              show: false
            }
          },
          yAxis: {
            type: 'value',
            name: '相关系数',
            nameLocation: 'end',
            nameGap: 6,
            min: config.yAxis_min || -1.1,
            max: config.yAxis_max || 1.1,
            axisLabel: {
              color: '#6b7280',
              formatter: (value) => {
                return (value * 100).toFixed(0) + '%'
              }
            },
            axisLine: {
              lineStyle: { color: '#e5e7eb' }
            },
            splitLine: {
              show: true,
              lineStyle: {
                color: '#f3f4f6',
                type: 'dashed'
              }
            }
          },
          series: [
            {
              type: 'line',
              name: '相关系数',
              data: chartDataWithDates,
              smooth: true,
              showSymbol: false,
              lineStyle: {
                width: 2,
                color: '#2563eb'
              },
              itemStyle: {
                color: '#2563eb'
              },
              areaStyle: {
                color: {
                  type: 'linear',
                  x: 0, y: 0, x2: 0, y2: 1,
                  colorStops: [
                    { offset: 0, color: 'rgba(37, 99, 235, 0.1)' },
                    { offset: 1, color: 'rgba(37, 99, 235, 0.0)' }
                  ]
                }
              }
            },
            {
              type: 'line',
              name: '零线',
              data: zeroLine,
              showSymbol: false,
              lineStyle: {
                color: '#9ca3af',
                type: 'dashed',
                width: 1
              },
              itemStyle: {
                color: '#9ca3af'
              }
            },
            {
              type: 'line',
              name: '正相关线',
              data: positiveLine,
              showSymbol: false,
              lineStyle: {
                color: '#10b981',
                type: 'dotted',
                width: 1
              },
              itemStyle: {
                color: '#10b981'
              }
            },
            {
              type: 'line',
              name: '负相关线',
              data: negativeLine,
              showSymbol: false,
              lineStyle: {
                color: '#ef4444',
                type: 'dotted',
                width: 1
              },
              itemStyle: {
                color: '#ef4444'
              }
            }
          ]
        }
        
        // 设置图表选项
        this._chart.setOption(option)
        
        // 监听窗口大小变化
        window.addEventListener('resize', this.resizeChart)
        
      } catch (error) {
        console.error('绘制图表失败:', error)
        this.$message.error('图表渲染失败')
      }
    },
    
    // 图表大小调整
    resizeChart() {
      if (this._chart) {
        this._chart.resize()
      }
    },
    
    // 格式化日期为YYYY-MM-DD格式
    formatDate(date) {
      if (!date) return null
      const d = new Date(date)
      const year = d.getFullYear()
      const month = String(d.getMonth() + 1).padStart(2, '0')
      const day = String(d.getDate()).padStart(2, '0')
      return `${year}-${month}-${day}`
    }
  }
}
</script>

<style scoped>
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

/* 移动端抽屉样式 */
.correlation-drawer-body {
  padding: 16px;
}

.correlation-section {
  margin-bottom: 24px;
}

.correlation-sec-title {
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 16px;
  padding-bottom: 8px;
  border-bottom: 2px solid #e5e7eb;
}

.correlation-form .el-form-item {
  margin-bottom: 20px;
}

.correlation-form .el-form-item__label {
  font-size: 14px;
  font-weight: 500;
  color: #374151;
  line-height: 1.5;
  margin-bottom: 8px;
}

/* 响应式布局 */
@media (max-width: 768px) {
  .stats-panel {
    padding: 12px;
  }
  
  .stat-value {
    font-size: 20px;
  }
  
  .stat-label {
    font-size: 11px;
  }
}

/* 工具栏按钮样式 */
.toolbar-btn {
  min-width: auto;
  padding: 6px 12px;
}
</style>
