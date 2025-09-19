<template>
  <div v-loading="loading" element-loading-text="回测中..." class="portfolio-backtest-container">
        <!-- 桌面端布局 - 全屏结果展示 -->
        <div v-if="!isMobile" class="desktop-layout">
          <div class="desktop-result-panel">
            <result-panel 
              :backtest-result="backtestResult"
              :active-tab="activeTab"
              :is-mobile="false"
              @update:active-tab="activeTab = $event"
              @show-config="showConfigDrawer = true"
            />
            </div>
          </div>
          
    <!-- 移动端布局 -->
    <div v-else class="mobile-layout">
      <!-- 移动端结果展示 -->
      <div class="mobile-result-panel">
        <result-panel 
          :backtest-result="backtestResult"
          :active-tab="activeTab"
          :is-mobile="true"
          @update:active-tab="activeTab = $event"
          @show-config="showConfigDrawer = true"
        />
            </div>
          </div>
          
    <!-- 配置抽屉 - 桌面端和移动端通用 -->
    <el-drawer 
      :visible.sync="showConfigDrawer" 
      title="组合配置"
      :size="isMobile ? '100%' : '50%'"
      :append-to-body="true" 
      :destroy-on-close="false"
      :class="['config-drawer', { 'mobile-config-drawer': isMobile, 'desktop-config-drawer': !isMobile }]"
    >
      <div :class="['config-content', { 'mobile-config-content': isMobile, 'desktop-config-content': !isMobile }]">
        <config-form
          :portfolio-items="portfolioItems"
          :rebalance-strategy="rebalanceStrategy"
          :initial-capital="initialCapital"
          :monthly-withdrawal="monthlyWithdrawal"
          :commission-rate="commissionRate"
          :date-range="dateRange"
          :loading="loading"
          :can-start-backtest="canStartBacktest"
          :is-mobile="isMobile"
          :is-portrait="isPortrait"
          :stock-searching="stockSearching"
          :fund-searching="fundSearching"
          :stock-search-results="stockSearchResults"
          :fund-search-results="fundSearchResults"
          @add-portfolio-item="addPortfolioItem"
          @remove-portfolio-item="removePortfolioItem"
          @type-change="onTypeChange"
          @code-change="onCodeChange"
          @search-securities="searchSecurities"
          @start-backtest="startBacktest"
          @update:rebalance-strategy="rebalanceStrategy = $event"
          @update:initial-capital="initialCapital = $event"
          @update:monthly-withdrawal="monthlyWithdrawal = $event"
          @update:commission-rate="commissionRate = $event"
          @update:date-range="dateRange = $event"
        />
      </div>
    </el-drawer>
  </div>
</template>

<script>
import smartViewportManager from '@/utils/smartViewportManager'
import axios from 'axios'
import * as echarts from 'echarts'
import ConfigForm from './components/PortfolioConfigForm.vue'
import ResultPanel from './components/PortfolioResultPanel.vue'

export default {
  name: 'PortfolioBacktest',
  components: {
    ConfigForm,
    ResultPanel
  },
  data() {
    return {
      // 设备信息
      isMobile: false,
      isPortrait: true,
      
      // 组合配置
      portfolioItems: [
        { type: 'stock', code: '', name: '', ratio: 0.5 },
        { type: 'cash', code: 'cash', name: '现金', ratio: 0.5 }
      ],
      
      // 再平衡策略
      rebalanceStrategy: {
        type: 'none',
        interval_days: 30,
        threshold: 0.1
      },
      
      // 其他参数
      initialCapital: 100000,
      monthlyWithdrawal: 0,
      commissionRate: '0.02854',
      dateRange: {
        type: 'year',
        start_date: null,
        end_date: null
      },
      
      // 状态
      loading: false,
      backtestResult: null,
      showConfigDrawer: false,
      activeTab: 'capital',
      
      // 搜索相关
      stockSearching: false,
      fundSearching: false,
      stockSearchResults: [],
      fundSearchResults: []
    }
  },
  computed: {
    canStartBacktest() {
      if (this.portfolioItems.length === 0) return false
      
      // 检查比例总和是否为1
      const totalRatio = this.portfolioItems.reduce((sum, item) => sum + (item.ratio || 0), 0)
      if (Math.abs(totalRatio - 1) > 0.01) return false
      
      // 检查是否有有效的标的
      const hasValidItems = this.portfolioItems.some(item => {
        if (item.type === 'cash') return true
        return item.code && item.code.trim() !== ''
      })
      
      return hasValidItems && this.initialCapital > 0
    }
  },
  created() {
    this.updateDeviceInfo()
    this.deviceInfoInterval = setInterval(this.updateDeviceInfo, 1000)
    this.setupMobileViewport()
  },
  beforeDestroy() {
    if (this.deviceInfoInterval) {
      clearInterval(this.deviceInfoInterval)
    }
  },
  methods: {
    updateDeviceInfo() {
      const viewportInfo = smartViewportManager.getViewportInfo()
      this.isMobile = viewportInfo.isMobile
      this.isPortrait = viewportInfo.isPortrait
    },
    
    // 设置移动端视口高度
    setupMobileViewport() {
      if (this.isMobile) {
        // 设置CSS变量来动态计算可用高度
        const setViewportHeight = () => {
          const vh = window.innerHeight * 0.01
          const bottomNavHeight = 60 // 底部导航栏高度
          const availableHeight = window.innerHeight - bottomNavHeight
          const availableVh = availableHeight * 0.01
          
          document.documentElement.style.setProperty('--vh', `${vh}px`)
          document.documentElement.style.setProperty('--available-vh', `${availableVh}px`)
          document.documentElement.style.setProperty('--bottom-nav-height', `${bottomNavHeight}px`)
        }
        
        setViewportHeight()
        window.addEventListener('resize', setViewportHeight)
        window.addEventListener('orientationchange', () => {
          setTimeout(setViewportHeight, 100)
        })
      }
    },
    
    
    // 组合标的管理
    addPortfolioItem() {
      this.portfolioItems.push({
        type: 'stock',
        code: '',
        name: '',
        ratio: 0
      })
    },
    
    removePortfolioItem(index) {
      if (this.portfolioItems.length > 1) {
        this.portfolioItems.splice(index, 1)
      }
    },
    
    // 类型变化处理
    onTypeChange(index, newType) {
      const item = this.portfolioItems[index]
      item.type = newType
      
      // 重置代码和名称
      if (newType === 'cash') {
        item.code = 'cash'
        item.name = '现金'
      } else {
        item.code = ''
        item.name = ''
      }
    },
    
    // 代码变化处理
    onCodeChange(index, code, type) {
      const item = this.portfolioItems[index]
      item.code = code
      
      // 根据代码查找名称
      if (type === 'stock') {
        const stock = this.stockSearchResults.find(s => s.code === code)
        if (stock) {
          item.name = stock.name
        }
      } else if (type === 'fund') {
        const fund = this.fundSearchResults.find(f => f.code === code)
        if (fund) {
          item.name = fund.name
        }
      }
    },
    
    // 搜索证券
    async searchSecurities(query, type) {
      if (!query || query.length < 2) {
        return
      }
      
      try {
        if (type === 'stock') {
          this.stockSearching = true
          const response = await axios.get('/webManager/stock/search', {
            params: { q: query }
          })
          if (response.data.code === 0) {
            this.stockSearchResults = (response.data.data || []).map(item => ({
              code: item.stock_code, // 直接使用后端返回的带前缀代码
              name: item.stock_name,
              start_date: item.start_date // 包含起始时间
            }))
          }
        } else if (type === 'fund') {
          this.fundSearching = true
          const response = await axios.get('/webManager/fund/search', {
            params: { keyword: query, limit: 20 }
          })
          if (response.data.code === 0) {
            this.fundSearchResults = (response.data.data || []).map(item => ({
              code: item.code,
              name: item.name,
              start_date: item.start_date // 包含起始时间
            }))
          }
        }
      } catch (error) {
        console.error('搜索证券失败:', error)
        this.$message.error('搜索失败，请稍后重试')
      } finally {
        if (type === 'stock') {
          this.stockSearching = false
        } else if (type === 'fund') {
          this.fundSearching = false
        }
      }
    },
    
    // 获取搜索结果
    getSearchResults(type) {
      if (type === 'stock') {
        return this.stockSearchResults
      } else if (type === 'fund') {
        return this.fundSearchResults
      }
      return []
    },
    
    getTypeTagType(type) {
      const types = {
        'stock': 'primary',
        'fund': 'success',
        'cash': 'info'
      }
      return types[type] || 'default'
    },
    
    getTypeText(type) {
      const texts = {
        'stock': '股票',
        'fund': 'ETF',
        'cash': '现金'
      }
      return texts[type] || type
    },
    
    
    // 开始回测
    async startBacktest() {
      if (!this.canStartBacktest) {
        this.$message.warning('请检查配置参数')
        return
      }
      
      this.loading = true
      try {
        const params = {
          portfolio_items: this.portfolioItems,
          rebalance_strategy: this.rebalanceStrategy,
          initial_capital: this.initialCapital,
          monthly_withdrawal: this.monthlyWithdrawal,
          commission_rate: this.commissionRate + '%',
          date_range: this.dateRange
        }
        
        // 调试信息
        console.log('发送给后端的参数:', JSON.stringify(params, null, 2))
        
        const response = await axios.post('/webManager/backtest/portfolio', params)
        
        if (response.data.code === 0) {
          this.backtestResult = response.data.data
          this.$message.success('回测完成')
          
          // 移动端自动关闭抽屉
          if (this.isMobile) {
            this.showConfigDrawer = false
          }
        } else {
          this.$message.error(response.data.msg || '回测失败')
        }
      } catch (error) {
        console.error('组合回测失败:', error)
        this.$message.error('回测失败，请稍后重试')
      } finally {
        this.loading = false
      }
    }
  }
}
</script>

<style scoped>
/* 主容器 */
.portfolio-backtest-container {
  height: 100%;
  overflow: hidden;
}

/* 桌面端布局 */
.desktop-layout {
  height: 100%;
}

.desktop-result-panel {
  height: 100%;
  width: 100%;
}

/* 移动端布局 */
.mobile-layout {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.mobile-result-panel {
  flex: 1;
  height: 100%;
  /* 移动端考虑底部导航栏 */
  max-height: calc(100vh - var(--bottom-nav-height, 60px));
  overflow: hidden;
}

/* 配置抽屉通用样式 */
.config-drawer :deep(.el-drawer__body) {
  padding: 0;
  height: 100%;
  overflow: hidden;
}

/* 移动端配置抽屉 */
.mobile-config-drawer {
  /* 移动端全屏显示 */
}

.mobile-config-content {
  height: 100%;
  overflow-y: auto;
  padding: 16px;
  /* 使用动态计算的可用高度 */
  padding-bottom: calc(16px + env(safe-area-inset-bottom, 0px) + var(--bottom-nav-height, 60px));
  box-sizing: border-box;
  /* 确保内容区域有足够滚动空间 */
  min-height: calc(100vh - var(--bottom-nav-height, 60px));
}

/* 桌面端配置抽屉 */
.desktop-config-drawer {
  /* 桌面端50%宽度显示 */
}

.desktop-config-content {
  height: 100%;
  overflow-y: auto;
  padding: 24px;
  box-sizing: border-box;
  /* 桌面端不需要考虑底部导航栏 */
  min-height: calc(100vh - 60px);
}


/* 响应式优化 */
@media (max-width: 768px) {
  .mobile-config-content {
    padding: 12px;
  }
}

@media (max-width: 480px) {
  .mobile-config-content {
    padding: 8px;
  }
}

/* 滚动条样式优化 */
.mobile-config-content::-webkit-scrollbar,
.desktop-config-content::-webkit-scrollbar {
  width: 6px;
}

.mobile-config-content::-webkit-scrollbar-track,
.desktop-config-content::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 3px;
}

.mobile-config-content::-webkit-scrollbar-thumb,
.desktop-config-content::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 3px;
}

.mobile-config-content::-webkit-scrollbar-thumb:hover,
.desktop-config-content::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}
</style>
