<template>
  <div class="portfolio-config-form">
    <!-- 组合标的配置 -->
    <div class="config-section">
      <div class="section-header">
        <span class="section-title">组合标的</span>
        <el-button 
          :size="isMobile ? 'small' : 'mini'" 
          type="primary" 
          @click="$emit('add-portfolio-item')"
        >
          <i class="el-icon-plus"></i>
          添加标的
        </el-button>
      </div>
      
      <!-- 桌面端表格布局 -->
      <div v-if="!isMobile" class="portfolio-table">
        <el-table :data="portfolioItems" :size="isMobile ? 'small' : 'mini'" style="width:100%">
          <el-table-column prop="type" label="类型" width="80">
            <template slot-scope="scope">
              <el-select 
                v-model="scope.row.type" 
                :size="isMobile ? 'small' : 'mini'" 
                style="width:100%"
                @change="$emit('type-change', scope.$index, scope.row.type)"
              >
                <el-option label="股票" value="stock"></el-option>
                <el-option label="ETF" value="fund"></el-option>
                <el-option label="现金" value="cash"></el-option>
              </el-select>
            </template>
          </el-table-column>
          <el-table-column prop="name" label="名称" min-width="100">
            <template slot-scope="scope">
              <div v-if="scope.row.type === 'cash'">
                <span>现金</span>
              </div>
              <div v-else>
                <el-select
                  v-model="scope.row.code"
                  filterable
                  remote
                  reserve-keyword
                  placeholder="搜索标的"
                  :remote-method="(query) => $emit('search-securities', query, scope.row.type)"
                  :loading="scope.row.type === 'stock' ? stockSearching : fundSearching"
                  :size="isMobile ? 'small' : 'mini'"
                  style="width:100%"
                  @change="$emit('code-change', scope.$index, scope.row.code, scope.row.type)"
                >
                  <el-option
                    v-for="item in getSearchResults(scope.row.type)"
                    :key="item.code"
                    :label="formatOptionLabel(item)"
                    :value="item.code"
                  />
                </el-select>
              </div>
            </template>
          </el-table-column>
          <el-table-column prop="ratio" label="比例" width="120">
            <template slot-scope="scope">
              <el-input-number 
                v-model="scope.row.ratio" 
                :min="0" 
                :max="1" 
                :step="0.01" 
                :precision="2"
                :size="isMobile ? 'small' : 'mini'"
                style="width:100%"
                controls-position="right"
              />
            </template>
          </el-table-column>
          <el-table-column label="操作" width="50">
            <template slot-scope="scope">
              <el-button 
                :size="isMobile ? 'small' : 'mini'" 
                type="text" 
                @click="$emit('remove-portfolio-item', scope.$index)"
                :disabled="portfolioItems.length <= 1"
              >删除</el-button>
            </template>
          </el-table-column>
        </el-table>
      </div>
      
      <!-- 移动端卡片布局 -->
      <div v-else class="portfolio-cards">
        <div 
          v-for="(item, index) in portfolioItems" 
          :key="index" 
          class="portfolio-card"
        >
          <div class="card-header">
            <div class="header-left">
              <el-select 
                v-model="item.type" 
                size="mini" 
                class="type-selector-inline"
                @change="$emit('type-change', index, item.type)"
              >
                <el-option label="股票" value="stock"></el-option>
                <el-option label="ETF" value="fund"></el-option>
                <el-option label="现金" value="cash"></el-option>
              </el-select>
            </div>
            <el-button 
              size="mini" 
              type="text" 
              class="delete-btn"
              @click="$emit('remove-portfolio-item', index)"
              :disabled="portfolioItems.length <= 1"
            >
              <i class="el-icon-delete"></i>
            </el-button>
          </div>
          
          <div class="card-content">
            
            <div v-if="item.type === 'cash'" class="cash-item">
              <span class="cash-label">现金</span>
            </div>
            <div v-else class="security-item">
              <el-select
                v-model="item.code"
                filterable
                remote
                reserve-keyword
                placeholder="搜索标的"
                :remote-method="(query) => $emit('search-securities', query, item.type)"
                :loading="item.type === 'stock' ? stockSearching : fundSearching"
                size="small"
                style="width:100%"
                @change="$emit('code-change', index, item.code, item.type)"
              >
                <el-option
                  v-for="searchItem in getSearchResults(item.type)"
                  :key="searchItem.code"
                  :label="formatOptionLabel(searchItem)"
                  :value="searchItem.code"
                />
              </el-select>
            </div>
            
            <div class="ratio-input">
              <label class="ratio-label">比例:</label>
              <el-input-number 
                v-model="item.ratio" 
                :min="0" 
                :max="1" 
                :step="0.01" 
                :precision="2"
                size="small"
                style="width:100%"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  
    <!-- 再平衡策略配置 -->
    <div class="config-section">
      <div class="section-title">再平衡策略</div>
      <el-form label-position="top" :size="isMobile ? 'small' : 'small'">
        <el-form-item label="策略类型">
          <el-select v-model="rebalanceStrategy.type" style="width:100%" @change="$emit('update:rebalance-strategy', rebalanceStrategy)">
            <el-option label="不再平衡" value="none"></el-option>
            <el-option label="定时平衡" value="time"></el-option>
            <el-option label="比例偏离再平衡" value="deviation"></el-option>
          </el-select>
        </el-form-item>
        
        <el-form-item v-if="rebalanceStrategy.type === 'time'" label="平衡间隔(天)">
          <el-input-number 
            v-model="rebalanceStrategy.interval_days" 
            :min="1" 
            :max="365"
            style="width:100%"
            @change="$emit('update:rebalance-strategy', rebalanceStrategy)"
          />
        </el-form-item>
        
        <el-form-item v-if="rebalanceStrategy.type === 'deviation'" label="偏离触发比例">
          <el-input-number 
            v-model="rebalanceStrategy.threshold" 
            :min="0.01" 
            :max="0.5" 
            :step="0.01"
            :precision="2"
            style="width:100%"
            @change="$emit('update:rebalance-strategy', rebalanceStrategy)"
          />
        </el-form-item>
      </el-form>
    </div>
    
    <!-- 其他参数配置 -->
    <div class="config-section">
      <div class="section-title">其他参数</div>
      <el-form label-position="top" :size="isMobile ? 'small' : 'small'">
        <el-form-item label="初始资金(元)">
          <el-input-number 
            v-model="initialCapital" 
            :min="1000" 
            :step="1000"
            style="width:100%"
            @change="$emit('update:initial-capital', initialCapital)"
          />
        </el-form-item>
        
        <el-form-item label="每月取出现金(元)">
          <el-input-number 
            v-model="monthlyWithdrawal" 
            :min="0" 
            :step="100"
            style="width:100%"
            @change="$emit('update:monthly-withdrawal', monthlyWithdrawal)"
          />
        </el-form-item>
        
        <el-form-item label="交易佣金率">
          <el-input 
            v-model="commissionRate" 
            placeholder="0.02854%" 
            style="width:100%"
            @input="$emit('update:commission-rate', commissionRate)"
          >
            <template slot="append">%</template>
          </el-input>
        </el-form-item>
        
        <el-form-item label="回测日期区间">
          <el-select v-model="dateRange.type" style="width:100%" @change="$emit('update:date-range', dateRange)">
            <el-option label="最近一个月" value="month"></el-option>
            <el-option label="最近半年" value="half_year"></el-option>
            <el-option label="最近一年" value="year"></el-option>
            <el-option label="最近五年" value="five_year"></el-option>
            <el-option label="最近十年" value="ten_year"></el-option>
            <el-option label="自定义" value="custom"></el-option>
          </el-select>
        </el-form-item>
        
        <template v-if="dateRange.type === 'custom'">
          <el-form-item label="开始日期">
            <el-date-picker 
              v-model="dateRange.start_date" 
              type="date" 
              value-format="yyyy-MM-dd"
              style="width:100%"
              @change="$emit('update:date-range', dateRange)"
            />
          </el-form-item>
          <el-form-item label="结束日期">
            <el-date-picker 
              v-model="dateRange.end_date" 
              type="date" 
              value-format="yyyy-MM-dd"
              style="width:100%"
              @change="$emit('update:date-range', dateRange)"
            />
          </el-form-item>
        </template>
      </el-form>
    </div>
    
          <!-- 开始回测按钮 -->
          <div class="action-section">
            <el-button 
              type="primary" 
              :loading="loading" 
              @click="$emit('start-backtest')" 
              :disabled="!canStartBacktest"
              :size="isMobile ? 'medium' : 'small'"
              style="width:100%"
            >
              <i class="el-icon-data-analysis" style="margin-right:4px"></i>
              开始回测
            </el-button>
          </div>
  </div>
</template>

<script>
export default {
  name: 'PortfolioConfigForm',
  props: {
    portfolioItems: {
      type: Array,
      required: true
    },
    rebalanceStrategy: {
      type: Object,
      required: true
    },
    initialCapital: {
      type: Number,
      required: true
    },
    monthlyWithdrawal: {
      type: Number,
      required: true
    },
    commissionRate: {
      type: String,
      required: true
    },
    dateRange: {
      type: Object,
      required: true
    },
    loading: {
      type: Boolean,
      default: false
    },
    canStartBacktest: {
      type: Boolean,
      default: false
    },
    isMobile: {
      type: Boolean,
      default: false
    },
    isPortrait: {
      type: Boolean,
      default: true
    },
    stockSearching: {
      type: Boolean,
      default: false
    },
    fundSearching: {
      type: Boolean,
      default: false
    },
    stockSearchResults: {
      type: Array,
      default: () => []
    },
    fundSearchResults: {
      type: Array,
      default: () => []
    }
  },
  methods: {
    getSearchResults(type) {
      if (type === 'stock') {
        return this.stockSearchResults
      } else if (type === 'fund') {
        return this.fundSearchResults
      }
      return []
    },
    
    formatOptionLabel(item) {
      // 格式化下拉选项标签，显示名称、代码和起始时间
      const name = item.name || item.stock_name || item.fund_name
      const code = item.code || item.stock_code || item.fund_code
      const startDate = item.start_date
      
      if (startDate) {
        // 格式化日期为 YYYY-MM-DD
        const dateStr = new Date(startDate).toLocaleDateString('zh-CN', {
          year: 'numeric',
          month: '2-digit',
          day: '2-digit'
        }).replace(/\//g, '-')
        
        // 移动端使用更简洁的格式
        if (this.isMobile) {
          return `${name} (${code}) - ${dateStr}`
        } else {
          return `${name} (${code}) - 起始: ${dateStr}`
        }
      } else {
        return `${name} (${code})`
      }
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
    }
  }
}
</script>

<style scoped>
.portfolio-config-form {
  display: flex;
  flex-direction: column;
  gap: 20px;
  /* 移除min-height: 100%，避免高度继承问题 */
  padding-bottom: 20px;
  width: 100%; /* 确保宽度正确 */
}

.config-section {
  background: #f8fafc;
  border-radius: 8px;
  padding: 16px;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.section-title {
  font-weight: 600;
  color: #374151;
  font-size: 14px;
}

.portfolio-table {
  width: 100%;
}

.portfolio-table :deep(.el-table) {
  font-size: 12px;
}

.portfolio-table :deep(.el-table .cell) {
  padding: 4px 8px;
}

.portfolio-table :deep(.el-input-number) {
  width: 100%;
}

.portfolio-table :deep(.el-input-number .el-input__inner) {
  text-align: center;
  padding: 0 8px;
}

.portfolio-cards {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.portfolio-card {
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 12px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid #f0f0f0;
}

.header-left {
  flex: 1;
  margin-right: 8px;
}

.type-selector-inline {
  width: 100%;
}

.type-selector-inline :deep(.el-input__inner) {
  font-size: 12px;
  height: 24px;
  line-height: 24px;
  padding: 0 8px;
  border-radius: 4px;
  border: 1px solid #dcdfe6;
  background-color: #f8f9fa;
}

.type-selector-inline :deep(.el-input__suffix) {
  right: 4px;
}

.delete-btn {
  color: #f56c6c;
  padding: 4px;
}

.delete-btn:hover {
  background-color: #fef0f0;
}

.card-content {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.security-item {
  width: 100%;
}

.ratio-input {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
}

.ratio-label {
  font-size: 12px;
  color: #6b7280;
  white-space: nowrap;
  min-width: 40px;
}

.ratio-input :deep(.el-input-number) {
  flex: 1;
}

.ratio-input :deep(.el-input-number .el-input__inner) {
  text-align: center;
  font-size: 14px;
}

.cash-item {
  padding: 8px;
  background: #f3f4f6;
  border-radius: 4px;
  text-align: center;
}

.cash-label {
  color: #6b7280;
  font-size: 14px;
}

.action-section {
  padding: 16px;
  border-top: 1px solid #e5e7eb;
  background: white;
  border-radius: 8px;
  min-height: 60px;
}

@media (max-width: 768px) {
  .portfolio-config-form {
    gap: 16px;
  }
  
  .config-section {
    padding: 12px;
  }
  
  .section-title {
    font-size: 13px;
  }
  
  .portfolio-card {
    padding: 10px;
  }
  
  .action-section {
    padding: 12px;
    position: sticky;
    bottom: 0;
    z-index: 10;
    margin-bottom: calc(env(safe-area-inset-bottom, 0px) + var(--bottom-nav-height, 60px));
  }
  
  /* 移动端下拉选项优化 */
  .security-item :deep(.el-select-dropdown__item) {
    font-size: 12px;
    line-height: 1.4;
    padding: 8px 12px;
    white-space: normal;
    word-break: break-all;
  }
  
  .security-item :deep(.el-select-dropdown__item span) {
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
}

@media (max-width: 480px) {
  .portfolio-config-form {
    gap: 12px;
  }
  
  .config-section {
    padding: 10px;
  }
  
  .portfolio-card {
    padding: 8px;
  }
  
  .action-section {
    padding: 10px;
  }
}
</style>
