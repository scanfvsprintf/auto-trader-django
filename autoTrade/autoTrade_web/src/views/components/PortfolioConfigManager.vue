<template>
  <div class="config-manager">
    <!-- 配置选择按钮 -->
    <el-button 
      :size="isMobile ? 'mini' : 'small'" 
      type="primary" 
      plain 
      @click="showConfigSelector = true"
      :icon="'el-icon-folder-opened'"
    >
      {{ isMobile ? '配置' : '加载配置' }}
    </el-button>

    <!-- 配置选择对话框 -->
    <el-dialog
      :title="'选择回测配置'"
      :visible.sync="showConfigSelector"
      :width="isMobile ? '95%' : '800px'"
      :close-on-click-modal="true"
      :close-on-press-escape="true"
      :modal="true"
      :append-to-body="true"
      :class="['config-selector-dialog', { 'mobile-dialog': isMobile }]"
    >
      <div class="config-selector-content">
        <!-- 搜索框 -->
        <div class="search-section">
          <el-input
            v-model="searchKeyword"
            :placeholder="'搜索配置名称...'"
            :size="isMobile ? 'small' : 'medium'"
            prefix-icon="el-icon-search"
            clearable
          />
        </div>

        <!-- 配置列表 -->
        <div class="config-list" :class="{ 'mobile-config-list': isMobile }">
          <div v-if="filteredConfigs.length === 0" class="empty-state">
            <i class="el-icon-folder-opened empty-icon"></i>
            <p v-if="searchKeyword">没有找到匹配的配置</p>
            <div v-else class="empty-tips">
              <p>暂无保存的配置</p>
              <p class="empty-hint">请先填写回测配置，然后点击"保存配置"按钮</p>
              <el-button 
                type="primary" 
                size="small" 
                @click="showConfigSelector = false"
                style="margin-top: 16px;"
              >
                我知道了
              </el-button>
            </div>
          </div>

          <div 
            v-for="config in filteredConfigs" 
            :key="config.id"
            class="config-item"
            :class="{ 'mobile-config-item': isMobile }"
            @click="selectConfig(config)"
          >
            <div class="config-info">
              <div class="config-name">{{ config.name }}</div>
              <div class="config-meta">
                <span class="config-type">
                  <i class="el-icon-s-data" v-if="config.hasStock"></i>
                  <i class="el-icon-pie-chart" v-if="config.hasFund"></i>
                  <i class="el-icon-money" v-if="!config.hasStock && !config.hasFund"></i>
                  {{ getConfigTypeText(config) }}
                </span>
                <span class="config-count">{{ config.portfolioCount }}项</span>
                <span class="config-date">{{ formatDate(config.lastUsed) }}</span>
              </div>
            </div>
            <div class="config-actions">
              <el-button 
                type="text" 
                size="mini" 
                @click.stop="deleteConfig(config.id)"
                icon="el-icon-delete"
                class="delete-btn"
              >
                删除
              </el-button>
            </div>
          </div>
        </div>

        <!-- 统计信息 -->
        <div class="config-stats">
          <span>共 {{ configs.length }} 个配置</span>
          <el-button 
            type="text" 
            size="mini" 
            @click="clearAllConfigs"
            v-if="configs.length > 0"
            class="clear-all-btn"
          >
            清空所有
          </el-button>
        </div>
      </div>

      <div slot="footer" class="dialog-footer">
        <el-button @click="showConfigSelector = false" :size="isMobile ? 'small' : 'medium'">
          关闭
        </el-button>
      </div>
    </el-dialog>

    <!-- 保存配置对话框 -->
    <el-dialog
      :title="'保存回测配置'"
      :visible.sync="showSaveDialog"
      :width="isMobile ? '95%' : '500px'"
      :close-on-click-modal="true"
      :close-on-press-escape="true"
      :modal="true"
      :append-to-body="true"
      :class="['save-config-dialog', { 'mobile-dialog': isMobile }]"
    >
      <div class="save-config-content">
        <el-form :model="saveForm" :label-width="isMobile ? '70px' : '80px'">
          <el-form-item label="配置名称">
            <el-input
              v-model="saveForm.name"
              :placeholder="'请输入配置名称'"
              :size="isMobile ? 'small' : 'medium'"
              maxlength="50"
              show-word-limit
            />
          </el-form-item>
          <el-form-item label="配置预览">
            <div class="config-preview" :class="{ 'mobile-preview': isMobile }">
              <div class="preview-item">
                <span class="preview-label">组合项数:</span>
                <span class="preview-value">{{ currentConfig.portfolioItems?.length || 0 }}项</span>
              </div>
              <div class="preview-item">
                <span class="preview-label">初始资金:</span>
                <span class="preview-value">¥{{ (currentConfig.initialCapital || 0).toLocaleString() }}</span>
              </div>
              <div class="preview-item">
                <span class="preview-label">再平衡策略:</span>
                <span class="preview-value">{{ getRebalanceStrategyText(currentConfig.rebalanceStrategy) }}</span>
              </div>
            </div>
          </el-form-item>
        </el-form>
      </div>

      <div slot="footer" class="dialog-footer">
        <el-button @click="showSaveDialog = false">取消</el-button>
        <el-button type="primary" @click="saveConfig" :loading="saving">
          保存配置
        </el-button>
      </div>
    </el-dialog>
  </div>
</template>

<script>
import portfolioConfigManager from '@/utils/portfolioConfigManager'

export default {
  name: 'PortfolioConfigManager',
  props: {
    isMobile: {
      type: Boolean,
      default: false
    },
    currentConfig: {
      type: Object,
      default: () => ({})
    }
  },
  data() {
    return {
      showConfigSelector: false,
      showSaveDialog: false,
      searchKeyword: '',
      configs: [],
      saveForm: {
        name: ''
      },
      saving: false
    }
  },
  computed: {
    filteredConfigs() {
      if (!this.searchKeyword) {
        return this.configs
      }
      return this.configs.filter(config => 
        config.name.toLowerCase().includes(this.searchKeyword.toLowerCase())
      )
    }
  },
  watch: {
    showConfigSelector(newVal) {
      if (newVal) {
        this.loadConfigs()
      } else {
        // 关闭时清理搜索关键词
        this.searchKeyword = ''
      }
    }
  },
  methods: {
    /**
     * 加载配置列表
     */
    loadConfigs() {
      this.configs = portfolioConfigManager.getAllConfigs()
      console.log('加载的配置列表:', this.configs)
      console.log('过滤后的配置列表:', this.filteredConfigs)
    },

    /**
     * 选择配置
     */
    selectConfig(config) {
      const result = portfolioConfigManager.loadConfig(config.id)
      if (result.success) {
        this.$emit('load-config', result.config)
        this.showConfigSelector = false
        this.$message.success('配置加载成功')
      } else {
        this.$message.error(result.message)
      }
    },

    /**
     * 删除配置
     */
    deleteConfig(configId) {
      this.$confirm('确定要删除这个配置吗？', '确认删除', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }).then(() => {
        const result = portfolioConfigManager.deleteConfig(configId)
        if (result.success) {
          this.loadConfigs()
          this.$message.success('配置删除成功')
        } else {
          this.$message.error(result.message)
        }
      }).catch(() => {
        // 用户取消删除
      })
    },

    /**
     * 清空所有配置
     */
    clearAllConfigs() {
      this.$confirm('确定要清空所有配置吗？此操作不可恢复！', '确认清空', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }).then(() => {
        const result = portfolioConfigManager.clearAllConfigs()
        if (result.success) {
          this.loadConfigs()
          this.$message.success('所有配置已清空')
        } else {
          this.$message.error(result.message)
        }
      }).catch(() => {
        // 用户取消清空
      })
    },

    /**
     * 显示保存对话框
     */
    showSaveConfigDialog() {
      this.saveForm.name = portfolioConfigManager.generateDefaultName(this.currentConfig)
      this.showSaveDialog = true
    },

    /**
     * 保存配置
     */
    saveConfig() {
      if (!this.saveForm.name.trim()) {
        this.$message.warning('请输入配置名称')
        return
      }

      this.saving = true
      const result = portfolioConfigManager.saveConfig(this.currentConfig, this.saveForm.name.trim())
      
      setTimeout(() => {
        this.saving = false
        if (result.success) {
          this.showSaveDialog = false
          this.$message.success('配置保存成功')
        } else {
          this.$message.error(result.message)
        }
      }, 500)
    },

    /**
     * 获取配置类型文本
     */
    getConfigTypeText(config) {
      if (config.hasStock && config.hasFund) return '股票+ETF'
      if (config.hasStock) return '股票'
      if (config.hasFund) return 'ETF'
      return '现金'
    },

    /**
     * 获取再平衡策略文本
     */
    getRebalanceStrategyText(strategy) {
      if (!strategy) return '无'
      const types = {
        'none': '无再平衡',
        'time': '定期再平衡',
        'threshold': '阈值再平衡'
      }
      return types[strategy.type] || '未知'
    },

    /**
     * 格式化日期
     */
    formatDate(dateString) {
      const date = new Date(dateString)
      const now = new Date()
      const diff = now - date
      const days = Math.floor(diff / (1000 * 60 * 60 * 24))
      
      if (days === 0) return '今天'
      if (days === 1) return '昨天'
      if (days < 7) return `${days}天前`
      if (days < 30) return `${Math.floor(days / 7)}周前`
      if (days < 365) return `${Math.floor(days / 30)}个月前`
      return date.toLocaleDateString('zh-CN')
    }
  }
}
</script>

<style scoped>
.config-manager {
  display: inline-block;
}

/* 配置选择对话框 */
.config-selector-dialog :deep(.el-dialog__body) {
  padding: 20px;
}

.mobile-dialog :deep(.el-dialog__body) {
  padding: 15px;
}

/* 确保对话框遮罩层正确显示 */
.config-selector-dialog :deep(.el-dialog__wrapper) {
  z-index: 2000;
}

.save-config-dialog :deep(.el-dialog__wrapper) {
  z-index: 2001;
}

/* 移动端对话框高度优化 */
.mobile-dialog.config-selector-dialog :deep(.el-dialog) {
  height: auto;
  max-height: 90vh;
}

.mobile-dialog.config-selector-dialog :deep(.el-dialog__body) {
  padding: 15px;
  height: auto;
}

.config-selector-content {
  max-height: 60vh;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

/* 移动端优化配置选择对话框 */
.mobile-dialog .config-selector-content {
  max-height: none;
  overflow: visible;
  min-height: auto;
}

.search-section {
  margin-bottom: 16px;
}

/* 移动端优化搜索区域 */
.mobile-dialog .search-section {
  margin-bottom: 12px;
}

.config-list {
  flex: 1;
  overflow-y: auto;
  border: 1px solid #e4e7ed;
  border-radius: 4px;
  max-height: 400px;
}

.mobile-config-list {
  max-height: 300px;
}

/* 移动端优化配置列表 */
.mobile-dialog .config-list {
  max-height: none;
  overflow-y: visible;
  border: none;
  border-radius: 0;
  min-height: auto;
}

.empty-state {
  text-align: center;
  padding: 40px 20px;
  color: #909399;
}

/* 移动端优化空状态 */
.mobile-dialog .empty-state {
  padding: 20px 15px;
}

.empty-icon {
  font-size: 48px;
  margin-bottom: 16px;
  display: block;
}

/* 移动端优化图标 */
.mobile-dialog .empty-icon {
  font-size: 36px;
  margin-bottom: 12px;
}

.empty-tips p {
  margin: 8px 0;
}

.empty-hint {
  font-size: 12px;
  color: #c0c4cc;
  margin-top: 8px;
}

.config-item {
  display: flex;
  align-items: center;
  padding: 12px 16px;
  border-bottom: 1px solid #f0f0f0;
  cursor: pointer;
  transition: background-color 0.2s;
}

.config-item:hover {
  background-color: #f5f7fa;
}

.config-item:last-child {
  border-bottom: none;
}

.mobile-config-item {
  padding: 10px 12px;
}

.config-info {
  flex: 1;
  min-width: 0;
}

.config-name {
  font-weight: 500;
  color: #303133;
  margin-bottom: 4px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.config-meta {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 12px;
  color: #909399;
}

.config-type {
  display: flex;
  align-items: center;
  gap: 4px;
}

.config-actions {
  display: flex;
  align-items: center;
}

.delete-btn {
  color: #f56c6c;
}

.delete-btn:hover {
  color: #f78989;
}

.config-stats {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 0;
  border-top: 1px solid #e4e7ed;
  font-size: 12px;
  color: #909399;
}

/* 移动端优化统计信息 */
.mobile-dialog .config-stats {
  padding: 8px 0;
  border-top: none;
}

.clear-all-btn {
  color: #f56c6c;
}

/* 保存配置对话框 */
.save-config-dialog :deep(.el-dialog__body) {
  padding: 20px;
}

.mobile-dialog.save-config-dialog :deep(.el-dialog__body) {
  padding: 15px;
}

.save-config-content {
  max-height: 50vh;
  overflow-y: auto;
}

/* 移动端优化保存配置对话框 */
.mobile-dialog .save-config-content {
  max-height: none;
  overflow-y: visible;
}

.config-preview {
  background: #f8f9fa;
  padding: 12px;
  border-radius: 4px;
  border: 1px solid #e4e7ed;
}

.mobile-preview {
  padding: 10px;
}

.preview-item {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
}

.preview-item:last-child {
  margin-bottom: 0;
}

.preview-label {
  color: #606266;
  font-size: 13px;
}

.mobile-preview .preview-label {
  font-size: 12px;
}

.preview-value {
  color: #303133;
  font-weight: 500;
  font-size: 13px;
}

.mobile-preview .preview-value {
  font-size: 12px;
}

/* 移动端优化 */
@media (max-width: 768px) {
  .config-meta {
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
  }
  
  .config-actions {
    margin-left: 8px;
  }
  
  .config-stats {
    flex-direction: column;
    gap: 8px;
    align-items: flex-start;
  }
  
  /* 移动端表单优化 */
  .mobile-dialog .save-config-content :deep(.el-form-item) {
    margin-bottom: 16px;
  }
  
  .mobile-dialog .save-config-content :deep(.el-form-item__label) {
    line-height: 1.4;
  }
}

/* 滚动条样式 */
.config-list::-webkit-scrollbar {
  width: 6px;
}

.config-list::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 3px;
}

.config-list::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 3px;
}

.config-list::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}
</style>
