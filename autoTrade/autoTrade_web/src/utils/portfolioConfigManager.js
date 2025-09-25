/**
 * 组合回测配置管理器
 * 支持本地存储100次回测配置，提供配置的保存、加载、删除等功能
 */

class PortfolioConfigManager {
  constructor() {
    this.STORAGE_KEY = 'portfolio_backtest_configs'
    this.MAX_CONFIGS = 100
    this.configs = this.loadConfigs()
  }

  /**
   * 从localStorage加载配置列表
   */
  loadConfigs() {
    try {
      const stored = localStorage.getItem(this.STORAGE_KEY)
      return stored ? JSON.parse(stored) : []
    } catch (error) {
      console.error('加载配置失败:', error)
      return []
    }
  }

  /**
   * 保存配置列表到localStorage
   */
  saveConfigs() {
    try {
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(this.configs))
      return true
    } catch (error) {
      console.error('保存配置失败:', error)
      return false
    }
  }

  /**
   * 生成配置的唯一ID
   */
  generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substr(2)
  }

  /**
   * 生成配置的默认名称
   */
  generateDefaultName(config) {
    const date = new Date().toLocaleDateString('zh-CN')
    const portfolioCount = config.portfolioItems ? config.portfolioItems.length : 0
    const hasStock = config.portfolioItems?.some(item => item.type === 'stock')
    const hasFund = config.portfolioItems?.some(item => item.type === 'fund')
    
    let type = '现金'
    if (hasStock && hasFund) type = '股票+ETF'
    else if (hasStock) type = '股票'
    else if (hasFund) type = 'ETF'
    
    return `${date} ${type}组合(${portfolioCount}项)`
  }

  /**
   * 保存配置
   * @param {Object} config - 回测配置对象
   * @param {string} customName - 自定义名称（可选）
   * @returns {Object} 保存结果
   */
  saveConfig(config, customName = null) {
    try {
      const configData = {
        id: this.generateId(),
        name: customName || this.generateDefaultName(config),
        config: JSON.parse(JSON.stringify(config)), // 深拷贝
        createdAt: new Date().toISOString(),
        lastUsed: new Date().toISOString()
      }

      // 检查是否已存在相同配置
      const existingIndex = this.configs.findIndex(item => 
        JSON.stringify(item.config) === JSON.stringify(configData.config)
      )

      if (existingIndex >= 0) {
        // 更新现有配置
        this.configs[existingIndex].lastUsed = configData.lastUsed
        if (customName) {
          this.configs[existingIndex].name = customName
        }
      } else {
        // 添加新配置
        this.configs.unshift(configData)
        
        // 限制配置数量
        if (this.configs.length > this.MAX_CONFIGS) {
          this.configs = this.configs.slice(0, this.MAX_CONFIGS)
        }
      }

      // 按最后使用时间排序
      this.configs.sort((a, b) => new Date(b.lastUsed) - new Date(a.lastUsed))

      const success = this.saveConfigs()
      return {
        success,
        config: configData,
        message: success ? '配置保存成功' : '配置保存失败'
      }
    } catch (error) {
      console.error('保存配置时出错:', error)
      return {
        success: false,
        message: '保存配置时出错: ' + error.message
      }
    }
  }

  /**
   * 加载配置
   * @param {string} configId - 配置ID
   * @returns {Object} 加载结果
   */
  loadConfig(configId) {
    try {
      const config = this.configs.find(item => item.id === configId)
      if (!config) {
        return {
          success: false,
          message: '配置不存在'
        }
      }

      // 更新最后使用时间
      config.lastUsed = new Date().toISOString()
      this.saveConfigs()

      return {
        success: true,
        config: config.config,
        message: '配置加载成功'
      }
    } catch (error) {
      console.error('加载配置时出错:', error)
      return {
        success: false,
        message: '加载配置时出错: ' + error.message
      }
    }
  }

  /**
   * 删除配置
   * @param {string} configId - 配置ID
   * @returns {Object} 删除结果
   */
  deleteConfig(configId) {
    try {
      const index = this.configs.findIndex(item => item.id === configId)
      if (index === -1) {
        return {
          success: false,
          message: '配置不存在'
        }
      }

      this.configs.splice(index, 1)
      const success = this.saveConfigs()

      return {
        success,
        message: success ? '配置删除成功' : '配置删除失败'
      }
    } catch (error) {
      console.error('删除配置时出错:', error)
      return {
        success: false,
        message: '删除配置时出错: ' + error.message
      }
    }
  }

  /**
   * 获取所有配置列表
   * @returns {Array} 配置列表
   */
  getAllConfigs() {
    return this.configs.map(item => ({
      id: item.id,
      name: item.name,
      createdAt: item.createdAt,
      lastUsed: item.lastUsed,
      portfolioCount: item.config.portfolioItems?.length || 0,
      hasStock: item.config.portfolioItems?.some(i => i.type === 'stock') || false,
      hasFund: item.config.portfolioItems?.some(i => i.type === 'fund') || false
    }))
  }

  /**
   * 清空所有配置
   * @returns {Object} 清空结果
   */
  clearAllConfigs() {
    try {
      this.configs = []
      const success = this.saveConfigs()
      return {
        success,
        message: success ? '所有配置已清空' : '清空配置失败'
      }
    } catch (error) {
      console.error('清空配置时出错:', error)
      return {
        success: false,
        message: '清空配置时出错: ' + error.message
      }
    }
  }

  /**
   * 获取配置统计信息
   * @returns {Object} 统计信息
   */
  getStats() {
    return {
      total: this.configs.length,
      maxConfigs: this.MAX_CONFIGS,
      oldestConfig: this.configs.length > 0 ? this.configs[this.configs.length - 1].createdAt : null,
      newestConfig: this.configs.length > 0 ? this.configs[0].createdAt : null
    }
  }
}

// 创建单例实例
const portfolioConfigManager = new PortfolioConfigManager()

export default portfolioConfigManager
