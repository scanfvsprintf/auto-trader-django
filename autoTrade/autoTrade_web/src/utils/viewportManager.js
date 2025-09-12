/**
 * 移动端视口管理器
 * 解决键盘弹起时的界面闪烁和跳转问题
 */

class ViewportManager {
  constructor() {
    this.isMobile = false
    this.isPortrait = true
    this.initialViewportHeight = 0
    this.currentViewportHeight = 0
    this.resizeTimeout = null
    this.isKeyboardOpen = false
    
    this.init()
  }

  init() {
    // 检测设备类型
    this.detectDevice()
    
    // 记录初始视口高度
    this.initialViewportHeight = window.innerHeight
    this.currentViewportHeight = this.initialViewportHeight
    
    // 移动端立即锁定视口
    if (this.isMobile) {
      this.lockViewportImmediately()
    }
    
    // 监听视口变化
    this.bindEvents()
    
    // 设置CSS环境变量
    this.updateCSSVariables()
  }

  detectDevice() {
    const w = window.innerWidth || 1024
    const h = window.innerHeight || 768
    this.isMobile = w <= 768
    this.isPortrait = h >= w
  }

  bindEvents() {
    // 防抖处理resize事件
    window.addEventListener('resize', this.handleResize.bind(this))
    window.addEventListener('orientationchange', this.handleOrientationChange.bind(this))
    
    // 监听输入框聚焦/失焦
    document.addEventListener('focusin', this.handleFocusIn.bind(this))
    document.addEventListener('focusout', this.handleFocusOut.bind(this))
    
    // 监听虚拟键盘事件（如果支持）
    if ('visualViewport' in window) {
      window.visualViewport.addEventListener('resize', this.handleVisualViewportResize.bind(this))
    }
  }

  handleResize() {
    // 防抖处理
    clearTimeout(this.resizeTimeout)
    this.resizeTimeout = setTimeout(() => {
      this.detectDevice()
      this.currentViewportHeight = window.innerHeight
      this.updateCSSVariables()
      this.detectKeyboardState()
    }, 100)
  }

  handleOrientationChange() {
    // 延迟处理，等待方向变化完成
    setTimeout(() => {
      this.detectDevice()
      this.initialViewportHeight = window.innerHeight
      this.currentViewportHeight = this.initialViewportHeight
      this.updateCSSVariables()
    }, 500)
  }

  handleFocusIn(event) {
    // 输入框聚焦时的处理 - 更激进的方案
    if (this.isMobile && this.isInputElement(event.target)) {
      // 立即稳定视口，不等待任何延迟
      this.stabilizeViewport()
      
      // 强制滚动到输入框位置
      setTimeout(() => {
        this.scrollToInput(event.target)
      }, 50)
    }
  }

  handleFocusOut(event) {
    // 输入框失焦时的处理
    if (this.isMobile && this.isInputElement(event.target)) {
      setTimeout(() => {
        this.restoreViewport()
      }, 300)
    }
  }

  handleVisualViewportResize() {
    // 使用Visual Viewport API（更精确）
    if (this.isMobile) {
      const viewport = window.visualViewport
      this.currentViewportHeight = viewport.height
      this.updateCSSVariables()
      this.detectKeyboardState()
    }
  }

  isInputElement(element) {
    const inputTypes = ['input', 'textarea', 'select']
    return inputTypes.includes(element.tagName.toLowerCase()) ||
           element.contentEditable === 'true' ||
           element.classList.contains('el-input__inner') ||
           element.classList.contains('el-textarea__inner')
  }

  detectKeyboardState() {
    if (!this.isMobile) return
    
    const heightDiff = this.initialViewportHeight - this.currentViewportHeight
    const wasKeyboardOpen = this.isKeyboardOpen
    this.isKeyboardOpen = heightDiff > 150 // 键盘高度阈值
    
    // 键盘状态变化时的处理
    if (this.isKeyboardOpen !== wasKeyboardOpen) {
      this.onKeyboardStateChange(this.isKeyboardOpen)
    }
  }

  onKeyboardStateChange(isOpen) {
    const body = document.body
    
    if (isOpen) {
      // 键盘弹起时的处理
      body.classList.add('keyboard-open')
      this.stabilizeViewport()
    } else {
      // 键盘收起时的处理
      body.classList.remove('keyboard-open')
      this.restoreViewport()
    }
  }

  stabilizeViewport() {
    if (!this.isMobile) return
    
    // 更激进的视口稳定方案
    const body = document.body
    const html = document.documentElement
    
    // 记录当前滚动位置
    this.scrollTop = window.pageYOffset || document.documentElement.scrollTop
    
    // 完全固定页面
    body.style.position = 'fixed'
    body.style.width = '100%'
    body.style.height = '100%'
    body.style.top = `-${this.scrollTop}px`
    body.style.left = '0'
    body.style.overflow = 'hidden'
    
    // 设置视口高度为初始高度，完全忽略键盘变化
    html.style.setProperty('--viewport-height', `${this.initialViewportHeight}px`)
    html.style.setProperty('--actual-viewport-height', `${this.currentViewportHeight}px`)
    
    // 强制所有容器使用固定高度
    this.forceFixedHeights()
  }

  restoreViewport() {
    if (!this.isMobile) return
    
    const body = document.body
    const html = document.documentElement
    
    // 恢复页面滚动
    body.style.position = ''
    body.style.width = ''
    body.style.height = ''
    body.style.top = ''
    body.style.left = ''
    body.style.overflow = ''
    
    // 恢复滚动位置
    if (this.scrollTop !== undefined) {
      window.scrollTo(0, this.scrollTop)
    }
    
    // 恢复视口高度
    html.style.setProperty('--viewport-height', `${this.initialViewportHeight}px`)
    html.style.setProperty('--actual-viewport-height', `${this.currentViewportHeight}px`)
    
    // 恢复容器高度
    this.restoreContainerHeights()
  }

  updateCSSVariables() {
    const root = document.documentElement
    
    // 设置视口高度变量
    root.style.setProperty('--viewport-height', `${this.currentViewportHeight}px`)
    root.style.setProperty('--initial-viewport-height', `${this.initialViewportHeight}px`)
    
    // 设置设备类型变量
    root.style.setProperty('--is-mobile', this.isMobile ? '1' : '0')
    root.style.setProperty('--is-portrait', this.isPortrait ? '1' : '0')
  }

  // 强制固定容器高度
  forceFixedHeights() {
    const containers = document.querySelectorAll('.layout, .layout-mobile-portrait, .selection-root, .ai-config-root, .system-root')
    this.originalHeights = new Map()
    
    containers.forEach(container => {
      this.originalHeights.set(container, container.style.height)
      container.style.height = `${this.initialViewportHeight}px`
      container.style.maxHeight = `${this.initialViewportHeight}px`
      container.style.overflow = 'hidden'
    })
  }
  
  // 恢复容器高度
  restoreContainerHeights() {
    if (this.originalHeights) {
      this.originalHeights.forEach((originalHeight, container) => {
        container.style.height = originalHeight
        container.style.maxHeight = ''
        container.style.overflow = ''
      })
      this.originalHeights.clear()
    }
  }
  
  // 滚动到输入框位置
  scrollToInput(inputElement) {
    if (!inputElement) return
    
    const rect = inputElement.getBoundingClientRect()
    const viewportHeight = this.initialViewportHeight
    const inputBottom = rect.bottom
    const inputTop = rect.top
    
    // 如果输入框在视口下半部分，滚动到合适位置
    if (inputBottom > viewportHeight * 0.6) {
      const scrollAmount = inputBottom - viewportHeight * 0.4
      const currentScroll = this.scrollTop || 0
      this.scrollTop = Math.max(0, currentScroll + scrollAmount)
      
      // 更新body的top位置
      document.body.style.top = `-${this.scrollTop}px`
    }
  }
  
  // 立即锁定视口 - 最激进的方案
  lockViewportImmediately() {
    const html = document.documentElement
    const body = document.body
    
    // 立即设置固定高度
    html.style.height = `${this.initialViewportHeight}px`
    html.style.maxHeight = `${this.initialViewportHeight}px`
    html.style.overflow = 'hidden'
    
    body.style.height = `${this.initialViewportHeight}px`
    body.style.maxHeight = `${this.initialViewportHeight}px`
    body.style.overflow = 'hidden'
    
    // 设置CSS变量
    html.style.setProperty('--viewport-height', `${this.initialViewportHeight}px`)
    html.style.setProperty('--locked-viewport', '1')
    
    // 强制所有容器使用固定高度
    this.forceFixedHeights()
    
    // 添加锁定类
    body.classList.add('viewport-locked')
  }

  // 公共方法：获取当前视口信息
  getViewportInfo() {
    return {
      isMobile: this.isMobile,
      isPortrait: this.isPortrait,
      initialHeight: this.initialViewportHeight,
      currentHeight: this.currentViewportHeight,
      isKeyboardOpen: this.isKeyboardOpen
    }
  }

  // 销毁方法
  destroy() {
    clearTimeout(this.resizeTimeout)
    window.removeEventListener('resize', this.handleResize.bind(this))
    window.removeEventListener('orientationchange', this.handleOrientationChange.bind(this))
    document.removeEventListener('focusin', this.handleFocusIn.bind(this))
    document.removeEventListener('focusout', this.handleFocusOut.bind(this))
    
    if ('visualViewport' in window) {
      window.visualViewport.removeEventListener('resize', this.handleVisualViewportResize.bind(this))
    }
  }
}

// 创建全局实例
const viewportManager = new ViewportManager()

// 导出实例和类
export default viewportManager
export { ViewportManager }
