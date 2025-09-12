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
    // 输入框聚焦时的处理
    if (this.isMobile && this.isInputElement(event.target)) {
      this.stabilizeViewport()
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
    
    // 防止页面滚动
    document.body.style.position = 'fixed'
    document.body.style.width = '100%'
    document.body.style.top = '0'
    
    // 设置视口高度为当前高度
    document.documentElement.style.setProperty('--viewport-height', `${this.currentViewportHeight}px`)
  }

  restoreViewport() {
    if (!this.isMobile) return
    
    // 恢复页面滚动
    document.body.style.position = ''
    document.body.style.width = ''
    document.body.style.top = ''
    
    // 恢复视口高度
    document.documentElement.style.setProperty('--viewport-height', `${this.initialViewportHeight}px`)
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
