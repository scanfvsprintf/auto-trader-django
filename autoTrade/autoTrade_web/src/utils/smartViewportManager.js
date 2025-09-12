/**
 * 智能视口管理器 - 主动适应键盘变化
 * 不再试图阻止键盘弹起，而是优雅地适应这种变化
 */

import orientationHelper from './orientationHelper'

class SmartViewportManager {
  constructor() {
    this.isMobile = false
    this.isPortrait = true
    this.initialViewportHeight = 0
    this.currentViewportHeight = 0
    this.isKeyboardOpen = false
    this.keyboardHeight = 0
    this.resizeTimeout = null
    this.focusTimeout = null
    
    this.init()
  }

  init() {
    this.detectDevice()
    this.initialViewportHeight = window.innerHeight
    this.currentViewportHeight = this.initialViewportHeight
    
    this.bindEvents()
    this.updateCSSVariables()
  }

  detectDevice() {
    // 使用智能横屏检测工具
    const orientation = orientationHelper.detectOrientation()
    this.isMobile = orientation.isMobile
    this.isPortrait = orientation.isPortrait
  }

  bindEvents() {
    // 使用更温和的resize处理
    window.addEventListener('resize', this.handleResize.bind(this))
    window.addEventListener('orientationchange', this.handleOrientationChange.bind(this))
    
    // 监听输入框事件
    document.addEventListener('focusin', this.handleFocusIn.bind(this))
    document.addEventListener('focusout', this.handleFocusOut.bind(this))
    
    // 使用Visual Viewport API（如果支持）
    if ('visualViewport' in window) {
      window.visualViewport.addEventListener('resize', this.handleVisualViewportResize.bind(this))
    }
  }

  handleResize() {
    clearTimeout(this.resizeTimeout)
    this.resizeTimeout = setTimeout(() => {
      const newWidth = window.innerWidth
      const newHeight = window.innerHeight
      
      // 检查是否是键盘弹起（高度变化但宽度不变）
      const heightDiff = Math.abs(newHeight - this.initialViewportHeight)
      const widthDiff = Math.abs(newWidth - (window.innerWidth || 1024))
      
      if (heightDiff > 100 && widthDiff < 50) {
        // 很可能是键盘弹起，不改变横竖屏状态
        this.currentViewportHeight = newHeight
        this.detectKeyboardState()
        this.updateCSSVariables()
      } else if (widthDiff > 50 || heightDiff > 200) {
        // 可能是真正的方向变化或窗口大小变化
        this.detectDevice()
        this.currentViewportHeight = newHeight
        this.detectKeyboardState()
        this.updateCSSVariables()
      } else {
        // 小幅变化，只更新高度
        this.currentViewportHeight = newHeight
        this.detectKeyboardState()
        this.updateCSSVariables()
      }
    }, 150) // 增加延迟，让浏览器完成布局
  }

  handleOrientationChange() {
    // 真正的方向变化处理
    setTimeout(() => {
      // 重置横屏检测工具的状态
      orientationHelper.reset()
      
      // 更新视口高度
      this.initialViewportHeight = window.innerHeight
      this.currentViewportHeight = window.innerHeight
      
      // 重新检测设备状态
      this.detectDevice()
      this.updateCSSVariables()
    }, 600) // 等待方向变化完成
  }

  handleFocusIn(event) {
    if (this.isMobile && this.isInputElement(event.target)) {
      // 延迟处理，让键盘有时间弹起
      clearTimeout(this.focusTimeout)
      this.focusTimeout = setTimeout(() => {
        this.adaptToKeyboard(event.target)
      }, 300)
    }
  }

  handleFocusOut(event) {
    if (this.isMobile && this.isInputElement(event.target)) {
      clearTimeout(this.focusTimeout)
      setTimeout(() => {
        this.restoreFromKeyboard()
      }, 300)
    }
  }

  handleVisualViewportResize() {
    if (this.isMobile && window.visualViewport) {
      const viewport = window.visualViewport
      this.currentViewportHeight = viewport.height
      this.keyboardHeight = this.initialViewportHeight - viewport.height
      this.detectKeyboardState()
      this.updateCSSVariables()
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
    this.isKeyboardOpen = heightDiff > 150
    this.keyboardHeight = Math.max(0, heightDiff)
    
    if (this.isKeyboardOpen !== wasKeyboardOpen) {
      this.onKeyboardStateChange(this.isKeyboardOpen)
    }
  }

  onKeyboardStateChange(isOpen) {
    const body = document.body
    
    if (isOpen) {
      body.classList.add('keyboard-open')
      this.adaptLayoutToKeyboard()
    } else {
      body.classList.remove('keyboard-open')
      this.restoreLayoutFromKeyboard()
    }
  }

  // 主动适应键盘 - 核心方法
  adaptToKeyboard(inputElement) {
    if (!this.isMobile) return
    
    const rect = inputElement.getBoundingClientRect()
    const viewportHeight = this.currentViewportHeight
    const inputBottom = rect.bottom
    const inputTop = rect.top
    
    // 计算输入框在视口中的位置
    const inputCenter = (inputTop + inputBottom) / 2
    const viewportCenter = viewportHeight / 2
    
    // 如果输入框在视口下半部分，需要调整
    if (inputCenter > viewportCenter) {
      const scrollAmount = inputCenter - viewportCenter + 50 // 额外50px缓冲
      this.smoothScrollTo(scrollAmount)
    }
    
    // 设置CSS变量，让布局适应键盘
    this.updateLayoutForKeyboard()
  }

  // 平滑滚动
  smoothScrollTo(amount) {
    const currentScroll = window.pageYOffset || document.documentElement.scrollTop
    const targetScroll = currentScroll + amount
    
    // 使用requestAnimationFrame实现平滑滚动
    const startTime = performance.now()
    const duration = 300 // 300ms动画
    
    const animateScroll = (currentTime) => {
      const elapsed = currentTime - startTime
      const progress = Math.min(elapsed / duration, 1)
      
      // 使用easeOut缓动函数
      const easeOut = 1 - Math.pow(1 - progress, 3)
      const currentAmount = currentScroll + (amount * easeOut)
      
      window.scrollTo(0, currentAmount)
      
      if (progress < 1) {
        requestAnimationFrame(animateScroll)
      }
    }
    
    requestAnimationFrame(animateScroll)
  }

  // 更新布局以适应键盘
  updateLayoutForKeyboard() {
    const html = document.documentElement
    
    // 设置可用高度（减去键盘高度）
    const availableHeight = this.currentViewportHeight
    html.style.setProperty('--available-height', `${availableHeight}px`)
    html.style.setProperty('--keyboard-height', `${this.keyboardHeight}px`)
    
    // 调整容器高度
    this.adjustContainerHeights(availableHeight)
  }

  // 调整容器高度
  adjustContainerHeights(availableHeight) {
    const containers = document.querySelectorAll('.layout, .layout-mobile-portrait, .selection-root, .ai-config-root, .system-root')
    
    containers.forEach(container => {
      // 计算合适的高度（减去底部导航等固定元素）
      const bottomNavHeight = 60 // 底部导航高度
      const headerHeight = 50 // 头部高度
      const adjustedHeight = availableHeight - bottomNavHeight - headerHeight
      
      container.style.height = `${adjustedHeight}px`
      container.style.maxHeight = `${adjustedHeight}px`
    })
  }

  // 从键盘状态恢复
  restoreFromKeyboard() {
    if (!this.isMobile) return
    
    // 恢复CSS变量
    const html = document.documentElement
    html.style.setProperty('--available-height', `${this.initialViewportHeight}px`)
    html.style.setProperty('--keyboard-height', '0px')
    
    // 恢复容器高度
    this.restoreContainerHeights()
  }

  // 恢复容器高度
  restoreContainerHeights() {
    const containers = document.querySelectorAll('.layout, .layout-mobile-portrait, .selection-root, .ai-config-root, .system-root')
    
    containers.forEach(container => {
      container.style.height = ''
      container.style.maxHeight = ''
    })
  }

  // 适应键盘布局
  adaptLayoutToKeyboard() {
    this.updateLayoutForKeyboard()
  }

  // 从键盘恢复布局
  restoreLayoutFromKeyboard() {
    this.restoreFromKeyboard()
  }

  updateCSSVariables() {
    const root = document.documentElement
    
    root.style.setProperty('--viewport-height', `${this.currentViewportHeight}px`)
    root.style.setProperty('--initial-viewport-height', `${this.initialViewportHeight}px`)
    root.style.setProperty('--available-height', `${this.currentViewportHeight}px`)
    root.style.setProperty('--keyboard-height', `${this.keyboardHeight}px`)
    root.style.setProperty('--is-mobile', this.isMobile ? '1' : '0')
    root.style.setProperty('--is-portrait', this.isPortrait ? '1' : '0')
  }

  getViewportInfo() {
    return {
      isMobile: this.isMobile,
      isPortrait: this.isPortrait,
      initialHeight: this.initialViewportHeight,
      currentHeight: this.currentViewportHeight,
      availableHeight: this.currentViewportHeight,
      keyboardHeight: this.keyboardHeight,
      isKeyboardOpen: this.isKeyboardOpen
    }
  }

  destroy() {
    clearTimeout(this.resizeTimeout)
    clearTimeout(this.focusTimeout)
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
const smartViewportManager = new SmartViewportManager()

export default smartViewportManager
export { SmartViewportManager }
