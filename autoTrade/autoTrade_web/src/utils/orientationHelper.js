/**
 * 横屏检测辅助工具
 * 解决键盘弹起时误判为横屏的问题
 */

class OrientationHelper {
  constructor() {
    this.lastWidth = 0
    this.lastHeight = 0
    this.isPortrait = true
    this.isMobile = false
    this.initialHeight = 0
  }

  // 智能检测设备方向，避免键盘弹起时的误判
  detectOrientation() {
    const w = window.innerWidth || 1024
    const h = window.innerHeight || 768
    
    this.isMobile = w <= 768
    
    // 记录初始高度
    if (!this.initialHeight) {
      this.initialHeight = h
    }
    
    // 检查是否是键盘弹起（高度变化但宽度不变）
    const heightDiff = Math.abs(h - this.initialHeight)
    const widthDiff = Math.abs(w - this.lastWidth)
    
    if (heightDiff > 100 && widthDiff < 50) {
      // 很可能是键盘弹起，保持当前的横竖屏状态
      // 不更新isPortrait
    } else if (widthDiff > 50 || heightDiff > 200) {
      // 可能是真正的方向变化
      this.isPortrait = h >= w
    } else if (!this.lastHeight || Math.abs(h - this.lastHeight) < 100) {
      // 小幅变化，正常检测
      this.isPortrait = h >= w
    }
    
    this.lastWidth = w
    this.lastHeight = h
    
    return {
      isMobile: this.isMobile,
      isPortrait: this.isPortrait,
      width: w,
      height: h
    }
  }

  // 重置状态（用于真正的方向变化）
  reset() {
    this.initialHeight = window.innerHeight
    this.lastWidth = window.innerWidth
    this.lastHeight = window.innerHeight
    this.isPortrait = window.innerHeight >= window.innerWidth
  }

  // 获取当前状态
  getState() {
    return {
      isMobile: this.isMobile,
      isPortrait: this.isPortrait,
      width: this.lastWidth,
      height: this.lastHeight
    }
  }
}

// 创建全局实例
const orientationHelper = new OrientationHelper()

export default orientationHelper
export { OrientationHelper }
