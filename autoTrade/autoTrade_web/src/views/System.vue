<template>
  <div class="system-root" v-loading="loading" element-loading-text="加载中...">
    <!-- 非竖屏移动端：标准布局 -->
    <div v-if="!isPortraitMobile" class="system-desktop">
      <el-card>
        <div slot="header">
          <span>系统管理</span>
        </div>
        
        <div class="system-grid">
          <div class="system-card" @click="navigateToSchema">
            <div class="system-card-icon">
              <i class="el-icon-database"></i>
            </div>
            <div class="system-card-content">
              <h3>Schema 管理</h3>
              <p>管理数据库Schema，包括创建、删除等操作</p>
            </div>
            <div class="system-card-arrow">
              <i class="el-icon-arrow-right"></i>
            </div>
          </div>
          
          <div class="system-card" @click="navigateToAiConfig">
            <div class="system-card-icon">
              <i class="el-icon-cpu"></i>
            </div>
            <div class="system-card-content">
              <h3>AI 模型配置</h3>
              <p>配置AI服务源和模型参数，管理AI功能</p>
            </div>
            <div class="system-card-arrow">
              <i class="el-icon-arrow-right"></i>
            </div>
          </div>
        </div>
      </el-card>
    </div>

    <!-- 竖屏移动端：优化布局 -->
    <div v-else class="system-mobile">
      <!-- 顶部工具栏 -->
      <div class="mobile-toolbar">
        <div class="mobile-toolbar-title">系统管理</div>
      </div>

      <!-- 内容区域 -->
      <div class="mobile-content">
        <div class="mobile-section">
          <div class="mobile-section-header">
            <span>管理模块</span>
          </div>
          
          <div class="mobile-card-list">
            <el-card class="mobile-card" @click.native="navigateToSchema">
              <div class="mobile-card-header">
                <div class="mobile-card-icon">
                  <i class="el-icon-database"></i>
                </div>
                <div class="mobile-card-content">
                  <div class="mobile-card-title">Schema 管理</div>
                  <div class="mobile-card-desc">管理数据库Schema</div>
                </div>
                <div class="mobile-card-arrow">
                  <i class="el-icon-arrow-right"></i>
                </div>
              </div>
            </el-card>
            
            <el-card class="mobile-card" @click.native="navigateToAiConfig">
              <div class="mobile-card-header">
                <div class="mobile-card-icon">
                  <i class="el-icon-cpu"></i>
                </div>
                <div class="mobile-card-content">
                  <div class="mobile-card-title">AI 模型配置</div>
                  <div class="mobile-card-desc">配置AI服务源和模型</div>
                </div>
                <div class="mobile-card-arrow">
                  <i class="el-icon-arrow-right"></i>
                </div>
              </div>
            </el-card>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'System',
  data() {
    return {
      loading: false,
      isMobile: false,
      isPortrait: true
    }
  },
  computed: {
    isPortraitMobile() {
      return this.isMobile && this.isPortrait
    }
  },
  created() {
    this.onResize()
    window.addEventListener('resize', this.onResize)
    window.addEventListener('orientationchange', this.onResize)
  },
  beforeDestroy() {
    window.removeEventListener('resize', this.onResize)
    window.removeEventListener('orientationchange', this.onResize)
  },
  methods: {
    onResize() {
      try {
        const w = window.innerWidth || 1024
        const h = window.innerHeight || 768
        this.isMobile = w <= 768
        this.isPortrait = h >= w
      } catch (e) {
        this.isMobile = false
        this.isPortrait = true
      }
    },
    navigateToSchema() {
      this.$router.push('/system/schema')
    },
    navigateToAiConfig() {
      this.$router.push('/ai-config')
    }
  }
}
</script>

<style scoped>
/* 桌面端样式 */
.system-desktop {
  padding: 20px;
}

.system-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  margin-top: 20px;
}

.system-card {
  display: flex;
  align-items: center;
  padding: 20px;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s ease;
  background: white;
}

.system-card:hover {
  border-color: #2563eb;
  box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
  transform: translateY(-2px);
}

.system-card-icon {
  width: 48px;
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f3f4f6;
  border-radius: 8px;
  margin-right: 16px;
  font-size: 24px;
  color: #6b7280;
}

.system-card:hover .system-card-icon {
  background: #dbeafe;
  color: #2563eb;
}

.system-card-content {
  flex: 1;
}

.system-card-content h3 {
  margin: 0 0 8px 0;
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
}

.system-card-content p {
  margin: 0;
  font-size: 14px;
  color: #6b7280;
  line-height: 1.4;
}

.system-card-arrow {
  color: #9ca3af;
  font-size: 16px;
}

.system-card:hover .system-card-arrow {
  color: #2563eb;
}

/* 移动端样式 */
.system-mobile {
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.mobile-toolbar {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 12px 16px;
  background: #f7f9fe;
  border-bottom: 1px solid #e6ebf2;
  flex-shrink: 0;
}

.mobile-toolbar-title {
  font-size: 16px;
  font-weight: 600;
  color: #1F2937;
}

.mobile-content {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  padding-bottom: 80px; /* 为底部导航留出空间 */
}

.mobile-section {
  margin-bottom: 24px;
}

.mobile-section-header {
  margin-bottom: 16px;
  padding-bottom: 8px;
  border-bottom: 1px solid #e6ebf2;
}

.mobile-section-header span {
  font-size: 16px;
  font-weight: 600;
  color: #1F2937;
}

.mobile-card-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.mobile-card {
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  cursor: pointer;
  transition: all 0.2s;
}

.mobile-card:active {
  transform: scale(0.98);
}

.mobile-card-header {
  display: flex;
  align-items: center;
  padding: 16px;
}

.mobile-card-icon {
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f3f4f6;
  border-radius: 8px;
  margin-right: 12px;
  font-size: 20px;
  color: #6b7280;
}

.mobile-card-content {
  flex: 1;
}

.mobile-card-title {
  font-size: 15px;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 4px;
}

.mobile-card-desc {
  font-size: 13px;
  color: #6b7280;
}

.mobile-card-arrow {
  color: #9ca3af;
  font-size: 14px;
}

/* 滚动条优化 */
.mobile-content::-webkit-scrollbar {
  width: 4px;
}

.mobile-content::-webkit-scrollbar-track {
  background: #f1f1f1;
}

.mobile-content::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 2px;
}

.mobile-content::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}
</style>
