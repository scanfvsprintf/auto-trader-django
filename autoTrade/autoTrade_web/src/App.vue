<template>
  <router-view />
  </template>

<script>
export default { name: 'App' }
  </script>

<style>
html, body, #app { 
  height: 100%; 
  height: 100dvh; /* 动态视口高度 */
  margin: 0; 
  /* 移动端视口稳定性 */
  position: relative;
  overflow-x: hidden;
}
.layout { 
  display: flex; 
  flex-direction: column; 
  height: 100%; 
  height: 100dvh; /* 动态视口高度 */
  min-height: 100vh; /* 降级方案 */
  position: relative;
}
.layout-header { 
  padding: 8px 12px; 
  border-bottom: 1px solid #eee; 
  display: flex; 
  justify-content: space-between; 
  align-items: center; 
  flex-shrink: 0; /* 防止压缩 */
}
.layout-body { 
  display: flex; 
  flex: 1; 
  min-height: 0; 
  position: relative;
}
.layout-sider { 
  width: 220px; 
  border-right: 1px solid #eee; 
  flex-shrink: 0; /* 防止压缩 */
}
.layout-content { 
  flex: 1; 
  padding: 12px; 
  overflow: auto; 
  position: relative;
  /* 移动端滚动优化 */
  -webkit-overflow-scrolling: touch;
}

/* 移动端特殊处理 */
@media (max-width: 768px) { 
  .layout-sider { width: 160px; }
  
  /* 防止键盘弹起时的布局跳动 */
  .layout {
    height: 100vh;
    height: 100dvh;
  }
  
  /* 输入框聚焦时的视口稳定 */
  .layout-content {
    transform: translateZ(0); /* 硬件加速 */
  }
}

/* 竖屏移动端布局优化 */
@media (max-width: 768px) and (orientation: portrait) {
  .layout {
    height: 100vh;
    height: 100dvh;
    overflow: hidden; /* 防止整体滚动 */
  }
}
  </style>


