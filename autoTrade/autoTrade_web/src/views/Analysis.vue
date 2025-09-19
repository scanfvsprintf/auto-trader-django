<template>
  <div style="height:100%">
    <el-tabs v-model="activeTab" style="height:100%" @tab-click="handleTabClick">
      <el-tab-pane label="相关性分析" name="correlation">
        <router-view></router-view>
      </el-tab-pane>
      <!-- 预留更多分析功能的选项卡 -->
      <!-- <el-tab-pane label="风险分析" name="risk">
        <div style="padding:20px;text-align:center;color:#999">
          风险分析功能开发中...
        </div>
      </el-tab-pane> -->
    </el-tabs>
  </div>
</template>

<script>
export default {
  name: 'Analysis',
  data() {
    return {
      activeTab: 'correlation'
    }
  },
  methods: {
    handleTabClick(tab) {
      // 根据选项卡切换路由
      const tabRoutes = {
        'correlation': '/analysis/correlation'
        // 'risk': '/analysis/risk'
      }
      const route = tabRoutes[tab.name]
      if (route && this.$route.path !== route) {
        this.$router.push(route)
      }
    }
  },
  created() {
    // 根据当前路由设置活动选项卡
    const routeTabMap = {
      '/analysis/correlation': 'correlation'
      // '/analysis/risk': 'risk'
    }
    this.activeTab = routeTabMap[this.$route.path] || 'correlation'
  },
  watch: {
    '$route'(to) {
      // 路由变化时更新选项卡
      const routeTabMap = {
        '/analysis/correlation': 'correlation'
        // '/analysis/risk': 'risk'
      }
      this.activeTab = routeTabMap[to.path] || 'correlation'
    }
  }
}
</script>

<style scoped>
/* 确保选项卡占满高度 */
.el-tabs {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.el-tabs__content {
  flex: 1;
  overflow: hidden;
}

.el-tab-pane {
  height: 100%;
}
</style>
