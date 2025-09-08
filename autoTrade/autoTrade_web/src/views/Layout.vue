<template>
  <div class="layout">
    <div class="layout-header">
      <div>autoTrade 管理端</div>
      <div>
        <el-button size="mini" @click="doLogout">退出</el-button>
      </div>
    </div>
    <div class="layout-body">
      <div class="layout-sider">
        <el-menu :default-active="$route.path" router>
          <el-menu-item index="/selection">选股管理</el-menu-item>
          <el-submenu index="/daily">
            <template slot="title">日线管理</template>
            <el-menu-item index="/daily/csi">沪深300</el-menu-item>
            <el-menu-item index="/daily/stock">单股K线</el-menu-item>
            <el-menu-item index="/daily/backfill">日线补拉</el-menu-item>
          </el-submenu>
          <el-menu-item index="/factors">因子管理</el-menu-item>
          <el-menu-item index="/system">系统管理</el-menu-item>
        </el-menu>
      </div>
      <div class="layout-content">
        <router-view></router-view>
      </div>
    </div>
  </div>
  </template>

<script>
export default {
  name: 'Layout',
  methods: {
    doLogout(){ localStorage.removeItem('authedUser'); this.$router.replace('/login') }
  }
}
  </script>


