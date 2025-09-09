<template>
  <div class="layout">
    <!-- 顶部栏（横屏与PC显示；竖屏移动端隐藏以给内容让路） -->
    <div class="layout-header" v-if="!isPortraitMobile">
      <div class="brand-header">
        <span class="brand-logo fc">FC</span>
        <span class="brand-title">浮城</span>
      </div>
      <div>
        <el-button size="mini" @click="doLogout">退出</el-button>
      </div>
    </div>

    <!-- 主体：
         - 非竖屏移动端：左侧菜单 + 右侧内容（保持现状）
         - 竖屏移动端：仅内容区，上方满屏；底部固定导航栏
    -->
    <div class="layout-body" v-if="!isPortraitMobile">
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
          <el-menu-item index="/backtest">回测管理</el-menu-item>
        </el-menu>
      </div>
      <div class="layout-content" :class="{ 'is-selection': isSelection }">
        <router-view></router-view>
      </div>
    </div>

    <!-- 竖屏移动端布局：上方全屏内容 + 底部导航 -->
    <div class="layout-mobile-portrait" v-else>
      <div class="layout-mobile-content" :class="{ 'is-selection': isSelection }">
        <router-view></router-view>
      </div>
      <div class="layout-bottom-nav">
        <div class="bottom-nav">
          <div class="bottom-item" :class="{active: activeBottom === '/selection'}" @click="$router.push('/selection')">选股管理</div>

          <el-popover placement="top" width="220" v-model="showDailyPop" popper-class="bottom-daily-pop" :append-to-body="true" trigger="manual">
            <div class="daily-pop">
              <div class="daily-item" @click="navigate('/daily/csi')">沪深300</div>
              <div class="daily-item" @click="navigate('/daily/stock')">单股K线</div>
              <div class="daily-item" @click="navigate('/daily/backfill')">日线补拉</div>
            </div>
            <div slot="reference" class="bottom-item" :class="{active: activeBottom.indexOf('/daily')===0}" @click="toggleDaily">
              <span>日线管理</span>
            </div>
          </el-popover>

          <div class="bottom-item" :class="{active: activeBottom === '/factors'}" @click="$router.push('/factors')">因子管理</div>
          <div class="bottom-item" :class="{active: activeBottom === '/backtest'}" @click="$router.push('/backtest')">回测管理</div>
          <div class="bottom-item" :class="{active: activeBottom === '/system'}" @click="$router.push('/system')">系统管理</div>
        </div>
      </div>
    </div>
  </div>
  </template>

<script>
export default {
  name: 'Layout',
  data(){
    return { isMobile: false, isPortrait: true, showDailyPop: false }
  },
  computed: {
    isPortraitMobile(){ return this.isMobile && this.isPortrait },
    isSelection(){ const p=this.$route && this.$route.path ? this.$route.path : ''; return p==='/selection' },
    activeBottom(){
      // 底部菜单的高亮：/daily/* 统一高亮对应子项
      const p = this.$route && this.$route.path ? this.$route.path : '/selection'
      if(p.startsWith('/daily/')){
        if(p.startsWith('/daily/stock')) return '/daily/stock'
        return '/daily/csi'
      }
      return p
    }
  },
  created(){ this.onResize(); window.addEventListener('resize', this.onResize); window.addEventListener('orientationchange', this.onResize) },
  beforeDestroy(){ window.removeEventListener('resize', this.onResize); window.removeEventListener('orientationchange', this.onResize) },
  methods: {
    doLogout(){ localStorage.removeItem('authedUser'); this.$router.replace('/login') },
    onResize(){
      try{
        const w = window.innerWidth || 1024
        const h = window.innerHeight || 768
        this.isMobile = w <= 768
        this.isPortrait = h >= w
      }catch(e){ this.isMobile=false; this.isPortrait=true }
    },
    toggleDaily(){ this.showDailyPop = !this.showDailyPop },
    navigate(path){ this.showDailyPop=false; this.$router.push(path) }
  }
}
  </script>
<style scoped>
.layout{ height:100vh; display:flex; flex-direction:column }
.layout-header{ height:48px; display:flex; align-items:center; justify-content:space-between; padding:0 12px; border-bottom:1px solid #e6ebf2; background:#f7f9fe }
.brand-header{ display:flex; align-items:center; gap:10px }
.brand-logo{ display:inline-flex; align-items:center; justify-content:center; width:28px; height:28px; border-radius:8px; color:#1F2937; font-size:12px; font-weight:800; letter-spacing:1px; background:rgba(255,255,255,0.85); border:1px solid rgba(17,24,39,0.08) }
.brand-title{ font-size:18px; color:#1F2937; font-weight:800; letter-spacing:0.3em }
.layout-body{ flex:1; display:flex; min-height:0 }
.layout-sider{ width:200px; border-right:1px solid #e6ebf2; background:#fbfdff }
.layout-content{ flex:1; padding:10px; overflow:auto }
.layout-content.is-selection{ overflow:hidden }

/* 竖屏移动端 */
.layout-mobile-portrait{ flex:1; display:flex; flex-direction:column; min-height:0 }
.layout-mobile-content{ flex:1; overflow:auto; padding:0 8px 56px 8px; position: relative; z-index: 1 }
.layout-mobile-content.is-selection{ overflow:hidden }
.layout-bottom-nav{ position:fixed; left:0; right:0; bottom:0; height:52px; border-top:1px solid #e6ebf2; background:#fff; z-index: 4000; }
.bottom-nav{ height:100%; display:flex; align-items:center; justify-content:space-around; padding:0 8px }
.bottom-item{ flex:1; text-align:center; color:#374151; padding:10px 4px; font-size:13px }
.bottom-item.active{ color:#2563eb; font-weight:600 }
/* 去除箭头占位 */
.bottom-item .arrow{ display:none }

/* 弹出层样式，模拟公众号底部二级菜单 */
.bottom-daily-pop{ padding:6px 0; z-index: 5000; }
/* 隐藏弹出层尖角，使按钮文字保持居中视觉 */
.bottom-daily-pop[x-placement^="top"] .popper__arrow { display:none }
.bottom-daily-pop .popper__arrow { display:none }
.daily-pop{ display:flex; flex-direction:column }
.daily-item{ padding:8px 12px; font-size:13px; color:#374151; }
.daily-item:hover{ background:#f3f6ff; color:#1d4ed8; cursor:pointer }
</style>

