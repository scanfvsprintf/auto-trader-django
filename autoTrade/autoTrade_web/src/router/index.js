import Vue from 'vue'
import Router from 'vue-router'
import Login from '../views/Login.vue'
import Layout from '../views/Layout.vue'
import Selection from '../views/Selection.vue'
import Daily from '../views/Daily.vue'
import DailyCsi from '../views/DailyCsi.vue'
import DailyStock from '../views/DailyStock.vue'
import DailyETF from '../views/DailyETF.vue'
import DailyBackfill from '../views/DailyBackfill.vue'
import Factors from '../views/Factors.vue'
import System from '../views/System.vue'
import Schema from '../views/Schema.vue'
import SystemBacktest from '../views/SystemBacktest.vue'
import AiConfig from '../views/AiConfig.vue'

Vue.use(Router)

function isAuthed() { return !!localStorage.getItem('authedUser') }

const router = new Router({
  mode: 'hash',
  routes: [
    { path: '/login', component: Login },
    { path: '/', component: Layout, children: [
      { path: '', redirect: '/selection' },
      { path: 'selection', component: Selection },
      { path: 'daily', component: Daily, redirect: '/daily/csi', children: [
        { path: 'csi', component: DailyCsi },
        { path: 'stock', component: DailyStock },
        { path: 'etf', component: DailyETF },
        { path: 'backfill', component: DailyBackfill }
      ]},
      { path: 'factors', component: Factors },
      { path: 'system', component: System },
      { path: 'system/schema', component: Schema },
      { path: 'backtest', component: SystemBacktest },
      { path: 'ai-config', component: AiConfig }
    ]}
  ]
})

router.beforeEach((to, from, next) => {
  if (to.path !== '/login' && !isAuthed()) { next('/login'); return }
  // 简单基于用户名的只读权限示例：xiangmei 视为只读
  if (to.path === '/selection' || to.path === '/factors' || to.path === '/system' || to.path.startsWith('/system/') || to.path === '/backtest' || to.path === '/ai-config') {
    const u = localStorage.getItem('authedUser') || ''
    // 在页面内部据此控制按钮的显示与禁用
    window.__READ_ONLY__ = (u === 'xiangmei')
  } else {
    window.__READ_ONLY__ = false
  }
  next()
})

export default router


