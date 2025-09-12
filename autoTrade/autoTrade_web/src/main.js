import Vue from 'vue'
import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'
import router from './router'
import App from './App.vue'
import './styles/theme.css'
import './styles/smart-mobile-optimization.css'
import smartViewportManager from './utils/smartViewportManager'

// 统一全局组件尺寸为 mini，更紧凑
Vue.use(ElementUI, { size: 'mini' })

// 全局注册智能视口管理器
Vue.prototype.$smartViewportManager = smartViewportManager

new Vue({
  router,
  render: h => h(App)
}).$mount('#app')


