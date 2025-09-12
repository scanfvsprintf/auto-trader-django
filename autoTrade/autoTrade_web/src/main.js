import Vue from 'vue'
import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'
import router from './router'
import App from './App.vue'
import './styles/theme.css'
import './styles/mobile-optimization.css'
import viewportManager from './utils/viewportManager'

// 统一全局组件尺寸为 mini，更紧凑
Vue.use(ElementUI, { size: 'mini' })

// 全局注册视口管理器
Vue.prototype.$viewportManager = viewportManager

new Vue({
  router,
  render: h => h(App)
}).$mount('#app')


