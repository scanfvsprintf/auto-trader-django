import Vue from 'vue'
import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'
import router from './router'
import App from './App.vue'
import './styles/theme.css'

// 统一全局组件尺寸为 mini，更紧凑
Vue.use(ElementUI, { size: 'mini' })

new Vue({
  router,
  render: h => h(App)
}).$mount('#app')


