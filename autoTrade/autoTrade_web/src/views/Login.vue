<template>
  <div class="login-root">
    <!-- 星空背景（极其缓慢、安静） -->
    <canvas ref="canvas" class="star-canvas"></canvas>

    <!-- 居中容器 -->
    <div class="login-center">
      <div class="login-card">
        <div class="brand">
          <div class="brand-logo fc">FC</div>
          <div class="brand-title">浮城</div>
        </div>

        <el-form label-position="top" class="login-form">
          <el-form-item label="用户名">
            <el-input v-model="username" placeholder="请输入用户名" size="small"></el-input>
          </el-form-item>
          <el-form-item label="密码">
            <el-input v-model="password" type="password" placeholder="请输入密码" size="small"></el-input>
          </el-form-item>
          <div class="login-actions">
            <el-button type="default" @click="login" size="small" class="btn-primary">登 录</el-button>
          </div>
        </el-form>
      </div>
    </div>
  </div>
</template>

<script>
import smartViewportManager from '@/utils/smartViewportManager'

export default {
  name: 'Login',
  data(){
    return {
      username: '',
      password: ''
    }
  },
  mounted(){
    // 绑定 this 到 step，避免 rAF 回调丢失上下文
    this.step = this.step.bind(this)
    this.setupCanvas()
    window.addEventListener('resize', this.handleResize)
    // 使用视口管理器确保登录页面稳定
    this.updateViewportHeight()
  },
  beforeDestroy(){
    cancelAnimationFrame(this._raf)
    window.removeEventListener('resize', this.handleResize)
  },
  methods: {
    login(){
      if ((this.username === 'xiangyangxin' || this.username === 'xiangmei') && this.password === '123456') {
        localStorage.setItem('authedUser', this.username)
        this.$message.success('登录成功')
        this.$router.replace('/')
      } else {
        this.$message.error('用户名或密码错误')
      }
    },
    /* 画布初始化与动画 */
    setupCanvas(){
      const c = this.$refs.canvas
      if (!c) return
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      c.width = window.innerWidth * dpr
      c.height = window.innerHeight * dpr
      c.style.width = window.innerWidth + 'px'
      c.style.height = window.innerHeight + 'px'
      this._ctx = c.getContext('2d')
      this._ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      this.initParticles()
      this.initLights()
      this._t = 0
      this.step()
    },
    initParticles(){
      // 呼吸尘埃：城堡轮廓内缓慢漂移 + 呼吸明暗（发光微粒）
      const count = 1300
      const sizeMin = 1.0
      const sizeMax = 3.2
      const w = window.innerWidth
      const h = window.innerHeight
      this._particles = Array.from({ length: count }).map(() => {
        const pt = this.sampleCastlePoint(w, h)
        return {
          x: pt.x,
          y: pt.y,
          size: sizeMin + Math.random() * (sizeMax - sizeMin),
          baseAlpha: 0.20 + Math.random() * 0.20, // 0.20 - 0.40 更柔和，配合glow
          phase: Math.random() * Math.PI * 2,
          period: 18 + Math.random() * 12, // 18-30s 更舒缓
          seed: Math.random() * 1000
        }
      })
    },
    // 归一化坐标下的城堡轮廓（扩大占屏比例）
    isInCastle(nx, ny){
      // 基座（更宽、更低）
      const inBase = nx >= 0.08 && nx <= 0.92 && ny >= 0.72 && ny <= 0.92
      // 中央主堡（更大）
      const inKeep = nx >= 0.34 && nx <= 0.66 && ny >= 0.38 && ny <= 0.78
      // 左右塔楼（更高）
      const inLeftTower = nx >= 0.18 && nx <= 0.26 && ny >= 0.40 && ny <= 0.86
      const inRightTower = nx >= 0.74 && nx <= 0.82 && ny >= 0.40 && ny <= 0.86
      // 女儿墙（长条分段）
      const battlementY = ny >= 0.60 && ny <= 0.68
      const battlementX = (nx >= 0.10 && nx <= 0.16) || (nx >= 0.18 && nx <= 0.24) || (nx >= 0.26 && nx <= 0.32) || (nx >= 0.34 && nx <= 0.40) || (nx >= 0.60 && nx <= 0.66) || (nx >= 0.68 && nx <= 0.74) || (nx >= 0.76 && nx <= 0.82) || (nx >= 0.84 && nx <= 0.90)
      const inBattlements = battlementY && battlementX
      // 镂空城门（负形，随整体变大）
      const inGate = nx >= 0.47 && nx <= 0.53 && ny >= 0.78 && ny <= 0.92
      const positive = inBase || inKeep || inLeftTower || inRightTower || inBattlements
      return positive && !inGate
    },
    // 在城堡轮廓内随机采样一个点（扩大采样范围，几乎全屏）
    sampleCastlePoint(w, h){
      let tries = 0
      while (tries++ < 6000){
        const nx = 0.04 + Math.random() * 0.92
        const ny = 0.18 + Math.random() * 0.74
        if (this.isInCastle(nx, ny)) return { x: nx * w, y: ny * h }
      }
      return { x: w * 0.5, y: h * 0.6 }
    },
    // 低频流场（近似水墨感的缓慢流动）
    flowVector(nx, ny, t){
      const freq1 = 2.8, freq2 = 3.6
      const t1 = t * 0.05, t2 = t * 0.038
      const angle = Math.sin(nx * freq1 + t1) * 0.8 + Math.cos(ny * freq2 - t2) * 0.6
      const swirl = Math.sin((nx + ny + t * 0.02) * 2.2) * 0.25
      const speed = 0.10 // px/帧
      const vx = (Math.cos(angle) + swirl) * speed
      const vy = (Math.sin(angle) - swirl) * speed
      return { vx, vy }
    },
    initLights(){
      const w = window.innerWidth
      const h = window.innerHeight
      const r = Math.max(w, h)
      // 极其柔和的"流光"，使用巨型径向渐变做缓慢位移
      this._lights = [
        { x: w * 0.25, y: h * 0.2,  r: r * 0.7, dx: 0.04,  dy: 0.00, a: 0.13 },
        { x: w * 0.75, y: h * 0.85, r: r * 0.8, dx: -0.03, dy: -0.02, a: 0.11 },
        { x: w * 0.55, y: h * 0.45, r: r * 0.9, dx: 0.01,  dy: 0.015, a: 0.10 }
      ]
    },
    step(){
      if (!this._ctx) return
      const ctx = this._ctx
      const w = window.innerWidth
      const h = window.innerHeight
      // 浅色基底：线性渐变
      ctx.globalCompositeOperation = 'source-over'
      const bg = ctx.createLinearGradient(0, 0, w, h)
      bg.addColorStop(0, '#F6FAFF')
      bg.addColorStop(1, '#EEF3FF')
      ctx.fillStyle = bg
      ctx.fillRect(0, 0, w, h)

      // 重新启用流光与对角光带、暗角（轻度增强）
      ctx.globalCompositeOperation = 'soft-light'
      for (const L of this._lights || []){
        const grad = ctx.createRadialGradient(L.x, L.y, L.r * 0.05, L.x, L.y, L.r)
        grad.addColorStop(0, `rgba(255,255,255,${L.a})`)
        grad.addColorStop(1, 'rgba(255,255,255,0)')
        ctx.fillStyle = grad
        ctx.fillRect(0, 0, w, h)
        // 缓慢位移，超低速度
        L.x += L.dx
        L.y += L.dy
        if (L.x < -L.r) L.x = w + L.r
        if (L.x > w + L.r) L.x = -L.r
        if (L.y < -L.r) L.y = h + L.r
        if (L.y > h + L.r) L.y = -L.r
      }
      // 对角光带
      ctx.globalCompositeOperation = 'screen'
      const diag = ctx.createLinearGradient(0, h * 0.2, w, h * 0.8)
      diag.addColorStop(0, 'rgba(255,255,255,0.00)')
      diag.addColorStop(0.5, 'rgba(255,255,255,0.06)')
      diag.addColorStop(1, 'rgba(255,255,255,0.00)')
      ctx.fillStyle = diag
      ctx.fillRect(0, 0, w, h)
      // 极淡暗角
      ctx.globalCompositeOperation = 'multiply'
      const vg = ctx.createRadialGradient(w * 0.5, h * 0.5, Math.min(w, h) * 0.6, w * 0.5, h * 0.5, Math.max(w, h))
      vg.addColorStop(0, 'rgba(255,255,255,0)')
      vg.addColorStop(1, 'rgba(221,230,245,0.5)')
      ctx.fillStyle = vg
      ctx.fillRect(0, 0, w, h)

      // 呼吸尘埃（城堡轮廓内：缓慢流动 + 呼吸 + 发光）
      ctx.globalCompositeOperation = 'screen' // 叠加更柔和的光感
      ctx.save()
      this._t = (this._t || 0) + 0.016 // ~60FPS 估算
      for (const p of this._particles || []){
        const nx = p.x / w
        const ny = p.y / h
        const { vx, vy } = this.flowVector(nx, ny, this._t + p.seed)
        let nx2 = (p.x + vx) / w
        let ny2 = (p.y + vy) / h
        if (!this.isInCastle(nx2, ny2)){
          const pt = this.sampleCastlePoint(w, h)
          p.x = pt.x
          p.y = pt.y
        } else {
          p.x += vx
          p.y += vy
        }
        const breath = 0.55 + 0.45 * Math.sin(this._t / p.period + p.phase)
        const alpha = p.baseAlpha * breath
        const glowR = p.size * 4.5
        const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, glowR)
        grad.addColorStop(0, `rgba(148,163,184,${alpha})`) // #94A3B8
        grad.addColorStop(1, 'rgba(148,163,184,0)')
        ctx.fillStyle = grad
        ctx.beginPath()
        ctx.arc(p.x, p.y, glowR, 0, Math.PI * 2)
        ctx.fill()
      }
      ctx.restore()
      this._raf = requestAnimationFrame(this.step)
    },
    handleResize(){
      // 重新设置画布与粒子，保持清晰与宁静
      cancelAnimationFrame(this._raf)
      this._ctx = null
      this.setupCanvas()
    },
    updateViewportHeight() {
      // 更新登录页面的视口高度
      const viewportInfo = smartViewportManager.getViewportInfo()
      document.documentElement.style.setProperty('--login-viewport-height', `${viewportInfo.currentHeight}px`)
    }
  }
}
</script>
<style scoped>
.login-root{ 
  position:relative; 
  height:100vh; 
  height: var(--login-viewport-height, 100vh); /* 使用动态视口高度 */
  overflow:hidden; 
}
.star-canvas{ position:absolute; inset:0; display:block }
.login-center{ position:relative; z-index:1; height:100%; display:flex; align-items:center; justify-content:center; padding:16px }
.login-card{ width:480px; max-width:100%; border-radius:16px; padding:20px 20px 16px; border:1px solid rgba(17,24,39,0.08); background: rgba(255,255,255,0.50); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); box-shadow:0 20px 60px rgba(17,24,39,0.06) }
.brand{ display:flex; align-items:center; justify-content:center; gap:10px; margin-bottom:16px }
.brand-logo{ width:44px; height:44px; border-radius:10px; display:flex; align-items:center; justify-content:center; color:#1F2937; font-weight:800; font-size:18px; letter-spacing:1px; background:rgba(255,255,255,0.85); border:1px solid rgba(17,24,39,0.08) }
.brand-logo.fc{ font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif }
.brand-title{ font-size:22px; color:#1F2937; font-weight:800; letter-spacing:0.35em }
.login-form .el-form-item{ margin-bottom:12px }
.login-actions{ margin-top:8px }
.btn-primary{ width:100%; background:rgba(255,255,255,0.9); color:#334155; border-color:rgba(0,0,0,0.06) }
.btn-primary:hover{ background:#ffffff; color:#1F2937; border-color:rgba(0,0,0,0.10) }

/* 浅色输入样式（淡雅） */
.login-card :deep(.el-form-item__label){ color:#475569 }
.login-card :deep(.el-input__inner){ background:rgba(255,255,255,0.70); color:#111827; border:none; border-bottom:1px solid rgba(0,0,0,0.06); border-radius:6px; height:32px; line-height:32px; box-shadow:none; transition:all .3s ease }
.login-card :deep(.el-input__inner::placeholder){ color:#94A3B8 }
.login-card :deep(.el-input.is-active .el-input__inner),
.login-card :deep(.el-input__inner:focus){ background:rgba(255,255,255,0.85); border-bottom-color:#3B82F6; box-shadow:0 0 0 2px rgba(59,130,246,0.10) inset }

/* 响应式适配 */
@media (max-width: 480px){
  .login-card{ width:92% }
  .brand-title{ letter-spacing:0.25em }
}
</style>
