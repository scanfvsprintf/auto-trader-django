<template>
  <div class="selection-root" v-loading="loading" element-loading-text="加载中...">
    <div class="selection-toolbar-static">
      <el-form inline class="selection-toolbar-form">
        <el-form-item label="日期">
          <el-date-picker v-model="queryDate" type="date" value-format="yyyy-MM-dd" placeholder="选择日期"></el-date-picker>
        </el-form-item>
        <el-form-item class="toolbar-actions">
          <div class="toolbar-actions-row">
            <el-button type="primary" plain size="mini" :loading="loading" @click="fetchPlans">查询</el-button>
            <el-button type="primary" plain size="mini" @click="showRunDrawer=true">手动发起选股</el-button>
          </div>
        </el-form-item>
      </el-form>
    </div>

    <div class="selection-content scroll-area">
    <!-- 非竖屏移动端：表格呈现 -->
    <el-table v-if="!isPortraitMobile" :data="rows" border style="width:100%" @row-click="showFactors">
      <el-table-column prop="rank" label="排名" width="80" />
      <el-table-column prop="stock_code" label="股票代码" width="140" />
      <el-table-column prop="stock_name" label="股票名称" width="140" />
      <el-table-column prop="miop" label="低开" />
      <el-table-column prop="maop" label="高开" />
      <el-table-column prop="final_score" label="得分" />
      <el-table-column label="操作" width="120">
        <template slot-scope="scope">
          <el-button type="primary" size="mini" @click.stop="goKline(scope.row)">查看K线</el-button>
        </template>
      </el-table-column>
    </el-table>

    <!-- 竖屏移动端：卡片列表，避免列拥挤与底部导航重叠 -->
    <div v-else class="mobile-card-list">
      <el-card v-for="row in rows" :key="row.stock_code + '_' + row.rank" class="mobile-card" @click.native="showFactors(row)">
        <div class="mobile-card-top">
          <div class="mobile-card-title">{{ row.stock_name }} <span class="mobile-card-code">{{ row.stock_code }}</span></div>
          <el-button type="primary" size="mini" @click.stop="goKline(row)">查看K线</el-button>
        </div>
        <div class="mobile-card-meta">
          <span>排名：{{ row.rank }}</span>
          <span>得分：{{ fmt(row.final_score) }}</span>
        </div>
        <div class="mobile-card-meta">
          <span>低开：{{ fmt(row.miop) }}</span>
          <span>高开：{{ fmt(row.maop) }}</span>
        </div>
      </el-card>
    </div>
    </div>

    <el-drawer :visible.sync="factorVisible" :title="'因子值 · ' + (queryDate||'')" :size="drawerSize" :append-to-body="true" custom-class="selection-drawer">
      <div class="factor-drawer" v-loading="loadingFactors" element-loading-text="加载因子中...">
        <div class="factor-drawer-header">
          <div class="factor-stock">{{ factorTitle }}</div>
          <div class="factor-sub">基于 {{ queryDate || '-' }} 的策略因子</div>
        </div>
        <el-table :data="factors" size="mini" border height="360">
          <el-table-column prop="factor_code" label="因子代码" width="180" />
          <el-table-column prop="factor_name" label="因子名称" />
          <el-table-column prop="raw_value" label="原始值" width="120" />
          <el-table-column prop="norm_score" label="标准分" width="120" />
        </el-table>
      </div>
    </el-drawer>

    <!-- 手动发起选股：抽屉样式（与系统/日线一致的美化风格） -->
    <el-drawer :visible.sync="showRunDrawer" title="手动发起选股" size="90%" :append-to-body="true" :destroy-on-close="true" custom-class="sysbt-drawer">
      <div class="sysbt-drawer-body">
        <div class="sysbt-section">
          <div class="sysbt-sec-title">参数</div>
          <el-form label-position="top" class="sysbt-form">
            <div class="sysbt-grid">
              <el-form-item label="模式">
                <el-radio-group v-model="runMode" size="mini">
                  <el-radio label="single">单日</el-radio>
                  <el-radio label="range">日期区间回补</el-radio>
                </el-radio-group>
              </el-form-item>
              <template v-if="runMode==='single'">
                <el-form-item label="日期" class="sysbt-grid-span">
                  <el-date-picker v-model="runDate" type="date" value-format="yyyy-MM-dd" placeholder="选择日期"></el-date-picker>
                </el-form-item>
                <el-form-item label="是否补推邮件">
                  <el-switch v-model="sendMail" />
                </el-form-item>
              </template>
              <template v-else>
                <el-form-item label="日期区间" class="sysbt-grid-span">
                  <el-date-picker v-model="runRange" type="daterange" range-separator="至" start-placeholder="开始日期" end-placeholder="结束日期" value-format="yyyy-MM-dd" :unlink-panels="true" />
                </el-form-item>
                <div class="sysbt-help sysbt-grid-span">按区间内的交易日逐日回补，只生成评分与次日预案。</div>
              </template>
            </div>
          </el-form>
        </div>
        <div class="sysbt-drawer-actions">
          <el-button size="mini" @click="showRunDrawer=false">取消</el-button>
          <el-button size="mini" type="primary" :loading="runLoading" @click="doRun">确定</el-button>
        </div>
      </div>
    </el-drawer>
  </div>
  </template>

<script>
import axios from 'axios'
import smartViewportManager from '@/utils/smartViewportManager'

export default {
  name: 'Selection',
  data(){
    return {
      queryDate: '',
      rows: [],
      loading: false,
      factorVisible: false,
      factors: [],
      loadingFactors: false,
      factorTitle: '',
      showRunDrawer: false,
      runMode: 'single',
      runDate: '',
      runRange: [],
      sendMail: false,
      runLoading: false,
      isMobile: false,
      isPortrait: true,
      drawerSize: '85%'
    }
  },
  created(){
    // 默认日期设为今日（后端做回溯到最近交易日的兼容）
    const d = new Date(); const t = `${d.getFullYear()}-${('0'+(d.getMonth()+1)).slice(-2)}-${('0'+d.getDate()).slice(-2)}`
    this.queryDate = t
    this.$nextTick(this.fetchPlans)
    this.onResize();
    // 使用视口管理器获取设备信息
    this.updateDeviceInfo();
    // 定期同步设备信息
    this.deviceInfoInterval = setInterval(this.updateDeviceInfo, 1000);
  },
  beforeDestroy(){ 
    if (this.deviceInfoInterval) {
      clearInterval(this.deviceInfoInterval);
    }
  },
  computed:{
    isPortraitMobile(){ return this.isMobile && this.isPortrait }
  },
  methods: {
    fetchPlans(){
      if (!this.queryDate) { this.$message.error('请选择日期'); return }
      this.loading = true
      axios.get('/webManager/selection/plans', { params: { date: this.queryDate } })
        .then(res => {
          if (res.data.code===0){
            const payload = res.data.data
            if (Array.isArray(payload)){
              this.rows = payload || []
            } else {
              this.rows = (payload && payload.rows) || []
              if(payload && payload.date){ this.queryDate = payload.date }
            }
          } else {
            this.$message.error(res.data.msg)
          }
        })
        .catch(()=> this.$message.error('查询失败'))
        .finally(()=> this.loading = false)
    },
    showFactors(row){
      if (!this.queryDate) return
      this.loadingFactors = true
      axios.get('/webManager/selection/factors', { params: { date: this.queryDate, stock: row.stock_code } })
        .then(res => {
          if (res.data.code===0) { this.factors = res.data.data || []; this.factorTitle = row.stock_code + ' ' + row.stock_name; this.factorVisible=true }
          else this.$message.error(res.data.msg)
        })
        .catch(()=> this.$message.error('查询因子失败'))
        .finally(()=> this.loadingFactors = false)
    },
    openRun(){ this.showRunDrawer = true },
    doRun(){
      if (this.runMode==='single'){
        if (!this.runDate) { this.$message.error('请选择日期'); return }
        this.runLoading = true
        axios.post('/webManager/selection/run', { date: this.runDate, send_mail: this.sendMail })
          .then(res => { if (res.data.code===0) this.$message.success('发起成功'); else this.$message.error(res.data.msg) })
          .catch(()=> this.$message.error('发起失败'))
          .finally(()=> { this.runLoading=false; this.showRunDrawer=false })
      } else {
        if (!this.runRange || this.runRange.length!==2){ this.$message.error('请选择日期区间'); return }
        const [start, end] = this.runRange
        this.runLoading = true
        axios.post('/webManager/selection/run_range', { start, end })
          .then(res => {
            if (res.data.code===0){
              const days = (res.data.data && res.data.data.days) || 0
              this.$message.success('回补完成：' + days + ' 个交易日')
            } else {
              this.$message.error(res.data.msg)
            }
          })
          .catch(()=> this.$message.error('回补失败'))
          .finally(()=> { this.runLoading=false; this.showRunDrawer=false })
      }
    },
    goKline(row){
      if(!row || !row.stock_code){ this.$message.error('无效股票'); return }
      this.$router.push({ path: '/daily/stock', query: { code: row.stock_code, name: row.stock_name||'', from: 'selection' } })
    },
    fmt(v){
      if(v===null || v===undefined || v==='') return '-'
      const n=Number(v)
      if(isNaN(n)) return String(v)
      return n.toFixed(2)
    },
    onResize(){
      try{
        const w = window.innerWidth || 1024
        const h = window.innerHeight || 768
        this.isMobile = w <= 768
        
        // 修复：避免键盘弹起时的横屏误判
        if (!this._lastHeight || Math.abs(h - this._lastHeight) < 100) {
          this.isPortrait = h >= w
        }
        this._lastHeight = h
        
        // 因子抽屉在竖屏使用更窄占比，横屏/PC 使用 40%
        this.drawerSize = this.isPortraitMobile ? '90%' : '40%'
      }catch(e){ this.isMobile=false; this.isPortrait=true }
    },
    updateDeviceInfo(){
      // 从视口管理器获取最新的设备信息
      const viewportInfo = smartViewportManager.getViewportInfo();
      this.isMobile = viewportInfo.isMobile;
      this.isPortrait = viewportInfo.isPortrait;
      // 更新抽屉大小
      this.drawerSize = this.isPortraitMobile ? '90%' : '40%'
    }
  }
}
  </script>

<style scoped>
.selection-root{ 
  display:flex; 
  flex-direction:column; 
  height:100%; 
  background:#ffffff;
  /* 使用CSS变量确保视口稳定性 */
  height: var(--viewport-height, 100vh);
  max-height: var(--viewport-height, 100vh);
}
.selection-toolbar-static{ 
  flex:0 0 auto; 
  background:#ffffff; 
  padding:12px 12px; 
  margin:0 -8px; 
  border-bottom:1px solid #e6ebf2; 
  overflow-x:auto; 
  -webkit-overflow-scrolling:touch;
  /* 防止键盘弹起时的布局跳动 */
  transform: translateZ(0);
}
.selection-toolbar-form{ display:flex; flex-wrap:nowrap; align-items:center; gap:8px; width:max-content }
.selection-toolbar-form .el-form-item{ margin-right:0; margin-bottom:0; flex:0 0 auto }
.selection-toolbar-form .toolbar-actions .toolbar-actions-row{ display:flex; flex-wrap:nowrap; gap:8px }
.selection-toolbar-form .el-input__inner, .selection-toolbar-form .el-date-editor{ height:28px; line-height:28px }
.selection-toolbar-form .el-button{ height:28px; padding: 0 10px }
.selection-content{ 
  flex:1 1 auto; 
  position: relative; 
  z-index: 1; 
  padding:12px 8px 8px;
  /* 移动端滚动优化 */
  -webkit-overflow-scrolling: touch;
  /* 防止键盘弹起时的布局跳动 */
  transform: translateZ(0);
}
.scroll-area{ 
  height:100%; 
  overflow:auto;
  /* 移动端滚动优化 */
  -webkit-overflow-scrolling: touch;
}
.factor-drawer{ padding:12px }
.factor-drawer-header{ margin-bottom:8px }
.factor-stock{ font-weight:600; color:#1f2937 }
.factor-sub{ color:#6b7280; font-size:12px; margin-top:2px }
/* 提升抽屉层级，避免被底部导航覆盖 */
::v-deep .el-drawer__wrapper.selection-drawer{ z-index: 6000 !important; }
.factor-drawer{ padding:12px }
.factor-drawer-header{ margin-bottom:8px }
.factor-stock{ font-weight:600; color:#1f2937 }
.factor-sub{ color:#6b7280; font-size:12px; margin-top:2px }
.mobile-card-list{ display:flex; flex-direction:column; gap:8px; padding-bottom:64px }
.mobile-card{ border:1px solid #e6ebf2 }
.mobile-card-top{ display:flex; align-items:center; justify-content:space-between; margin-bottom:4px }
.mobile-card-title{ font-weight:600; color:#374151 }
.mobile-card-code{ color:#6b7280; font-weight:400; margin-left:6px; font-size:12px }
.mobile-card-meta{ display:flex; gap:14px; color:#4b5563; font-size:12px; margin-top:2px }

/* 复用系统抽屉美化样式 */
.sysbt-drawer >>> .el-drawer__body{ padding:0; background:#f9fbff }
.sysbt-drawer-body{ padding:12px }
.sysbt-section{ background:#fff; border:1px solid #e6ebf2; border-radius:6px; padding:10px 12px; box-shadow:0 1px 2px rgba(0,0,0,0.03) }
.sysbt-sec-title{ font-size:13px; color:#374151; font-weight:600; margin-bottom:8px }
.sysbt-grid{ display:grid; grid-template-columns:1fr 1fr; gap:8px 12px }
.sysbt-grid-span{ grid-column:1 / span 2 }
.sysbt-help{ color:#6b7280; font-size:12px }
.sysbt-drawer-actions{ display:flex; gap:8px; justify-content:flex-end; margin-top:10px }
@media (max-width: 480px){
  .selection-toolbar-static{ padding:10px 8px }
  .selection-toolbar-form .el-date-editor{ width: 152px }
}
</style>


