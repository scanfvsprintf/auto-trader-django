<template>
  <div v-loading="loading" element-loading-text="加载中..." style="height:100%">
    <!-- 移动端：按钮并入标题栏右侧 -->

    <el-row :gutter="12" style="height:100%">
      <el-col v-if="!isMobile" :xs="24" :sm="7" :md="6" style="height:100%">
        <el-card style="height:100%;display:flex;flex-direction:column">
          <div slot="header">股票列表</div>
          <div style="padding-bottom:8px;display:flex">
            <el-input v-model="keyword" placeholder="代码/名称" size="small" @keyup.enter.native="searchList" style="flex:1;margin-right:6px" />
            <el-button size="small" type="primary" @click="searchList">搜索</el-button>
          </div>
          <el-table
            :data="list"
            size="mini"
            highlight-current-row
            @row-click="handleSelect"
            :height="tableHeight"
          >
            <el-table-column prop="stock_code" label="代码" width="120" />
            <el-table-column prop="stock_name" label="名称" />
          </el-table>
        </el-card>
      </el-col>
      <el-col :xs="24" :sm="17" :md="18" style="height:100%">
        <el-card style="height:100%;display:flex;flex-direction:column">
          <div slot="header" style="display:flex;align-items:center;justify-content:space-between">
            <div>
              {{ currentName || '单股K线' }} <span v-if="code" style="color:#6b7280;font-weight:normal">（{{code}}）</span>
            </div>
            <div style="display:flex; gap:8px; align-items:center">
              <template v-if="isMobile">
                <el-button size="mini" type="primary" plain class="toolbar-btn" @click="showDrawer=true">
                  <i class="el-icon-menu" style="margin-right:4px"></i> 股票
                </el-button>
                <el-button size="mini" type="primary" plain class="toolbar-btn" @click="showSettingDrawer=true">
                  <i class="el-icon-setting" style="margin-right:4px"></i> 设置
                </el-button>
              </template>
              <el-button v-else size="mini" v-if="$route.query && $route.query.from==='selection'" @click="$router.back()">返回</el-button>
            </div>
          </div>
          <div v-if="!isMobile" style="padding:8px 0">
            <el-form inline>
              <el-form-item label="时间区间">
                <el-date-picker v-model="stockRange" type="daterange" value-format="yyyy-MM-dd" range-separator="至" start-placeholder="开始日期" end-placeholder="结束日期" :picker-options="pickerOptions" />
              </el-form-item>
              <el-form-item label="显示">
                <el-radio-group v-model="subMode" @change="onSubModeChange">
                  <el-radio-button label="ma">均线</el-radio-button>
                  <el-radio-button label="vol">成交量</el-radio-button>
                  <el-radio-button label="amt">成交额</el-radio-button>
                </el-radio-group>
              </el-form-item>
              <el-form-item v-if="subMode==='ma'" label="均线"><el-input v-model="ma" placeholder="5,10,20" style="width:140px" @change="onMAChange" /></el-form-item>
              <el-form-item label="后复权"><el-switch v-model="useHfq" @change="onHfqChange" /></el-form-item>
              <el-form-item label="评分线"><el-switch v-model="showScore" @change="onToggleScore" /></el-form-item>
              <el-form-item>
                <el-button type="primary" :loading="loading" @click="fetchStock">查询</el-button>
              </el-form-item>
            </el-form>
          </div>
          <div ref="stockChart" :style="chartStyle"></div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 移动端抽屉：股票列表 -->
    <el-drawer :visible.sync="showDrawer" title="股票列表" size="80%" :append-to-body="true" custom-class="stock-drawer" :destroy-on-close="true">
      <div style="padding:0 8px 8px 8px">
        <div style="display:flex;padding-bottom:8px">
          <el-input v-model="keyword" placeholder="代码/名称" size="small" @keyup.enter.native="searchList" style="flex:1;margin-right:6px" />
          <el-button size="small" type="primary" @click="searchList">搜索</el-button>
        </div>
        <el-table :data="list" size="mini" highlight-current-row @row-click="(r)=>{ handleSelect(r); showDrawer=false }" :height="400">
          <el-table-column prop="stock_code" label="代码" width="120" />
          <el-table-column prop="stock_name" label="名称" />
        </el-table>
      </div>
    </el-drawer>

    <!-- 移动端抽屉：设置 -->
    <el-drawer :visible.sync="showSettingDrawer" title="图表设置" size="90%" :append-to-body="true" :destroy-on-close="true" custom-class="sysbt-drawer">
      <div class="sysbt-drawer-body">
        <div class="sysbt-section">
          <div class="sysbt-sec-title">基础参数</div>
          <el-form label-position="top" class="sysbt-form">
            <div class="sysbt-grid">
              <el-form-item label="时间区间" class="sysbt-grid-span">
                <el-date-picker v-model="stockRange" type="daterange" :unlink-panels="true" value-format="yyyy-MM-dd" range-separator="至" start-placeholder="开始日期" end-placeholder="结束日期" :picker-options="pickerOptions" />
              </el-form-item>
              <el-form-item label="显示">
                <el-radio-group v-model="subMode" @change="onSubModeChange">
                  <el-radio-button label="ma">均线</el-radio-button>
                  <el-radio-button label="vol">成交量</el-radio-button>
                  <el-radio-button label="amt">成交额</el-radio-button>
                </el-radio-group>
              </el-form-item>
              <el-form-item v-if="subMode==='ma'" label="均线"><el-input v-model="ma" placeholder="5,10,20" @change="onMAChange" /></el-form-item>
              <el-form-item label="后复权"><el-switch v-model="useHfq" @change="onHfqChange" /></el-form-item>
              <el-form-item label="评分线"><el-switch v-model="showScore" @change="onToggleScore" /></el-form-item>
            </div>
          </el-form>
        </div>
        <div class="sysbt-drawer-actions">
          <el-button size="mini" @click="showSettingDrawer=false">关闭</el-button>
          <el-button size="mini" type="primary" :loading="loading" @click="()=>{ showSettingDrawer=false; fetchStock() }">查询</el-button>
        </div>
      </div>
    </el-drawer>
  </div>
  </template>

<script>
import axios from 'axios'
import * as echarts from 'echarts'
export default {
  name: 'DailyStock',
  data(){ return { code:'', currentName:'', options:[], searching:false, stockRange:[], ma:'5,10,20', subMode:'ma', useHfq:false, showScore:true, stockData:[], pickerOptions:{}, loading:false, keyword:'', list:[], tableHeight:360, isMobile:false, showDrawer:false, showSettingDrawer:false, _chart:null, chartHeight:380 } },
  computed: {
    chartStyle(){
      // 确保容器有明确高度，ECharts 才能渲染
      const h = this.isMobile ? `${this.chartHeight}px` : '420px'
      return { width: '100%', height: h, minHeight: '320px', flex: '1' }
    }
  },
  created(){
    const end = new Date(); const start = new Date(); start.setMonth(start.getMonth()-1)
    const fmt = d => `${d.getFullYear()}-${('0'+(d.getMonth()+1)).slice(-2)}-${('0'+d.getDate()).slice(-2)}`
    this.pickerOptions = {
      shortcuts: [
        { text:'最近一周', onClick: p=>{ const e=new Date(); const s=new Date(); s.setDate(s.getDate()-7); p.$emit('pick',[fmt(s),fmt(e)]) }},
        { text:'最近一月', onClick: p=>{ const e=new Date(); const s=new Date(); s.setMonth(s.getMonth()-1); p.$emit('pick',[fmt(s),fmt(e)]) }},
        { text:'最近半年', onClick: p=>{ const e=new Date(); const s=new Date(); s.setMonth(s.getMonth()-6); p.$emit('pick',[fmt(s),fmt(e)]) }},
        { text:'最近一年', onClick: p=>{ const e=new Date(); const s=new Date(); s.setFullYear(s.getFullYear()-1); p.$emit('pick',[fmt(s),fmt(e)]) }}
      ]
    }
    // 默认查询列表并选中第一只
    this.stockRange = [fmt(start), fmt(end)]
    this.$nextTick(()=>{ 
      this.onResize();
      const q = this.$route && this.$route.query ? this.$route.query : {}
      if(q.code){
        this.code = q.code
        this.currentName = q.name || this.currentName
        this.fetchStock()
        // 同步填充左侧列表，但避免覆盖当前选中
        this.searchList(q.code, true)
      } else {
        this.searchList('sh.')
      }
    })
    window.addEventListener('resize', this.onResize)
  },
  beforeDestroy(){ window.removeEventListener('resize', this.onResize); if(this._chart){ this._chart.dispose(); this._chart=null } },
  methods: {
    remoteSearch(q){ if(!q){ this.options=[]; return } this.searching=true; axios.get('/webManager/stock/search',{ params:{ q } }).then(r=>{ if(r.data.code===0) this.options=r.data.data||[] }).finally(()=> this.searching=false) },
    onResize(){
      try{
        const w = window.innerWidth || 768
        const h = window.innerHeight || 700
        this.isMobile = (w <= 768)
        this.tableHeight = Math.max(240, h - 220)
        const headerH = this.isMobile ? 40 : 0
        const bottomNav = this.isMobile ? 56 : 0
        const paddings = 24
        this.chartHeight = Math.max(320, h - headerH - bottomNav - paddings - 80)
        if(this._chart){ this._chart.resize() }
      }catch(e){ this.tableHeight = 360 }
    },
    searchList(init, noAutoSelect){
      const q = typeof init==='string' ? init : (this.keyword || '')
      if(!q){ this.$message.info('请输入关键词'); return }
      this.loading = true
      axios.get('/webManager/stock/search', { params:{ q } })
        .then(res=>{
          if(res.data.code===0){
            this.list = res.data.data || []
            if(this.list.length){
              if(!noAutoSelect){
                const first = this.list[0]
                this.code = first.stock_code
                this.currentName = first.stock_name || this.currentName
                this.fetchStock()
              }
            }
          } else {
            this.$message.error(res.data.msg)
          }
        })
        .catch(()=> this.$message.error('搜索失败'))
        .finally(()=> this.loading = false)
    },
    handleSelect(row){ if(!row || !row.stock_code) return; this.code = row.stock_code; this.currentName = row.stock_name || this.currentName; this.fetchStock() },
    fetchStock(){
      if(!this.code || !this.stockRange || this.stockRange.length!==2){ this.$message.error('缺少参数'); return }
      const [start, end] = this.stockRange
      this.loading = true
      const withVol = this.subMode==='vol' ? 1 : 0
      const withAmt = this.subMode==='amt' ? 1 : 0
      const maParam = this.subMode==='ma' ? this.ma : ''
      axios.get('/webManager/daily/stock', { params: { code:this.code, start, end, ma:maParam, with_vol:withVol, with_amt:withAmt, hfq:this.useHfq?1:0 } })
        .then(res=>{ if(res.data.code===0){ this.stockData=res.data.data||[]; this.draw() } else this.$message.error(res.data.msg) })
        .catch(()=> this.$message.error('查询失败'))
        .finally(()=> this.loading=false)
    },
    onSubModeChange(){ this.fetchStock() },
    onToggleScore(){ this.draw() },
    onMAChange(){ if(this.subMode==='ma'){ this.fetchStock() } },
    onHfqChange(){ this.fetchStock() },
    draw(){
      const el=this.$refs.stockChart; if(!el) return
      if(this._chart){ this._chart.dispose(); this._chart=null }
      const chart=echarts.init(el, null, { renderer:'canvas', devicePixelRatio:(window.devicePixelRatio||1) })
      this._chart = chart
      const x=this.stockData.map(x=>x.trade_date)
      const kdata=this.stockData.map(x=>[x.open,x.close,x.low,x.high])
      const series=[ { type:'candlestick', data:kdata, xAxisIndex:0, yAxisIndex:0, name:this.code, itemStyle:{ color:'#ec0000', color0:'#00da3c' } } ]
      // 主图评分线（副坐标）+ 面积着色（>0 绿色，<0 红色）
      if(this.showScore){
        const score = this.stockData.map(x=>x.final_score)
        // 评分主线
        series.push({ type:'line', data:score, name:'评分', xAxisIndex:0, yAxisIndex:1, smooth:true, lineStyle:{ width:1.2, color:'#7c3aed' }, symbol:'none', z:3 })
        // 面积着色：使用 stack 到 0 的方式，确保严格以 0 为基线
        const scorePos = score.map(v=>{ const n=Number(v); return isNaN(n) ? 0 : (n>0 ? n : 0) })
        const scoreNeg = score.map(v=>{ const n=Number(v); return isNaN(n) ? 0 : (n<0 ? n : 0) })
        series.push({ type:'line', data:scorePos, name:'_score_pos', xAxisIndex:0, yAxisIndex:1, stack:'scoreFill', smooth:false, connectNulls:false, symbol:'none', lineStyle:{ width:0 }, areaStyle:{ color:{ type:'linear', x:0,y:0,x2:0,y2:1, colorStops:[{offset:0,color:'rgba(16,185,129,0.25)'},{offset:1,color:'rgba(16,185,129,0.00)'}] } }, z:1, tooltip:{show:false}, silent:true })
        series.push({ type:'line', data:scoreNeg, name:'_score_neg', xAxisIndex:0, yAxisIndex:1, stack:'scoreFill', smooth:false, connectNulls:false, symbol:'none', lineStyle:{ width:0 }, areaStyle:{ color:{ type:'linear', x:0,y:0,x2:0,y2:1, colorStops:[{offset:0,color:'rgba(239,68,68,0.22)'},{offset:1,color:'rgba(239,68,68,0.00)'}] } }, z:1, tooltip:{show:false}, silent:true })
      }
      // 副图三选一
      if(this.subMode==='ma'){
        (this.ma||'').split(',').forEach(s=>{ const key='ma'+(s||'').trim(); if(key && this.stockData.length && this.stockData[0][key]!==undefined){ series.push({ type:'line', data:this.stockData.map(x=>x[key]), xAxisIndex:1, yAxisIndex:2, name:key.toUpperCase(), smooth:true, showSymbol:false, symbol:'none' }) } })
      } else if(this.subMode==='vol'){
        series.push({ type:'bar', data:this.stockData.map(x=>x.volume), name:'成交量', xAxisIndex:1, yAxisIndex:2, opacity:0.3 })
      } else if(this.subMode==='amt'){
        series.push({ type:'line', data:this.stockData.map(x=>x.turnover), name:'成交额', xAxisIndex:1, yAxisIndex:2, smooth:true, showSymbol:false, symbol:'none' })
      }
      // 动态计算副图坐标轴与尺寸（均线需要更合理的缩放区间）
      let subYAxisName = this.subMode==='ma' ? 'MA' : (this.subMode==='vol' ? '量' : '额')
      let subYAxis = { type:'value', name: subYAxisName, gridIndex:1, nameLocation:'middle', nameGap:40, axisLabel:{ formatter:v=>Number(v).toFixed(2), color:'#6b7280', margin: 8 } }
      if(this.subMode==='ma'){
        const keys = (this.ma||'').split(',').map(s=>('ma'+(s||'').trim())).filter(Boolean)
        const vals = []
        keys.forEach(k=>{ this.stockData.forEach(row=>{ const v=row[k]; if(v!==undefined && v!==null && !isNaN(v)) vals.push(Number(v)) }) })
        if(vals.length){
          const minV = Math.min.apply(null, vals)
          const maxV = Math.max.apply(null, vals)
          const span = Math.max(1e-6, maxV - minV)
          subYAxis.min = minV - span * 0.08
          subYAxis.max = maxV + span * 0.08
          subYAxis.scale = true
        }
      }
      const grids = [ { left:'3%', right:'3%', top:'10%', height: this.subMode==='ma' ? '56%' : '58%', containLabel:true }, { left:'3%', right:'3%', top: this.subMode==='ma' ? '76%' : '80%', height: this.subMode==='ma' ? '22%' : '18%', containLabel:true } ]

      // 动态控制横轴标签密度，避免重叠（移动端更稀疏）
      const labelInterval = x.length > (this.isMobile? 40 : 60) ? Math.ceil(x.length / (this.isMobile? 10 : 12)) : 'auto'

      chart.setOption({ 
        textStyle:{
          color:'#374151',
          fontSize:12,
          fontFamily:'-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif'
        },
        tooltip:{trigger:'axis'}, 
        grid: grids, 
        xAxis:[ 
          { type:'category', data:x, axisLabel:{ color:'#6b7280', hideOverlap:true, interval: labelInterval, rotate: this.isMobile?30:0, fontSize: this.isMobile?10:12 } }, 
          { type:'category', data:x, gridIndex:1, axisLabel:{ color:'#6b7280', hideOverlap:true, interval: labelInterval, rotate: this.isMobile?30:0, fontSize: this.isMobile?10:12 } } 
        ], 
        yAxis:[ 
          { type:'value', name:'价格', nameLocation:'middle', nameGap: this.isMobile?28:36, scale:true, axisLabel:{ formatter:v=>Number(v).toFixed(2), color:'#6b7280', margin: this.isMobile?4:6, fontSize: this.isMobile?10:12 }, gridIndex:0 }, 
          { type:'value', name:'评分', position:'right', gridIndex:0, min:-1, max:1, axisLine:{ show:true, lineStyle:{ color:'#8b5cf6' } }, axisLabel:{ color:'#8b5cf6', formatter:v=>Number(v).toFixed(1), margin: this.isMobile?4:6, fontSize: this.isMobile?10:12 }, splitLine:{ show:false }, nameTextStyle:{ color:'#8b5cf6' } }, 
          subYAxis 
        ], 
        series 
      })
      this.$nextTick(()=>{ chart.resize() })
    }
  }
}
  </script>

<style scoped>
.mobile-stock-page .el-card__body{ padding:8px }
.sysbt-drawer >>> .el-drawer__body{ padding:0; background:#f9fbff }
.sysbt-drawer-body{ padding:12px }
.sysbt-section{ background:#fff; border:1px solid #e6ebf2; border-radius:6px; padding:10px 12px; box-shadow:0 1px 2px rgba(0,0,0,0.03) }
.sysbt-sec-title{ font-size:13px; color:#374151; font-weight:600; margin-bottom:8px }
.sysbt-grid{ display:grid; grid-template-columns:1fr 1fr; gap:8px 12px }
.sysbt-grid-span{ grid-column:1 / span 2 }
.sysbt-drawer-actions{ display:flex; gap:8px; justify-content:flex-end; margin-top:6px }
.toolbar-btn{ height:28px; padding: 0 10px }
</style>
