<template>
  <div v-loading="loading" element-loading-text="加载中..." style="height:100%">
    <!-- 移动端：按钮并入标题栏右侧 -->

    <el-row :gutter="12" style="height:100%">
      <el-col v-if="!isMobile" :xs="24" :sm="7" :md="6" style="height:100%">
        <el-card style="height:100%;display:flex;flex-direction:column">
          <div slot="header">ETF列表</div>
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
            <el-table-column prop="fund_code" label="代码" width="120" />
            <el-table-column prop="fund_name" label="名称" />
          </el-table>
        </el-card>
      </el-col>
      <el-col :xs="24" :sm="17" :md="18" style="height:100%">
        <el-card style="height:100%;display:flex;flex-direction:column">
          <div slot="header" style="display:flex;align-items:center;justify-content:space-between">
            <div>
              {{ currentName || 'ETF K线' }} <span v-if="code" style="color:#6b7280;font-weight:normal">（{{code}}）</span>
            </div>
            <div style="display:flex; gap:8px; align-items:center">
              <template v-if="isMobile">
                <el-button size="mini" type="primary" plain class="toolbar-btn" @click="showDrawer=true">
                  <i class="el-icon-menu" style="margin-right:4px"></i> ETF
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
                <el-date-picker v-model="etfRange" type="daterange" value-format="yyyy-MM-dd" range-separator="至" start-placeholder="开始日期" end-placeholder="结束日期" :picker-options="pickerOptions" />
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
              <el-form-item>
                <el-button type="primary" :loading="loading" @click="fetchETF">查询</el-button>
              </el-form-item>
            </el-form>
          </div>
          <div ref="etfChart" :style="chartStyle"></div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 移动端抽屉：ETF列表 -->
    <el-drawer :visible.sync="showDrawer" title="ETF列表" size="80%" :append-to-body="true" custom-class="etf-drawer" :destroy-on-close="true">
      <div style="padding:0 8px 8px 8px">
        <div style="display:flex;padding-bottom:8px">
          <el-input v-model="keyword" placeholder="代码/名称" size="small" @keyup.enter.native="searchList" style="flex:1;margin-right:6px" />
          <el-button size="small" type="primary" @click="searchList">搜索</el-button>
        </div>
        <el-table :data="list" size="mini" highlight-current-row @row-click="(r)=>{ handleSelect(r); showDrawer=false }" :height="400">
          <el-table-column prop="fund_code" label="代码" width="120" />
          <el-table-column prop="fund_name" label="名称" />
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
                <el-date-picker v-model="etfRange" type="daterange" :unlink-panels="true" value-format="yyyy-MM-dd" range-separator="至" start-placeholder="开始日期" end-placeholder="结束日期" :picker-options="pickerOptions" />
              </el-form-item>
              <el-form-item label="显示">
                <el-radio-group v-model="subMode" @change="onSubModeChange">
                  <el-radio-button label="ma">均线</el-radio-button>
                  <el-radio-button label="vol">成交量</el-radio-button>
                  <el-radio-button label="amt">成交额</el-radio-button>
                </el-radio-group>
              </el-form-item>
              <el-form-item v-if="subMode==='ma'" label="均线">
                <el-input v-model="ma" placeholder="5,10,20" @change="onMAChange" />
              </el-form-item>
              <el-form-item label="后复权">
                <el-switch v-model="useHfq" @change="onHfqChange" />
              </el-form-item>
            </div>
          </el-form>
        </div>
        
        <div class="sysbt-drawer-actions">
          <el-button @click="showSettingDrawer = false">取消</el-button>
          <el-button type="primary" :loading="loading" @click="fetchETF">查询</el-button>
        </div>
      </div>
    </el-drawer>
  </div>
  </template>

<script>
import axios from 'axios'
import * as echarts from 'echarts'
export default {
  name: 'DailyETF',
  data(){ return { 
    code:'', 
    currentName:'', 
    options:[], 
    searching:false, 
    etfRange:[], 
    ma:'5,10,20', 
    subMode:'ma', 
    useHfq:false, 
    etfData:[], 
    pickerOptions:{}, 
    loading:false, 
    keyword:'', 
    list:[], 
    tableHeight:360, 
    isMobile:false, 
    showDrawer:false, 
    showSettingDrawer:false, 
    _chart:null, 
    chartHeight:380
  } },
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
    this.etfRange = [fmt(start), fmt(end)]
    this.$nextTick(()=>{ 
      this.onResize();
      const q = this.$route && this.$route.query ? this.$route.query : {}
      if(q.code){
        this.code = q.code
        this.currentName = q.name || this.currentName
        this.fetchETF()
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
    remoteSearch(q){ if(!q){ this.options=[]; return } this.searching=true; axios.get('/webManager/etf/search',{ params:{ q } }).then(r=>{ if(r.data.code===0) this.options=r.data.data||[] }).finally(()=> this.searching=false) },
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
      axios.get('/webManager/etf/search', { params:{ q } })
        .then(res=>{
          if(res.data.code===0){
            this.list = res.data.data || []
            if(this.list.length){
              if(!noAutoSelect){
                const first = this.list[0]
                this.code = first.fund_code
                this.currentName = first.fund_name || this.currentName
                this.fetchETF()
              }
            }
          } else {
            this.$message.error(res.data.msg)
          }
        })
        .catch(()=> this.$message.error('搜索失败'))
        .finally(()=> this.loading = false)
    },
    handleSelect(row){ if(!row || !row.fund_code) return; this.code = row.fund_code; this.currentName = row.fund_name || this.currentName; this.fetchETF() },
    fetchETF(){
      if(!this.code || !this.etfRange || this.etfRange.length!==2){ this.$message.error('缺少参数'); return }
      const [start, end] = this.etfRange
      this.loading = true
      const withVol = this.subMode==='vol' ? 1 : 0
      const withAmt = this.subMode==='amt' ? 1 : 0
      const maParam = this.subMode==='ma' ? this.ma : ''
      axios.get('/webManager/daily/etf', { params: { code:this.code, start, end, ma:maParam, with_vol:withVol, with_amt:withAmt, hfq:this.useHfq?1:0 } })
        .then(res=>{ 
          console.log('ETF查询响应:', res.data); // 添加调试日志
          if(res.data.code===0){ 
            this.etfData=res.data.data||[]; 
            this.draw() 
          } else { 
            console.error('ETF查询业务错误:', res.data.msg); // 添加调试日志
            this.$message.error(res.data.msg) 
          } 
        })
        .catch(err=>{ 
          console.error('ETF查询网络错误:', err); // 添加详细错误信息
          this.$message.error(`查询失败: ${err.message || '网络错误'}`) 
        })
        .finally(()=> this.loading=false)
    },
    onSubModeChange(){ this.fetchETF() },
    onMAChange(){ if(this.subMode==='ma'){ this.fetchETF() } },
    onHfqChange(){ this.fetchETF() },
    draw(){
      const el=this.$refs.etfChart; if(!el) return
      if(this._chart){ this._chart.dispose(); this._chart=null }
      
      // 检查数据是否有效
      if(!this.etfData || this.etfData.length === 0){
        console.warn('ETF数据为空，无法绘制图表');
        return;
      }
      
      try {
        const chart=echarts.init(el, null, { renderer:'canvas', devicePixelRatio:(window.devicePixelRatio||1) })
        this._chart = chart
        const x=this.etfData.map(x=>x.trade_date)
        const kdata=this.etfData.map(x=>[x.trade_date,x.open,x.close,x.low,x.high])
      const toPoints = (ys)=> ys.map((v,i)=> [x[i], v])
      const series=[ { type:'candlestick', data:kdata, xAxisIndex:0, yAxisIndex:0, name:this.code, encode:{ x:0, y:[1,2,3,4] }, itemStyle:{ color:'#ec0000', color0:'#00da3c' } } ]
      
      // ETF暂时没有分数线，不绘制分数
      
      // 副图三选一
      if(this.subMode==='ma'){
        (this.ma||'').split(',').forEach(s=>{ const key='ma'+(s||'').trim(); if(key && this.etfData.length && this.etfData[0][key]!==undefined){ series.push({ type:'line', data:toPoints(this.etfData.map(x=>x[key])), xAxisIndex:1, yAxisIndex:2, name:key.toUpperCase(), smooth:true, showSymbol:false, symbol:'none', encode:{ x:0, y:1 } }) } })
      } else if(this.subMode==='vol'){
        series.push({ type:'bar', data:toPoints(this.etfData.map(x=>x.volume)), name:'成交量', xAxisIndex:1, yAxisIndex:2, opacity:0.3, encode:{ x:0, y:1 } })
      } else if(this.subMode==='amt'){
        series.push({ type:'line', data:toPoints(this.etfData.map(x=>x.turnover)), name:'成交额', xAxisIndex:1, yAxisIndex:2, smooth:true, showSymbol:false, symbol:'none', encode:{ x:0, y:1 } })
      }
      // 动态计算副图坐标轴与尺寸（均线需要更合理的缩放区间）
      let subYAxisName = this.subMode==='ma' ? 'MA' : (this.subMode==='vol' ? '量' : '额')
      let subYAxis = { type:'value', name: subYAxisName, gridIndex:1, nameLocation:'middle', nameGap:40, axisLabel:{ formatter:v=>Number(v).toFixed(2), color:'#6b7280', margin: 8 } }
      if(this.subMode==='ma'){
        const keys = (this.ma||'').split(',').map(s=>('ma'+(s||'').trim())).filter(Boolean)
        const vals = []
        keys.forEach(k=>{ this.etfData.forEach(row=>{ const v=row[k]; if(v!==undefined && v!==null && !isNaN(v)) vals.push(Number(v)) }) })
        if(vals.length){
          const minV = Math.min.apply(null, vals)
          const maxV = Math.max.apply(null, vals)
          const span = Math.max(1e-6, maxV - minV)
          subYAxis.min = minV - span * 0.08
          subYAxis.max = maxV + span * 0.08
          subYAxis.scale = true
        }
      }
      const grids = [ { left: this.isMobile? 38 : '3%', right: this.isMobile? 38 : '3%', top:'10%', height: this.subMode==='ma' ? '70%' : '72%', containLabel:true }, { left: this.isMobile? 38 : '3%', right: this.isMobile? 38 : '3%', top: this.subMode==='ma' ? '82%' : '84%', height: this.subMode==='ma' ? '16%' : '14%', containLabel:true } ]

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
          { type:'time', boundaryGap:false, axisLabel:{ color:'#6b7280', hideOverlap:true, interval: labelInterval, rotate: this.isMobile?30:0, fontSize: this.isMobile?10:12 } }, 
          { type:'time', gridIndex:1, boundaryGap:false, axisLabel:{ color:'#6b7280', hideOverlap:true, interval: labelInterval, rotate: this.isMobile?30:0, fontSize: this.isMobile?10:12 } } 
        ], 
        yAxis:[ 
          { type:'value', name:'价格', nameLocation:'middle', nameGap: this.isMobile?28:36, scale:true, axisLabel:{ formatter:v=>Number(v).toFixed(2), color:'#6b7280', margin: this.isMobile?6:8, fontSize: this.isMobile?10:12 }, gridIndex:0 }, 
          { type:'value', name:'', gridIndex:0, show:false }, // 占位yAxis，索引1
          subYAxis 
        ], 
        series 
      })
      this.$nextTick(()=>{ chart.resize() })
      } catch (error) {
        console.error('ETF图表绘制失败:', error);
        this.$message.error('图表绘制失败，请检查数据格式');
      }
    }
  }
}
  </script>

<style scoped>
.mobile-etf-page .el-card__body{ padding:8px }
.sysbt-drawer >>> .el-drawer__body{ padding:0; background:#f9fbff }
.sysbt-drawer-body{ padding:12px }
.sysbt-section{ background:#fff; border:1px solid #e6ebf2; border-radius:6px; padding:10px 12px; box-shadow:0 1px 2px rgba(0,0,0,0.03) }
.sysbt-sec-title{ font-size:13px; color:#374151; font-weight:600; margin-bottom:8px }
.sysbt-grid{ display:grid; grid-template-columns:1fr 1fr; gap:8px 12px }
.sysbt-grid-span{ grid-column:1 / span 2 }
.sysbt-drawer-actions{ display:flex; gap:8px; justify-content:flex-end; margin-top:6px }
.toolbar-btn{ height:28px; padding: 0 10px }
</style>
