<template>
  <div>
    <el-card v-loading="loading" element-loading-text="加载中...">
      <div slot="header" v-if="!isPortraitMobile">回测管理</div>

      <!-- 竖屏移动端：将按钮并入卡片标题右侧，保持简洁 -->
      <template v-if="isPortraitMobile">
        <div slot="header" style="display:flex;align-items:center;justify-content:space-between">
          <div>回测管理</div>
          <div style="display:flex;gap:8px;align-items:center">
            <el-button size="mini" type="primary" plain @click="showSettings=true"><i class="el-icon-setting" style="margin-right:4px"></i> 设置</el-button>
            <el-button size="mini" type="primary" plain :loading="loading" @click="fetchResults"><i class="el-icon-search" style="margin-right:4px"></i> 查询</el-button>
          </div>
        </div>
      </template>

      <!-- PC/横屏：展示完整工具表单 -->
      <el-form v-if="!isPortraitMobile" inline>
        <el-form-item label="模式">
          <el-tag size="mini" type="info">{{ modeText }}</el-tag>
        </el-form-item>
        <el-form-item label="对照">
          <el-radio-group v-model="compare" size="mini" @change="onCompareChange">
            <el-radio-button label="csi">沪深300</el-radio-button>
            <el-radio-button label="m">M值</el-radio-button>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="schema">
          <el-select v-model="qSchema" filterable placeholder="请选择或搜索 backtest_*" style="min-width:240px" @visible-change="onSchemaDropdownVisible" @change="onSchemaSelectChange">
            <el-option v-for="s in backtestSchemas" :key="s" :label="s" :value="s"></el-option>
          </el-select>
        </el-form-item>
        <el-form-item label="无风险年化%"><el-input v-model.number="rfAnnualPct" placeholder="2" style="width:100px"></el-input></el-form-item>
        <el-form-item label="区间">
          <el-date-picker v-model="range" type="daterange" :unlink-panels="true" value-format="yyyy-MM-dd" range-separator="至" start-placeholder="开始日期" end-placeholder="结束日期" :picker-options="pickerOptions" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :loading="loading" @click="fetchResults">查询</el-button>
        </el-form-item>
      </el-form>

      <!-- 竖屏移动端设置抽屉 -->
      <el-drawer :visible.sync="showSettings" title="回测设置" size="90%" :append-to-body="true" :destroy-on-close="true" custom-class="sysbt-drawer">
        <div class="sysbt-drawer-body">
          <div class="sysbt-section">
            <div class="sysbt-sec-title">基础参数</div>
            <el-form label-position="top" class="sysbt-form">
              <div class="sysbt-grid">
                <el-form-item label="模式">
                  <el-tag size="mini" type="info">{{ modeText }}</el-tag>
                </el-form-item>
                <el-form-item label="对照">
                  <el-radio-group v-model="compare" size="mini" @change="onCompareChange">
                    <el-radio-button label="csi">沪深300</el-radio-button>
                    <el-radio-button label="m">M值</el-radio-button>
                  </el-radio-group>
                </el-form-item>
                <el-form-item label="schema">
                  <el-select v-model="qSchema" filterable placeholder="请选择或搜索 backtest_*" @visible-change="onSchemaDropdownVisible" @change="onSchemaSelectChange">
                    <el-option v-for="s in backtestSchemas" :key="s" :label="s" :value="s"></el-option>
                  </el-select>
                </el-form-item>
                <el-form-item label="无风险年化%">
                  <el-input v-model.number="rfAnnualPct" placeholder="2"></el-input>
                </el-form-item>
                <el-form-item label="区间" class="sysbt-grid-span">
                  <el-date-picker v-model="range" type="daterange" :unlink-panels="true" value-format="yyyy-MM-dd" range-separator="至" start-placeholder="开始日期" end-placeholder="结束日期" :picker-options="pickerOptions" />
                </el-form-item>
              </div>
            </el-form>
          </div>
          <div class="sysbt-drawer-actions">
            <el-button size="mini" @click="showSettings=false">关闭</el-button>
            <el-button size="mini" type="primary" :loading="loading" @click="()=>{ showSettings=false; fetchResults() }">查询</el-button>
          </div>
        </div>
      </el-drawer>

      <div v-if="mode==='normal'">
        <!-- 统计概览：紧凑卡片栅格，更美观易读 -->
        <div class="stats" v-if="summary">
          <div class="stat">
            <div class="stat-label">年化</div>
            <div class="stat-value" :class="isPositive(summary.annualized)?'pos':'neg'">{{ fmtPct(summary.annualized) }}</div>
          </div>
          <div class="stat">
            <div class="stat-label">最大回撤</div>
            <div class="stat-value neg">{{ fmtPct(summary.max_drawdown) }}</div>
          </div>
          <div class="stat">
            <div class="stat-label">夏普</div>
            <div class="stat-value" :class="isPositive(summary.sharpe)?'pos':'neg'">{{ fmtNum(summary.sharpe) }}</div>
          </div>
          <div class="stat wide" v-if="rangeMin && rangeMax">
            <div class="stat-label">可选区间</div>
            <div class="stat-value muted">{{ rangeMin }} ~ {{ rangeMax }}</div>
          </div>
        </div>
        <div ref="chartEquity" style="width:100%;height:360px"></div>
        <div ref="chartSharpe" style="width:100%;height:200px;margin-top:8px"></div>
      </div>
      <div v-else style="color:#6b7280">M回测前端渲染开发中，先选择普通回测查看资金与夏普。</div>
    </el-card>
  </div>
  </template>

<script>
import axios from 'axios'
import * as echarts from 'echarts'
export default {
  name: 'SystemBacktest',
  data(){ return { 
    schemaList: [], backtestSchemas: [],
    mode: 'normal', compare: 'csi', qSchema: '', range: [], rfAnnualPct: 2,
    pickerOptions: { shortcuts: [] },
    loading: false, summary: null, rangeMin: '', rangeMax: '',
    _chartEquity: null, _chartSharpe: null,
    lastData: null,
    isMobile: false, isPortrait: true, showSettings: false
  }},
  created(){ this.onResize(); window.addEventListener('resize', this.onResize); window.addEventListener('orientationchange', this.onResize); this.loadSchemas() },
  beforeDestroy(){ window.removeEventListener('resize', this.onResize); window.removeEventListener('orientationchange', this.onResize) },
  methods: {
    onResize(){
      try{
        const w = window.innerWidth || 1024
        const h = window.innerHeight || 768
        this.isMobile = w <= 768
        this.isPortrait = h >= w
      }catch(e){ this.isMobile=false; this.isPortrait=true }
    },
    onSchemaDropdownVisible(v){ if(v){ this.loadSchemas() } },
    loadSchemas(){ axios.get('/webManager/system/schema').then(res=>{ if(res.data.code===0){ const all=res.data.data||[]; this.schemaList=all; this.backtestSchemas=(all||[]).filter(s=>s && s.indexOf('backtest_')===0).sort().reverse(); if(!this.qSchema && this.backtestSchemas.length){ this.qSchema=this.backtestSchemas[0]; this.onSchemaSelectChange(); } } }) },
    onModeChange(){},
    onCompareChange(){ if(this.lastData){ this.drawCharts(this.lastData) } },
    buildShortcuts(minDate, maxDate){
      const toStr=d=>`${d.getFullYear()}-${('0'+(d.getMonth()+1)).slice(-2)}-${('0'+d.getDate()).slice(-2)}`
      const add=(d, {y=0,m=0,days=0})=>{ const nd=new Date(d); nd.setFullYear(nd.getFullYear()+y); nd.setMonth(nd.getMonth()+m); nd.setDate(nd.getDate()+days); return nd }
      const min = new Date(minDate.replace(/-/g,'/')); const max = new Date(maxDate.replace(/-/g,'/'))
      this.pickerOptions = {
        shortcuts: [
          { text:'开始后一周', onClick:p=>p.$emit('pick',[toStr(min), toStr(add(min,{days:7}))]) },
          { text:'开始后一月', onClick:p=>p.$emit('pick',[toStr(min), toStr(add(min,{m:1}))]) },
          { text:'开始后半年', onClick:p=>p.$emit('pick',[toStr(min), toStr(add(min,{m:6}))]) },
          { text:'开始后一年', onClick:p=>p.$emit('pick',[toStr(min), toStr(add(min,{y:1}))]) },
          { text:'最后一周', onClick:p=>p.$emit('pick',[toStr(add(max,{days:-7})), toStr(max)]) },
          { text:'最后一月', onClick:p=>p.$emit('pick',[toStr(add(max,{m:-1})), toStr(max)]) },
          { text:'最后半年', onClick:p=>p.$emit('pick',[toStr(add(max,{m:-6})), toStr(max)]) },
          { text:'最近一年', onClick:p=>p.$emit('pick',[toStr(add(max,{y:-1})), toStr(max)]) }
        ]
      }
    },
    fetchResults(){
      if(!this.qSchema){ this.$message.error('请选择 schema'); return }
      // 模式由 schema 自动判定：backtest_* 为普通，m_* 为M回测（暂仅实现普通回测渲染）
      const autoMode = (this.qSchema||'').startsWith('backtest_') ? 'normal' : ((this.qSchema||'').startsWith('m_') ? 'm' : 'normal')
      this.mode = autoMode
      if(this.mode!=='normal'){ this.$message.info('M回测前端渲染开发中'); return }
      const params = { schema: this.qSchema, rf: (Number(this.rfAnnualPct)||2)/100 }
      if(this.range && this.range.length===2){ params.start=this.range[0]; params.end=this.range[1] }
      this.loading = true
      axios.get('/webManager/system/backtest/results',{ params })
        .then(res=>{
          if(res.data.code!==0){ this.$message.error(res.data.msg||'查询失败'); return }
          const d = res.data.data || {}
          this.summary = d.summary || null
          this.rangeMin = (d.range && d.range.min) || ''
          this.rangeMax = (d.range && d.range.max) || ''
          if(!this.range || this.range.length!==2){ this.range = [d.start||this.rangeMin, d.end||this.rangeMax] }
          this.buildShortcuts(this.rangeMin || d.start, this.rangeMax || d.end)
          this.drawCharts(d)
        })
        .catch(()=> this.$message.error('查询失败'))
        .finally(()=> this.loading=false)
    },
    onSchemaSelectChange(){ this.range=[]; this.fetchResults() },
    drawCharts(d){
      try{
        const el1=this.$refs.chartEquity; const el2=this.$refs.chartSharpe
        if(!el1 || !el2) return
        if(this._chartEquity){ this._chartEquity.dispose(); this._chartEquity=null }
        if(this._chartSharpe){ this._chartSharpe.dispose(); this._chartSharpe=null }
        const c1=echarts.init(el1, null, { renderer:'canvas', devicePixelRatio:(window.devicePixelRatio||1) })
        const c2=echarts.init(el2, null, { renderer:'canvas', devicePixelRatio:(window.devicePixelRatio||1) })
        this._chartEquity=c1; this._chartSharpe=c2
        const x=d.dates||[]; const equity=d.equity||[]; const m=d.m_value||[]; const sharpe=d.sharpe||[]; const csi=(d.csi300||[])
        // 将序列转换为 time 轴点位
        const toPoints = (xs, ys) => ys.map((v,i)=> (v===null||v===undefined)? [xs[i], null] : [xs[i], Number(v)])
        // 计算首日对齐所需：不改变沪深300数值，仅在右侧副轴上调整范围（min/max）
        const arrMinMax = (arr)=>{
          let mn=Number.POSITIVE_INFINITY, mx=Number.NEGATIVE_INFINITY
          for(let i=0;i<arr.length;i++){ const v=Number(arr[i]); if(!isNaN(v)){ if(v<mn) mn=v; if(v>mx) mx=v } }
          if(mn===Number.POSITIVE_INFINITY) return {min:0,max:1}
          return {min:mn,max:mx}
        }
        // 备注：曾尝试对右轴做“漂亮刻度”，但会破坏首日对齐，这里改为保持严格对齐的 min/max，仅做标签格式化
        const eMM = arrMinMax(equity)
        const cMM = arrMinMax(csi)
        const e0 = (equity && equity.length)? Number(equity[0]) : null
        const c0 = (csi && csi.length)? Number(csi.find(v=>v!==null&&v!==undefined)) : null
        // 固定左轴上下边界，防止 ECharts 自动“漂亮刻度”打破对齐
        let leftMin = eMM.min, leftMax = eMM.max
        if(!(isFinite(leftMin) && isFinite(leftMax))){ leftMin = 0; leftMax = 1 }
        if(Math.abs(leftMax-leftMin) < 1e-9){ leftMax = leftMin + 1 }
        let rightAxis = { type:'value', position:'right', nameLocation:'end', nameGap:6, scale:true, splitLine:{ show:false } }
        // 为后续左轴映射准备：全局保存右轴区间
        let csiMinGlobal = null, csiMaxGlobal = null
        if(this.compare==='csi'){
          // 右轴用于沪深300，并将首日对齐资金首日位置
          let csiAxisMin=null, csiAxisMax=null
          if(e0!=null && c0!=null && isFinite(leftMin) && isFinite(leftMax)){
            const eRange = Math.max(1e-9, leftMax - leftMin)
            const frac = Math.max(0, Math.min(1, (e0 - leftMin) / eRange))
            const up = Math.max(0, cMM.max - c0)
            const dn = Math.max(0, c0 - cMM.min)
            let R = 1
            const eps = 1e-9
            if(frac>eps && (1-frac)>eps){ R = Math.max(dn/frac, up/(1-frac)) * 1.05 }
            else if(frac<=eps){ R = up * 1.05 }
            else { R = dn * 1.05 }
            csiAxisMin = c0 - frac * R
            csiAxisMax = c0 + (1-frac) * R
          }
          // 兜底：避免 null/无穷或区间为0 导致 ECharts 报错
          if(!(isFinite(csiAxisMin) && isFinite(csiAxisMax))){
            csiAxisMin = cMM.min; csiAxisMax = cMM.max
          }
          if(!isFinite(csiAxisMin) || !isFinite(csiAxisMax) || Math.abs(csiAxisMax - csiAxisMin) < 1e-9){
            csiAxisMin = (isFinite(cMM.min)? cMM.min : 0)
            csiAxisMax = (isFinite(cMM.max)? cMM.max : csiAxisMin + 1)
            if(Math.abs(csiAxisMax - csiAxisMin) < 1e-9){ csiAxisMax = csiAxisMin + 1 }
          }
          // 保存给后续左轴映射使用
          csiMinGlobal = csiAxisMin; csiMaxGlobal = csiAxisMax
          rightAxis = { type:'value', position:'right', name:'沪深300', nameLocation:'end', nameGap:6, min:csiAxisMin, max:csiAxisMax, axisLine:{ show:true, lineStyle:{ color:'#10b981' } }, axisLabel:{ color:'#10b981', formatter:(v)=>this.fmtThousand(v, 0) }, splitLine:{ show:false } }
        }else{
          // 右轴用于M值（使用紫色，与资金主线蓝色区分）
          rightAxis = { type:'value', position:'right', name:'M值', nameLocation:'end', nameGap:6, scale:true, axisLine:{ show:true, lineStyle:{ color:'#7c3aed' } }, axisLabel:{ color:'#7c3aed', formatter:(v)=>Number(v).toFixed(2) }, splitLine:{ show:false } }
        }
        const sma=(arr,n)=>{ const out=[]; let sum=0; for(let i=0;i<arr.length;i++){ const v=Number(arr[i]); sum+=(isNaN(v)?0:v); if(i>=n){ const p=Number(arr[i-n]); sum-=(isNaN(p)?0:p) } out.push(i>=n-1? sum/n : null) } return out }
        const equitySmooth=sma(equity,20)
        c1.setOption({
          tooltip:{ trigger:'axis', formatter:(ps)=>{ const lines=[]; if(ps&&ps.length){ lines.push(ps[0].axisValueLabel || ps[0].axisValue) } (ps||[]).forEach(p=>{ const name=p.seriesName; const val=(name==='M值')? Number(p.data && p.data[1]).toFixed(2) : this.fmtCompact(p.data && p.data[1]); lines.push(`${p.marker} ${name}: ${val}`) }); return lines.join('<br/>') } },
          legend:{ top: 4, left: 8, data: (this.compare==='csi' ? ['资金','资金均线(20)','沪深300'] : ['资金','资金均线(20)','M值']), selected: { '资金均线(20)': false } },
          dataZoom:[ { type:'inside', xAxisIndex:[0,0] }, { type:'slider', xAxisIndex:[0,0], height:14, bottom:6 } ],
          grid:{ left:'6%', right:'12%', top: 56, bottom: 56, containLabel:true },
          xAxis:{ type:'time', boundaryGap:false, axisLabel:{ color:'#6b7280', hideOverlap:true } },
          yAxis:[
            { type:'value', name:'资金', position:'left', nameLocation:'end', nameGap:6, min:leftMin, max:leftMax, axisLabel:{ color:'#6b7280', formatter: (v)=>this.fmtCompact(v) }, splitLine:{ show:true, lineStyle:{ color:'#e5e7eb' } } },
            rightAxis
          ],
          series:[
            // 资金主线
            { type:'line', data: toPoints(x,equity), name:'资金', smooth:true, showSymbol:false, z:2, lineStyle:{ width:1, color:'#9ca3af' }, opacity:0.75 },
            // 资金均线
            { type:'line', data: toPoints(x,equitySmooth), name:'资金均线(20)', smooth:true, showSymbol:false, z:1, lineStyle:{ width:1.2, color:'#60a5fa' }, areaStyle:{ color:{ type:'linear', x:0,y:0,x2:0,y2:1, colorStops:[{offset:0,color:'rgba(96,165,250,0.16)'},{offset:1,color:'rgba(96,165,250,0.00)'}] } } },
            ...(this.compare==='csi'
              ? (()=>{
                  // 计算左轴上的沪深300等价序列
                  const leftMinV = leftMin, leftMaxV = leftMax
                  let rightMinV = csiMinGlobal, rightMaxV = csiMaxGlobal
                  if(!(isFinite(rightMinV) && isFinite(rightMaxV))){ rightMinV = cMM.min; rightMaxV = cMM.max }
                  if(!isFinite(rightMinV) || !isFinite(rightMaxV) || Math.abs(rightMaxV-rightMinV)<1e-9){ rightMinV = (isFinite(cMM.min)? cMM.min:0); rightMaxV = (isFinite(cMM.max)? cMM.max:rightMinV+1); if(Math.abs(rightMaxV-rightMinV)<1e-9){ rightMaxV = rightMinV+1 } }
                  const denom = Math.max(1e-9, (rightMaxV - rightMinV))
                  const csiLeft = csi.map(v=>{
                    if(v===null || v===undefined) return null
                    const vr = Number(v)
                    const frac = (vr - rightMinV) / denom
                    return leftMinV + frac * (leftMaxV - leftMinV)
                  })
                  const base = []
                  const diffPos = []
                  const diffNeg = []
                  for(let i=0;i<equity.length;i++){
                    const ev = Number(equity[i])
                    const cv = Number(csiLeft[i])
                    if(isNaN(ev) || isNaN(cv)){
                      base.push(null); diffPos.push(null); diffNeg.push(null)
                    }else{
                      base.push(Math.min(ev, cv))
                      const d = ev - cv
                      diffPos.push(d>0? d: 0)
                      diffNeg.push(d<0? -d: 0)
                    }
                  }
                  return [
                    { type:'line', data: toPoints(x,csi), name:'沪深300', yAxisIndex:1, smooth:true, showSymbol:false, z:1, lineStyle:{ color:'#10b981', width:0.8, type:'dashed', opacity:0.7 } },
                    { type:'line', data: toPoints(x,base), name:'_base', yAxisIndex:0, stack:'cmp', showSymbol:false, smooth:false, symbol:'none', clip:true, lineStyle:{ opacity:0 }, areaStyle:{ opacity:0 }, silent:true, tooltip:{show:false} },
                    { type:'line', data: toPoints(x,diffPos), name:'_alpha_pos', yAxisIndex:0, stack:'cmp', showSymbol:false, smooth:false, symbol:'none', clip:true, lineStyle:{ opacity:0 }, areaStyle:{ color:{ type:'linear', x:0,y:0,x2:0,y2:1, colorStops:[{offset:0,color:'rgba(16,185,129,0.18)'},{offset:1,color:'rgba(16,185,129,0.00)'}] } }, emphasis:{disabled:true}, tooltip:{show:false} },
                    { type:'line', data: toPoints(x,diffNeg), name:'_alpha_neg', yAxisIndex:0, stack:'cmp', showSymbol:false, smooth:false, symbol:'none', clip:true, lineStyle:{ opacity:0 }, areaStyle:{ color:{ type:'linear', x:0,y:0,x2:0,y2:1, colorStops:[{offset:0,color:'rgba(239,68,68,0.18)'},{offset:1,color:'rgba(239,68,68,0.00)'}] } }, emphasis:{disabled:true}, tooltip:{show:false} }
                  ]
                })()
              : [{ type:'line', data: toPoints(x,m), name:'M值', yAxisIndex:1, smooth:true, showSymbol:false, z:2, lineStyle:{ color:'#7c3aed', width:1.2, type:'dashed' } }]
            )
          ]
        })
        const sharpeSmooth=sma(sharpe,10)
        // 夏普轴采用鲁棒范围：去掉极端 2%/98% 以避免早期极值把坐标挤扁
        const onlyNums = (sharpe||[]).map(Number).filter(v=>!isNaN(v))
        const percentile=(arr,p)=>{ if(!arr.length) return 0; const a=[...arr].sort((a,b)=>a-b); const idx=(a.length-1)*p; const lo=Math.floor(idx), hi=Math.ceil(idx); if(lo===hi) return a[lo]; const w=idx-lo; return a[lo]*(1-w)+a[hi]*w }
        let yMinR = null, yMaxR = null
        if(onlyNums.length){
          const p2 = percentile(onlyNums, 0.02)
          const p98 = percentile(onlyNums, 0.98)
          if(isFinite(p2) && isFinite(p98) && p98>p2){ const pad=(p98-p2)*0.08; yMinR=p2-pad; yMaxR=p98+pad }
        }
        if(yMinR!=null && yMaxR!=null){ yMinR=Math.min(yMinR,0); yMaxR=Math.max(yMaxR,0) }
        const sharpeMain=(sharpe||[]).map(v=>{ const n=Number(v); return isNaN(n)? null:n })
        const sharpePos=sharpeMain.map(v=> v==null? null : (v>0? v:null))
        const sharpeNeg=sharpeMain.map(v=> v==null? null : (v<0? v:null))
        // sharpe 图与主图联动（共用 x 轴 index 0，通过 axisPointer link）
        c2.setOption({
          tooltip:{ trigger:'axis' },
          legend:{ top: 4, left: 8, data:['夏普','夏普均线(10)'] },
          grid:{ left:'6%', right:'12%', top: 46, bottom: 24, containLabel:true },
          xAxis:{ type:'time', boundaryGap:false, axisLabel:{ color:'#6b7280', hideOverlap:true }, axisPointer:{ show:true, label:{show:false} } },
          yAxis:{ type:'value', name:'夏普', nameLocation:'end', nameGap:6, min:yMinR, max:yMaxR, axisLine:{ show:true, lineStyle:{ color:'#7c3aed' } }, axisLabel:{ color:'#7c3aed', formatter:(v)=>Number(v).toFixed(2) }, splitLine:{ show:true, lineStyle:{ color:'#eef2ff', type:'dashed' } } },
          series:[
            { type:'line', name:'_sharp_pos', data: toPoints(x,sharpePos), smooth:true, showSymbol:false, z:2, lineStyle:{ width:0 }, areaStyle:{ opacity:1, color:{ type:'linear', x:0,y:0,x2:0,y2:1, colorStops:[{offset:0,color:'rgba(16,185,129,0.25)'},{offset:1,color:'rgba(16,185,129,0.00)'}] } }, emphasis:{disabled:true}, tooltip:{show:false}, silent:true },
            { type:'line', name:'_sharp_neg', data: toPoints(x,sharpeNeg), smooth:true, showSymbol:false, z:1, lineStyle:{ width:0 }, areaStyle:{ opacity:1, color:{ type:'linear', x:0,y:0,x2:0,y2:1, colorStops:[{offset:0,color:'rgba(239,68,68,0.22)'},{offset:1,color:'rgba(239,68,68,0.00)'}] } }, emphasis:{disabled:true}, tooltip:{show:false}, silent:true },
            { type:'line', data: toPoints(x,sharpeMain), name:'夏普', smooth:true, showSymbol:false, lineStyle:{ color:'#7c3aed', width:1 } },
            { type:'line', data: toPoints(x,sharpeSmooth), name:'夏普均线(10)', smooth:true, showSymbol:false, lineStyle:{ color:'#a78bfa', width:1.2, type:'dashed' } }
          ]
        })
        // 两图 axisPointer 联动 & 数据缩放联动
        echarts.connect([c1, c2])
        this.lastData=d
      }catch(e){
        // 将错误输出，避免“白屏无报错”
        try{ console.error('[SystemBacktest] drawCharts error:', e) }catch(_){}
        if(this.$message){ this.$message.error('绘图失败，请查看控制台错误') }
      }
    },
    fmtCompact(v){ const n=Number(v); if(isNaN(n)) return '-'; if(Math.abs(n)>=1e8) return (n/1e8).toFixed(2)+'亿'; if(Math.abs(n)>=1e4) return (n/1e4).toFixed(2)+'万'; return n.toFixed(0) },
    fmtPct(v){ if(v===null||v===undefined) return '-'; const n=Number(v); if(isNaN(n)) return '-'; return (n*100).toFixed(2)+'%' },
    fmtNum(v){ if(v===null||v===undefined) return '-'; const n=Number(v); if(isNaN(n)) return '-'; return n.toFixed(2) },
    fmtThousand(v, decimals=0){ const n=Number(v); if(isNaN(n)) return '-'; const f = n.toFixed(decimals); return f.replace(/\B(?=(\d{3})+(?!\d))/g, ',') }
  },
  computed:{
    isPortraitMobile(){ return this.isMobile && this.isPortrait },
    modeText(){ return this.mode==='normal' ? '普通回测' : 'M回测' },
    isPositive(){ return (v)=> Number(v||0) >= 0 }
  }
}
  </script>

<style scoped>
.summary{ display:flex; flex-wrap:wrap; gap:16px; margin: 6px 0 10px; color:#374151 }
.stats{ display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:8px; margin:6px 0 10px }
.stat{ background:#fff; border:1px solid #e6ebf2; border-radius:6px; padding:8px 10px }
.stat.wide{ grid-column:1 / span 3 }
.stat-label{ color:#6b7280; font-size:12px; margin-bottom:4px }
.stat-value{ font-weight:700; color:#111827 }
.stat-value.pos{ color:#059669 }
.stat-value.neg{ color:#b91c1c }
.stat-value.muted{ color:#6b7280; font-weight:500 }
.sysbt-drawer{ --pad:12px; }
.sysbt-drawer >>> .el-drawer__header{ margin-bottom:0; padding: var(--pad) var(--pad) 0 var(--pad); border-bottom:1px solid #e6ebf2; }
.sysbt-drawer >>> .el-drawer__body{ padding:0; background:#f9fbff }
.sysbt-drawer-body{ padding:12px }
.sysbt-section{ background:#fff; border:1px solid #e6ebf2; border-radius:6px; padding:10px 12px; box-shadow:0 1px 2px rgba(0,0,0,0.03) }
.sysbt-sec-title{ font-size:13px; color:#374151; font-weight:600; margin-bottom:8px }
.sysbt-grid{ display:grid; grid-template-columns:1fr 1fr; gap:8px 12px }
.sysbt-grid-span{ grid-column:1 / span 2 }
.sysbt-drawer .el-form-item{ margin-bottom:10px }
.sysbt-drawer-actions{ display:flex; gap:8px; justify-content:flex-end; margin-top:6px }
</style>
