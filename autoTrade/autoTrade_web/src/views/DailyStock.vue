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
                <el-button size="mini" type="success" plain class="toolbar-btn" :loading="aiEvaluating" @click="showAiEvaluationDialog" :disabled="!code">
                  <i class="el-icon-cpu" style="margin-right:4px"></i> AI评测
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
                <el-button type="success" :loading="aiEvaluating" @click="showAiEvaluationDialog" :disabled="!code">
                  <i class="el-icon-cpu" style="margin-right:4px"></i>AI评测
                </el-button>
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
    
    <!-- AI评测对话框 -->
    <el-dialog
      :title="`AI评测${currentName || '个股'}走势`"
      :visible.sync="showAiDialog"
      :width="isMobile ? '95%' : '600px'"
      :top="isMobile ? '5vh' : '15vh'"
      :class="{ 
        'ai-evaluation-mobile': isMobile, 
        'ai-evaluation-dialog': !isMobile,
        'has-result': !!aiResult
      }"
      :append-to-body="isMobile"
      :close-on-click-modal="false"
      :close-on-press-escape="true"
      :destroy-on-close="true">
      
      <!-- 模型选择 -->
      <div v-if="!aiResult" class="ai-evaluation-content">
        <el-form label-width="100px">
          <el-form-item label="选择AI模型">
            <el-select v-model="selectedModelId" placeholder="请选择AI模型" style="width: 100%;">
              <el-option
                v-for="model in availableModels"
                :key="model.id"
                :label="model.name"
                :value="model.id">
                <span style="float: left">{{ model.name }}</span>
                <span style="float: right; color: #8492a6; font-size: 13px">{{ model.source_name }}</span>
              </el-option>
            </el-select>
          </el-form-item>
        </el-form>
        
        <!-- 评测进度 -->
        <div v-if="aiEvaluating" class="ai-evaluation-progress">
          <el-progress :percentage="aiProgress" :status="aiProgressStatus"></el-progress>
          <p class="progress-text">{{ aiProgressText }}</p>
        </div>
      </div>
      
      <!-- 评测结果 -->
      <div v-if="aiResult" class="ai-evaluation-result">
        <!-- 竖屏模式：紧凑布局 -->
        <div v-if="isMobile" class="mobile-compact-layout">
          <!-- 综合分数 - 水平布局 -->
          <div class="mobile-overall-score">
            <div class="score-circle" :class="getScoreClass(aiResult.analysis_result['综合看涨分数'])">
              <div class="score-value">{{ aiResult.analysis_result['综合看涨分数'] }}</div>
              <div class="score-label">综合{{ getScoreLabel(aiResult.analysis_result['综合看涨分数']) }}</div>
            </div>
            <div class="mobile-summary-text">
              <h4>AI分析总结</h4>
              <p>{{ aiResult.analysis_result['总结'] }}</p>
            </div>
          </div>
          
          <!-- 子分数 - 网格布局 -->
          <div class="mobile-sub-scores">
            <div class="mobile-score-item">
              <div class="score-name">趋势动能</div>
              <div class="score-value-text">{{ aiResult.analysis_result['趋势动能看涨分数'] }}</div>
              <el-progress 
                :percentage="Math.abs(aiResult.analysis_result['趋势动能看涨分数'])" 
                :color="getProgressColor(aiResult.analysis_result['趋势动能看涨分数'])"
                :show-text="false"
                :stroke-width="4">
              </el-progress>
            </div>
            
            <div class="mobile-score-item">
              <div class="score-name">均值回归</div>
              <div class="score-value-text">{{ aiResult.analysis_result['均值回归看涨分数'] }}</div>
              <el-progress 
                :percentage="Math.abs(aiResult.analysis_result['均值回归看涨分数'])" 
                :color="getProgressColor(aiResult.analysis_result['均值回归看涨分数'])"
                :show-text="false"
                :stroke-width="4">
              </el-progress>
            </div>
            
            <div class="mobile-score-item">
              <div class="score-name">质量波动</div>
              <div class="score-value-text">{{ aiResult.analysis_result['质量波动看涨分数'] }}</div>
              <el-progress 
                :percentage="Math.abs(aiResult.analysis_result['质量波动看涨分数'])" 
                :color="getProgressColor(aiResult.analysis_result['质量波动看涨分数'])"
                :show-text="false"
                :stroke-width="4">
              </el-progress>
            </div>
          </div>
          
          <!-- 数据信息 -->
          <div class="mobile-data-info">
            <p><strong>分析数据：</strong>{{ aiResult.data_period.start_date }} 至 {{ aiResult.data_period.end_date }} ({{ aiResult.data_period.data_count }}个交易日)</p>
          </div>
        </div>
        
        <!-- PC模式：原有布局 -->
        <div v-else class="desktop-layout">
          <!-- 综合分数 -->
          <div class="overall-score">
            <div class="score-circle" :class="getScoreClass(aiResult.analysis_result['综合看涨分数'])">
              <div class="score-value">{{ aiResult.analysis_result['综合看涨分数'] }}</div>
              <div class="score-label">综合{{ getScoreLabel(aiResult.analysis_result['综合看涨分数']) }}</div>
            </div>
          </div>
          
          <!-- 子分数 -->
          <div class="sub-scores">
            <div class="score-item">
              <div class="score-name">趋势动能{{ getScoreLabel(aiResult.analysis_result['趋势动能看涨分数']) }}</div>
              <el-progress 
                :percentage="Math.abs(aiResult.analysis_result['趋势动能看涨分数'])" 
                :color="getProgressColor(aiResult.analysis_result['趋势动能看涨分数'])"
                :show-text="false">
              </el-progress>
              <div class="score-value-text">{{ aiResult.analysis_result['趋势动能看涨分数'] }}</div>
            </div>
            
            <div class="score-item">
              <div class="score-name">均值回归{{ getScoreLabel(aiResult.analysis_result['均值回归看涨分数']) }}</div>
              <el-progress 
                :percentage="Math.abs(aiResult.analysis_result['均值回归看涨分数'])" 
                :color="getProgressColor(aiResult.analysis_result['均值回归看涨分数'])"
                :show-text="false">
              </el-progress>
              <div class="score-value-text">{{ aiResult.analysis_result['均值回归看涨分数'] }}</div>
            </div>
            
            <div class="score-item">
              <div class="score-name">质量波动{{ getScoreLabel(aiResult.analysis_result['质量波动看涨分数']) }}</div>
              <el-progress 
                :percentage="Math.abs(aiResult.analysis_result['质量波动看涨分数'])" 
                :color="getProgressColor(aiResult.analysis_result['质量波动看涨分数'])"
                :show-text="false">
              </el-progress>
              <div class="score-value-text">{{ aiResult.analysis_result['质量波动看涨分数'] }}</div>
            </div>
          </div>
          
          <!-- 总结 -->
          <div class="ai-summary">
            <h4>AI分析总结</h4>
            <p>{{ aiResult.analysis_result['总结'] }}</p>
          </div>
          
          <!-- 数据信息 -->
          <div class="data-info">
            <p><strong>分析数据：</strong>{{ aiResult.data_period.start_date }} 至 {{ aiResult.data_period.end_date }} ({{ aiResult.data_period.data_count }}个交易日)</p>
          </div>
        </div>
      </div>
      
      <div slot="footer" class="dialog-footer">
        <el-button @click="showAiDialog = false">关闭</el-button>
        <el-button v-if="aiResult" type="primary" @click="resetAiEvaluation">重新评测</el-button>
        <el-button v-if="!aiResult && selectedModelId && !aiEvaluating" type="primary" @click="startAiEvaluation">开始评测</el-button>
        <el-button v-if="aiEvaluating" type="primary" :loading="true" disabled>分析中...</el-button>
      </div>
    </el-dialog>
  </div>
  </template>

<script>
import axios from 'axios'
import * as echarts from 'echarts'
export default {
  name: 'DailyStock',
  data(){ return { code:'', currentName:'', options:[], searching:false, stockRange:[], ma:'5,10,20', subMode:'ma', useHfq:false, showScore:true, stockData:[], pickerOptions:{}, loading:false, keyword:'', list:[], tableHeight:360, isMobile:false, showDrawer:false, showSettingDrawer:false, _chart:null, chartHeight:380, 
    // AI评测相关
    showAiDialog: false,
    selectedModelId: null,
    availableModels: [],
    aiEvaluating: false,
    aiProgress: 0,
    aiProgressStatus: '',
    aiProgressText: '',
    aiResult: null
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
    this.loadAvailableModels()
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
      const kdata=this.stockData.map(x=>[x.trade_date,x.open,x.close,x.low,x.high])
      const toPoints = (ys)=> ys.map((v,i)=> [x[i], v])
      const series=[ { type:'candlestick', data:kdata, xAxisIndex:0, yAxisIndex:0, name:this.code, encode:{ x:0, y:[1,2,3,4] }, itemStyle:{ color:'#ec0000', color0:'#00da3c' } } ]
      // 主图评分线（副坐标）+ 面积着色（>0 绿色，<0 红色）
      if(this.showScore){
        const score = this.stockData.map(x=>x.final_score)
        // 评分主线
        series.push({ type:'line', data:toPoints(score), name:'评分', xAxisIndex:0, yAxisIndex:1, smooth:true, lineStyle:{ width:1.2, color:'#7c3aed' }, symbol:'none', z:3, encode:{ x:0, y:1 }, showSymbol:false })
        // 面积着色：使用 stack 到 0 的方式，确保严格以 0 为基线
        const scorePos = score.map(v=>{ const n=Number(v); return isNaN(n) ? 0 : (n>0 ? n : 0) })
        const scoreNeg = score.map(v=>{ const n=Number(v); return isNaN(n) ? 0 : (n<0 ? n : 0) })
        series.push({ type:'line', data:toPoints(scorePos), name:'_score_pos', xAxisIndex:0, yAxisIndex:1, stack:'scoreFill', smooth:false, connectNulls:false, symbol:'none', lineStyle:{ width:0 }, areaStyle:{ color:{ type:'linear', x:0,y:0,x2:0,y2:1, colorStops:[{offset:0,color:'rgba(16,185,129,0.25)'},{offset:1,color:'rgba(16,185,129,0.00)'}] } }, z:1, tooltip:{show:false}, silent:true, encode:{ x:0, y:1 } })
        series.push({ type:'line', data:toPoints(scoreNeg), name:'_score_neg', xAxisIndex:0, yAxisIndex:1, stack:'scoreFill', smooth:false, connectNulls:false, symbol:'none', lineStyle:{ width:0 }, areaStyle:{ color:{ type:'linear', x:0,y:0,x2:0,y2:1, colorStops:[{offset:0,color:'rgba(239,68,68,0.22)'},{offset:1,color:'rgba(239,68,68,0.00)'}] } }, z:1, tooltip:{show:false}, silent:true, encode:{ x:0, y:1 } })
      }
      // 副图三选一
      if(this.subMode==='ma'){
        (this.ma||'').split(',').forEach(s=>{ const key='ma'+(s||'').trim(); if(key && this.stockData.length && this.stockData[0][key]!==undefined){ series.push({ type:'line', data:toPoints(this.stockData.map(x=>x[key])), xAxisIndex:1, yAxisIndex:2, name:key.toUpperCase(), smooth:true, showSymbol:false, symbol:'none', encode:{ x:0, y:1 } }) } })
      } else if(this.subMode==='vol'){
        series.push({ type:'bar', data:toPoints(this.stockData.map(x=>x.volume)), name:'成交量', xAxisIndex:1, yAxisIndex:2, opacity:0.3, encode:{ x:0, y:1 } })
      } else if(this.subMode==='amt'){
        series.push({ type:'line', data:toPoints(this.stockData.map(x=>x.turnover)), name:'成交额', xAxisIndex:1, yAxisIndex:2, smooth:true, showSymbol:false, symbol:'none', encode:{ x:0, y:1 } })
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
      const grids = [ { left: this.isMobile? 38 : '3%', right: this.isMobile? 38 : '3%', top:'10%', height: this.subMode==='ma' ? '56%' : '58%', containLabel:true }, { left: this.isMobile? 38 : '3%', right: this.isMobile? 38 : '3%', top: this.subMode==='ma' ? '76%' : '80%', height: this.subMode==='ma' ? '22%' : '18%', containLabel:true } ]

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
          { type:'value', name:'评分', position:'right', gridIndex:0, min:-1, max:1, axisLine:{ show:true, lineStyle:{ color:'#8b5cf6' } }, axisLabel:{ color:'#8b5cf6', formatter:v=>Number(v).toFixed(1), margin: this.isMobile?6:8, fontSize: this.isMobile?10:12 }, splitLine:{ show:false }, nameTextStyle:{ color:'#8b5cf6' } }, 
          subYAxis 
        ], 
        series 
      })
      this.$nextTick(()=>{ chart.resize() })
    },
    
    // AI评测相关方法
    async loadAvailableModels() {
      try {
        const response = await axios.get('/webManager/ai/available/models')
        if (response.data.code === 0) {
          this.availableModels = response.data.data || []
        }
      } catch (error) {
        console.error('加载AI模型失败:', error)
      }
    },
    
    showAiEvaluationDialog() {
      if (!this.code) {
        this.$message.error('请先选择股票')
        return
      }
      this.showAiDialog = true
      this.resetAiEvaluation()
    },
    
    resetAiEvaluation() {
      this.aiResult = null
      this.aiEvaluating = false
      this.aiProgress = 0
      this.aiProgressStatus = ''
      this.aiProgressText = ''
    },
    
    async startAiEvaluation() {
      if (!this.selectedModelId) {
        this.$message.error('请选择AI模型')
        return
      }
      
      this.aiEvaluating = true
      this.aiProgress = 0
      this.aiProgressStatus = ''
      this.aiProgressText = '正在分析股票数据...'
      
      // 模拟进度更新
      const progressInterval = setInterval(() => {
        if (this.aiProgress < 80) {
          this.aiProgress = Math.min(80, this.aiProgress + Math.floor(Math.random() * 10))
          if (this.aiProgress < 30) {
            this.aiProgressText = '正在获取历史数据...'
          } else if (this.aiProgress < 60) {
            this.aiProgressText = '正在分析技术指标...'
          } else {
            this.aiProgressText = 'AI正在生成分析报告...'
          }
        }
      }, 500)
      
      try {
        const response = await axios.post('/webManager/ai/evaluate/stock', {
          model_id: this.selectedModelId,
          stock_code: this.code,
          stock_name: this.currentName
        })
        
        clearInterval(progressInterval)
        this.aiProgress = 100
        this.aiProgressStatus = 'success'
        this.aiProgressText = '分析完成！'
        
        if (response.data.code === 0) {
          this.aiResult = response.data.data
        } else {
          throw new Error(response.data.msg || 'AI评测失败')
        }
      } catch (error) {
        clearInterval(progressInterval)
        this.aiProgressStatus = 'exception'
        this.aiProgressText = '分析失败'
        console.error('AI评测失败:', error)
        this.$message.error(`AI评测失败: ${error.response?.data?.msg || error.message}`)
      } finally {
        this.aiEvaluating = false
      }
    },
    
    getScoreClass(score) {
      if (score >= 50) return 'score-bullish'
      if (score >= 20) return 'score-slightly-bullish'
      if (score >= -20) return 'score-neutral'
      if (score >= -50) return 'score-slightly-bearish'
      return 'score-bearish'
    },
    
    getScoreLabel(score) {
      if (score >= 0) return '看涨分数'
      return '看跌分数'
    },
    
    getProgressColor(score) {
      if (score >= 50) return '#67C23A'
      if (score >= 20) return '#85CE61'
      if (score >= -20) return '#E6A23C'
      if (score >= -50) return '#F56C6C'
      return '#F56C6C'
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

/* AI评测对话框样式 */
.ai-evaluation-content {
  padding: 20px 0;
}

.ai-evaluation-progress {
  margin-top: 20px;
}

.progress-text {
  text-align: center;
  margin-top: 10px;
  color: #606266;
  font-size: 14px;
}

.ai-evaluation-result {
  padding: 20px 0;
}

/* 综合分数圆圈 */
.overall-score {
  text-align: center;
  margin-bottom: 20px;
}

.score-circle {
  display: inline-block;
  width: 100px;
  height: 100px;
  border-radius: 50%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  margin: 0 auto;
  border: 3px solid;
  transition: all 0.3s ease;
}

.score-circle.score-bullish {
  background: linear-gradient(135deg, #67C23A, #85CE61);
  border-color: #67C23A;
  color: white;
}

.score-circle.score-slightly-bullish {
  background: linear-gradient(135deg, #85CE61, #B3E19D);
  border-color: #85CE61;
  color: white;
}

.score-circle.score-neutral {
  background: linear-gradient(135deg, #E6A23C, #F0C78A);
  border-color: #E6A23C;
  color: white;
}

.score-circle.score-slightly-bearish {
  background: linear-gradient(135deg, #F56C6C, #F89898);
  border-color: #F56C6C;
  color: white;
}

.score-circle.score-bearish {
  background: linear-gradient(135deg, #F56C6C, #FBC4C4);
  border-color: #F56C6C;
  color: white;
}

.score-value {
  font-size: 28px;
  font-weight: bold;
  line-height: 1;
}

.score-label {
  font-size: 12px;
  margin-top: 3px;
  opacity: 0.9;
}

/* 子分数 */
.sub-scores {
  margin-bottom: 20px;
}

.score-item {
  margin-bottom: 15px;
}

.score-name {
  font-size: 13px;
  color: #606266;
  margin-bottom: 6px;
  font-weight: 500;
}

.score-value-text {
  text-align: right;
  margin-top: 3px;
  font-size: 14px;
  font-weight: bold;
}

/* AI总结 */
.ai-summary {
  background: #f8f9fa;
  padding: 15px;
  border-radius: 6px;
  margin-bottom: 15px;
}

.ai-summary h4 {
  margin: 0 0 10px 0;
  color: #303133;
  font-size: 14px;
}

.ai-summary p {
  margin: 0;
  line-height: 1.5;
  color: #606266;
  font-size: 13px;
}

/* 数据信息 */
.data-info {
  background: #f0f9ff;
  padding: 12px;
  border-radius: 6px;
  border-left: 3px solid #3b82f6;
}

.data-info p {
  margin: 0;
  color: #1e40af;
  font-size: 12px;
}

/* 对话框底部按钮 */
.dialog-footer {
  text-align: right;
}

.dialog-footer .el-button {
  margin-left: 10px;
}

/* 桌面端对话框优化 - 内容驱动高度 */
.ai-evaluation-dialog .el-dialog {
  max-height: 90vh !important;
  overflow: hidden !important;
}

.ai-evaluation-dialog .el-dialog__body {
  overflow-y: auto !important;
  padding: 20px !important;
}

/* 移动端对话框样式 */
.ai-evaluation-mobile .el-dialog {
  width: 95% !important;
  margin: 0 auto !important;
  max-width: 420px;
  height: auto !important;
  max-height: calc(100vh - 60px) !important;
  border-radius: 12px;
  overflow: hidden;
  position: fixed !important;
  top: 30px !important;
  left: 50% !important;
  transform: translateX(-50%) !important;
  z-index: 5000 !important;
  display: flex !important;
  flex-direction: column !important;
}

.ai-evaluation-mobile .el-dialog__wrapper {
  background-color: rgba(0, 0, 0, 0.5) !important;
  position: fixed !important;
  top: 0 !important;
  left: 0 !important;
  width: 100% !important;
  height: 100vh !important;
  z-index: 4999 !important;
}

.ai-evaluation-mobile .el-dialog__header {
  padding: 12px 16px 8px;
  background: #f8fafc;
  border-bottom: 1px solid #e2e8f0;
  flex-shrink: 0;
}

.ai-evaluation-mobile .el-dialog__title {
  font-size: 15px;
  font-weight: 600;
  color: #1e293b;
}

.ai-evaluation-mobile .el-dialog__body {
  padding: 12px 16px;
  overflow-y: auto;
  background: #fff;
  flex: 1;
  min-height: 0;
  -webkit-overflow-scrolling: touch;
}

.ai-evaluation-mobile .el-dialog__footer {
  padding: 8px 16px 12px;
  background: #f8fafc;
  border-top: 1px solid #e2e8f0;
  flex-shrink: 0;
  display: flex;
  gap: 8px;
}

.ai-evaluation-mobile .el-dialog__footer .el-button {
  flex: 1;
  padding: 10px 16px;
  border-radius: 6px;
  font-weight: 500;
  z-index: 5001 !important;
  position: relative;
}

.ai-evaluation-mobile .el-form-item {
  margin-bottom: 12px;
}

.ai-evaluation-mobile .el-form-item__label {
  font-size: 13px;
  margin-bottom: 4px;
  line-height: 1.4;
  color: #374151;
}

.ai-evaluation-mobile .el-input,
.ai-evaluation-mobile .el-select,
.ai-evaluation-mobile .el-textarea {
  width: 100%;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .score-circle {
    width: 70px;
    height: 70px;
  }
  
  .score-value {
    font-size: 20px;
  }
  
  .score-label {
    font-size: 9px;
  }
  
  .ai-summary {
    padding: 8px;
    margin: 8px 0;
  }
  
  .ai-summary h4 {
    font-size: 12px;
    margin-bottom: 4px;
  }
  
  .ai-summary p {
    font-size: 11px;
    line-height: 1.4;
    margin: 0;
  }
  
  .score-item {
    margin-bottom: 8px;
  }
  
  .score-name {
    font-size: 11px;
    margin-bottom: 2px;
  }
  
  .score-value-text {
    font-size: 12px;
  }
  
  .data-info {
    padding: 6px;
    margin: 4px 0;
  }
  
  .data-info p {
    font-size: 10px;
    margin: 0;
  }
  
  /* 优化子分数布局，使其更紧凑 */
  .sub-scores {
    margin: 8px 0;
  }
  
  /* 优化整体分数区域 */
  .overall-score {
    margin: 8px 0;
  }
  
  /* 移动端紧凑布局样式 */
  .mobile-compact-layout {
    padding: 0;
  }
  
  .mobile-overall-score {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 12px;
    padding: 8px;
    background: #f8fafc;
    border-radius: 8px;
  }
  
  .mobile-summary-text {
    flex: 1;
    min-width: 0;
  }
  
  .mobile-summary-text h4 {
    font-size: 12px;
    margin: 0 0 4px 0;
    color: #374151;
  }
  
  .mobile-summary-text p {
    font-size: 11px;
    line-height: 1.4;
    margin: 0;
    color: #6b7280;
  }
  
  .mobile-sub-scores {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 8px;
    margin-bottom: 12px;
  }
  
  .mobile-score-item {
    text-align: center;
    padding: 8px 4px;
    background: #f9fafb;
    border-radius: 6px;
    border: 1px solid #e5e7eb;
  }
  
  .mobile-score-item .score-name {
    font-size: 10px;
    color: #6b7280;
    margin-bottom: 4px;
    font-weight: 500;
  }
  
  .mobile-score-item .score-value-text {
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 4px;
    color: #374151;
  }
  
  .mobile-data-info {
    padding: 6px 8px;
    background: #f3f4f6;
    border-radius: 6px;
    text-align: center;
  }
  
  .mobile-data-info p {
    font-size: 10px;
    margin: 0;
    color: #6b7280;
  }
}
</style>
