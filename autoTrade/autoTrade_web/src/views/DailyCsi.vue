<template>
  <div v-loading="loading" element-loading-text="加载中...">
    <el-card>
      <div slot="header">沪深300指数</div>
      <el-form inline>
        <el-form-item label="时间区间">
          <el-date-picker
            v-model="csiRange"
            type="daterange"
            :unlink-panels="true"
            value-format="yyyy-MM-dd"
            range-separator="至"
            start-placeholder="开始日期"
            end-placeholder="结束日期"
            :picker-options="pickerOptions"
          />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :loading="loading" @click="fetchCSI">查询</el-button>
          <el-button :loading="loadingRemote" @click="onConfirmFetch">补获取</el-button>
          <el-button type="success" :loading="aiEvaluating" @click="showAiEvaluationDialog">
            <i class="el-icon-cpu" style="margin-right:4px"></i>AI评测
          </el-button>
        </el-form-item>
      </el-form>
      <div ref="csiChart" style="width:100%;height:420px"></div>
    </el-card>
    
    <!-- AI评测对话框 -->
    <el-dialog
      title="AI评测沪深300走势"
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
import smartViewportManager from '@/utils/smartViewportManager'

export default {
  name: 'DailyCsi',
  data(){ 
    return { 
      csiRange: [], 
      csiData: [], 
      pickerOptions: {}, 
      loading:false, 
      loadingRemote:false, 
      isMobile:false,
      isPortrait: true,
      // AI评测相关
      showAiDialog: false,
      selectedModelId: null,
      availableModels: [],
      aiEvaluating: false,
      aiProgress: 0,
      aiProgressStatus: '',
      aiProgressText: '',
      aiResult: null
    } 
  },
  created(){
    const end = new Date(); const start = new Date(); start.setMonth(start.getMonth()-1)
    const fmt = d => `${d.getFullYear()}-${('0'+(d.getMonth()+1)).slice(-2)}-${('0'+d.getDate()).slice(-2)}`
    this.csiRange = [fmt(start), fmt(end)]
    this.pickerOptions = {
      shortcuts: [
        { text:'最近一周', onClick: p=>{ const e=new Date(); const s=new Date(); s.setDate(s.getDate()-7); p.$emit('pick',[fmt(s),fmt(e)]) }},
        { text:'最近一月', onClick: p=>{ const e=new Date(); const s=new Date(); s.setMonth(s.getMonth()-1); p.$emit('pick',[fmt(s),fmt(e)]) }},
        { text:'最近半年', onClick: p=>{ const e=new Date(); const s=new Date(); s.setMonth(s.getMonth()-6); p.$emit('pick',[fmt(s),fmt(e)]) }},
        { text:'最近一年', onClick: p=>{ const e=new Date(); const s=new Date(); s.setFullYear(s.getFullYear()-1); p.$emit('pick',[fmt(s),fmt(e)]) }}
      ]
    }
    this.$nextTick(this.fetchCSI)
    this.updateDeviceInfo();
    this.deviceInfoInterval = setInterval(this.updateDeviceInfo, 1000);
    this.loadAvailableModels()
  },
  beforeDestroy(){ 
    if (this.deviceInfoInterval) {
      clearInterval(this.deviceInfoInterval);
    }
  },
  computed: {
    isPortraitMobile(){ return this.isMobile && this.isPortrait }
  },
  methods: {
    
    fetchCSI(){
      if (!this.csiRange || this.csiRange.length!==2) { this.$message.error('请选择时间区间'); return }
      const [start, end] = this.csiRange
      this.loading = true
      axios.get('/webManager/daily/csi300', { params: { start, end, with_m: 1 } })
        .then(res=>{ if(res.data.code===0){ this.csiData=res.data.data||[]; this.drawCSI() } else this.$message.error(res.data.msg) })
        .catch(()=> this.$message.error('查询失败'))
        .finally(()=> this.loading=false)
    },
    onConfirmFetch(){
      if (!this.csiRange || this.csiRange.length!==2) { this.$message.error('请选择时间区间'); return }
      const [s,e] = this.csiRange
      this.$confirm(`将从远程数据源补获取 ${s} ~ ${e} 的沪深300数据，可能刷新/覆盖本地记录。\n该操作对库有写入，请谨慎执行。是否继续？`, '二次确认', {
        type:'warning', confirmButtonText:'继续', cancelButtonText:'取消', dangerouslyUseHTMLString:false
      }).then(()=>{ this.fetchCSIRemote() }).catch(()=>{})
    },
    fetchCSIRemote(){
      if (!this.csiRange || this.csiRange.length!==2) { this.$message.error('请选择时间区间'); return }
      const [start, end] = this.csiRange
      this.loadingRemote = true
      axios.post('/webManager/daily/csi300/fetch', { start, end })
        .then(res=>{ if(res.data.code===0) this.$message.success('补获取成功'); else this.$message.error(res.data.msg) })
        .catch(()=> this.$message.error('补获取失败'))
        .finally(()=> this.loadingRemote=false)
    },
    drawCSI(){
      const el = this.$refs.csiChart; if(!el) return
      const chart = echarts.init(el, null, { renderer: 'canvas', devicePixelRatio: (window.devicePixelRatio || 1) })
      const kdata = this.csiData.map(x=>[x.trade_date, x.open,x.close,x.low,x.high])
      const m = this.csiData.map(x=>[x.trade_date, x.m_value])
      chart.setOption({
        textStyle:{
          color:'#374151',
          fontSize:12,
          fontFamily:'-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif'
        },
        tooltip:{ trigger:'axis' },
        grid:{ left: this.isMobile? 36 : '6%', right: this.isMobile? 36 : '6%', top: this.isMobile? 16: '8%', bottom: this.isMobile? 36 : '8%', containLabel:true },
        xAxis:{ type:'time', axisLine:{ onZero:false }, boundaryGap:false, axisLabel:{ color:'#6b7280', hideOverlap:true, rotate: this.isMobile? 30:0, fontSize: this.isMobile? 10:12 } },
        yAxis:[
          { type:'value', name:'价格', scale:true, axisLabel:{ color:'#6b7280', margin: this.isMobile? 6:8 } },
          { type:'value', name:'M值', min:-1, max:1, axisLabel:{ formatter:v=>Number(v).toFixed(1), color:'#6b7280', margin: this.isMobile? 6:8 } }
        ],
        dataZoom:[ { type:'inside', xAxisIndex:[0,0] }, { type:'slider', xAxisIndex:[0,0], height:14, bottom:6 } ],
        series:[
          { type:'candlestick', data:kdata, name:'CSI300', yAxisIndex:0, encode:{ x:0, y:[1,2,3,4] }, itemStyle:{ color:'#ec0000', color0:'#00da3c' } },
          { type:'line', data:m, name:'M值', yAxisIndex:1, smooth:true, encode:{ x:0, y:1 }, showSymbol:false }
        ]
      })
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
      this.aiProgressText = '正在查询历史数据...'
      
      try {
        // 模拟进度更新
        const progressInterval = setInterval(() => {
          if (this.aiProgress < 80) {
            this.aiProgress += 10
            if (this.aiProgress <= 30) {
              this.aiProgressText = '正在查询历史数据...'
            } else if (this.aiProgress <= 60) {
              this.aiProgressText = '正在调用AI模型分析...'
            } else {
              this.aiProgressText = '正在生成分析结果...'
            }
          }
        }, 500)
        
        const response = await axios.post('/webManager/ai/evaluate/csi300', {
          model_id: this.selectedModelId
        })
        
        clearInterval(progressInterval)
        
        if (response.data.code === 0) {
          this.aiProgress = 100
          this.aiProgressStatus = 'success'
          this.aiProgressText = '分析完成！'
          this.aiResult = response.data.data
          
          setTimeout(() => {
            this.aiEvaluating = false
          }, 1000)
        } else {
          throw new Error(response.data.msg || 'AI评测失败')
        }
      } catch (error) {
        clearInterval(progressInterval)
        this.aiEvaluating = false
        this.aiProgress = 0
        this.aiProgressStatus = 'exception'
        this.aiProgressText = '分析失败'
        this.$message.error(error.message || 'AI评测失败，请稍后重试')
      }
    },
    
    getScoreClass(score) {
      if (score >= 60) return 'score-bullish'
      if (score >= 20) return 'score-slightly-bullish'
      if (score >= -20) return 'score-neutral'
      if (score >= -60) return 'score-slightly-bearish'
      return 'score-bearish'
    },
    
    getScoreLabel(score) {
      if (score >= 0) return '看涨分数'
      return '看跌分数'
    },
    
    getProgressColor(score) {
      if (score >= 60) return '#67C23A'  // 绿色 - 强烈看涨
      if (score >= 20) return '#85CE61'  // 浅绿色 - 看涨
      if (score >= -20) return '#E6A23C' // 橙色 - 中性
      if (score >= -60) return '#F56C6C' // 浅红色 - 看跌
      return '#F56C6C' // 红色 - 强烈看跌
    },
    updateDeviceInfo(){
      // 从视口管理器获取最新的设备信息
      const viewportInfo = smartViewportManager.getViewportInfo();
      this.isMobile = viewportInfo.isMobile;
      this.isPortrait = viewportInfo.isPortrait;
    }
  }
}
  </script>

<style scoped>
/* AI评测相关样式 */
.ai-evaluation-content {
  padding: 20px 0;
}

.ai-evaluation-progress {
  margin-top: 20px;
  text-align: center;
}

.progress-text {
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

