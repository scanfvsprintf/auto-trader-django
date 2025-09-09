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
        </el-form-item>
      </el-form>
      <div ref="csiChart" style="width:100%;height:420px"></div>
    </el-card>
  </div>
  </template>

<script>
import axios from 'axios'
import * as echarts from 'echarts'
export default {
  name: 'DailyCsi',
  data(){ return { csiRange: [], csiData: [], pickerOptions: {}, loading:false, loadingRemote:false, isMobile:false } },
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
    try{ const w=window.innerWidth||1024; this.isMobile = w <= 768 }catch(e){ this.isMobile=false }
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
    }
  }
}
  </script>


