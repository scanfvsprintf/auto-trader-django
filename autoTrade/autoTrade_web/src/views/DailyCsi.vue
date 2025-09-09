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
          <el-button :loading="loadingRemote" @click="fetchCSIRemote">补获取</el-button>
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
  data(){ return { csiRange: [], csiData: [], pickerOptions: {}, loading:false, loadingRemote:false } },
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
      const x = this.csiData.map(x=>x.trade_date)
      const kdata = this.csiData.map(x=>[x.open,x.close,x.low,x.high])
      const m = this.csiData.map(x=>x.m_value)
      chart.setOption({
        textStyle:{
          color:'#374151',
          fontSize:12,
          fontFamily:'-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif'
        },
        tooltip:{ trigger:'axis' },
        grid:{ left:'3%', right:'3%', top:'8%', bottom:'8%' },
        xAxis:{ type:'category', data:x, axisLine:{ onZero:false }, axisLabel:{ color:'#6b7280' } },
        yAxis:[
          { type:'value', name:'价格', scale:true, axisLabel:{ color:'#6b7280' } },
          { type:'value', name:'M值', min:-1, max:1, axisLabel:{ formatter:v=>Number(v).toFixed(1), color:'#6b7280' } }
        ],
        series:[
          { type:'candlestick', data:kdata, name:'CSI300', yAxisIndex:0, itemStyle:{ color:'#ec0000', color0:'#00da3c' } },
          { type:'line', data:m, name:'M值', yAxisIndex:1, smooth:true }
        ]
      })
    }
  }
}
  </script>


