<template>
  <div v-loading="loading" element-loading-text="执行中...">
    <el-card>
      <div slot="header">日线数据补拉</div>
      <el-row :gutter="12">
        <el-col :xs="24" :sm="14" :md="14">
          <el-form :inline="true" label-width="86px">
            <el-form-item label="模式">
              <el-radio-group v-model="mode">
                <el-radio-button label="range">时间区间补拉</el-radio-button>
                <el-radio-button label="missing">缺失日补拉</el-radio-button>
              </el-radio-group>
            </el-form-item>

            <template v-if="mode==='range'">
              <el-form-item label="codes">
                <el-input v-model="codes" placeholder="留空=全部；或 sh.600000,sz.000001" style="width:320px" />
              </el-form-item>
              <el-form-item label="时间区间">
                <el-date-picker v-model="range" type="daterange" :unlink-panels="true" value-format="yyyy-MM-dd" range-separator="至" start-placeholder="开始日期" end-placeholder="结束日期" :picker-options="pickerOptions" />
              </el-form-item>
            </template>

            <template v-else>
              <el-form-item label="缺失日">
                <el-date-picker v-model="missingDate" type="date" value-format="yyyy-MM-dd" />
              </el-form-item>
            </template>

            <el-form-item>
              <el-button type="primary" :loading="loading" @click="doBackfill">执行</el-button>
            </el-form-item>
          </el-form>
        </el-col>
        <el-col :xs="24" :sm="10" :md="10">
          <div class="help-panel">
            <div class="help-title">如何选择补拉模式</div>
            <ul class="help-list">
              <li><b>时间区间补拉</b>：为选定股票在起止日期内补齐/更新日线</li>
              <li><b>缺失日补拉</b>：仅针对某一天，自动找出该日缺少日线的股票并补齐（忽略 codes）</li>
              <li>二者互斥：填写“缺失日”时，仅执行缺失日补拉</li>
              <li>建议顺序：先缺失日 → 再时间区间</li>
            </ul>
          </div>
        </el-col>
      </el-row>
    </el-card>
  </div>
  </template>

<script>
import axios from 'axios'
import viewportManager from '@/utils/viewportManager'

export default {
  name: 'DailyBackfill',
  data(){ return { mode:'range', codes:'', range:[], missingDate:'', pickerOptions:{}, loading:false, isMobile: false, isPortrait: true } },
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
    this.updateDeviceInfo();
    this.deviceInfoInterval = setInterval(this.updateDeviceInfo, 1000);
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
    doBackfill(){
      this.loading = true
      if(this.mode==='missing'){
        if(!this.missingDate){ this.loading=false; this.$message.error('请选择缺失日'); return }
        axios.post('/webManager/daily/fetch', { start:this.missingDate, end:this.missingDate, fill_missing_for_date: this.missingDate })
          .then(res=>{ if(res.data.code===0) this.$message.success('已执行缺失日补拉'); else this.$message.error(res.data.msg) })
          .catch(()=> this.$message.error('补拉失败'))
          .finally(()=> this.loading=false)
      } else {
        if(!this.range || this.range.length!==2){ this.loading=false; this.$message.error('请选择时间区间'); return }
        const [start, end] = this.range
        const codesPayload = (this.codes && this.codes.trim()) ? this.codes.split(',').map(s=>s.trim()).filter(Boolean) : []
        axios.post('/webManager/daily/fetch', { codes: codesPayload, start, end })
          .then(res=>{ if(res.data.code===0) this.$message.success('已执行时间区间补拉'); else this.$message.error(res.data.msg) })
          .catch(()=> this.$message.error('补拉失败'))
          .finally(()=> this.loading=false)
      }
    },
    updateDeviceInfo(){
      // 从视口管理器获取最新的设备信息
      const viewportInfo = viewportManager.getViewportInfo();
      this.isMobile = viewportInfo.isMobile;
      this.isPortrait = viewportInfo.isPortrait;
    }
  }
}
  </script>

<style scoped>
.help-panel{
  background: #f5f7ff;
  border: 1px solid #e6ebf2;
  border-radius: 8px;
  padding: 12px 14px;
}
.help-title{
  font-size: 14px;
  color: #374151;
  margin-bottom: 6px;
  font-weight: 600;
}
.help-list{
  margin: 0;
  padding-left: 18px;
  color: #4b5563;
  line-height: 1.7;
  font-size: 13px;
}
.help-list li{ margin: 2px 0; }
</style>
