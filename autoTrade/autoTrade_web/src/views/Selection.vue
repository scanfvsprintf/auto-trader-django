<template>
  <div v-loading="loading" element-loading-text="加载中...">
    <el-form inline>
      <el-form-item label="日期">
        <el-date-picker v-model="queryDate" type="date" value-format="yyyy-MM-dd" placeholder="选择日期"></el-date-picker>
      </el-form-item>
      <el-form-item>
        <el-button type="primary" :loading="loading" @click="fetchPlans">查询</el-button>
      </el-form-item>
      <el-form-item>
        <el-button @click="openRun">手动发起选股</el-button>
      </el-form-item>
    </el-form>

    <el-table :data="rows" border style="width:100%" @row-click="showFactors">
      <el-table-column prop="rank" label="排名" width="80" />
      <el-table-column prop="stock_code" label="股票代码" width="140" />
      <el-table-column prop="stock_name" label="股票名称" width="140" />
      <el-table-column prop="miop" label="低开" />
      <el-table-column prop="maop" label="高开" />
      <el-table-column prop="final_score" label="得分" />
      <el-table-column label="操作" fixed="right" width="120">
        <template slot-scope="scope">
          <el-button type="primary" size="mini" @click.stop="goKline(scope.row)">查看K线</el-button>
        </template>
      </el-table-column>
    </el-table>

    <el-drawer :visible.sync="factorVisible" title="因子值" size="40%">
      <div style="padding:10px" v-loading="loadingFactors" element-loading-text="加载因子中...">
        <div style="margin-bottom:8px">{{ factorTitle }}</div>
        <el-table :data="factors" size="mini" border>
          <el-table-column prop="factor_code" label="因子代码" width="180" />
          <el-table-column prop="factor_name" label="因子名称" />
          <el-table-column prop="raw_value" label="原始值" width="120" />
          <el-table-column prop="norm_score" label="标准分" width="120" />
        </el-table>
      </div>
    </el-drawer>

    <el-dialog :visible.sync="runVisible" title="手动发起一天选股">
      <el-form label-width="100px">
        <el-form-item label="日期">
          <el-date-picker v-model="runDate" type="date" value-format="yyyy-MM-dd" placeholder="选择日期"></el-date-picker>
        </el-form-item>
        <el-form-item label="是否补推邮件">
          <el-switch v-model="sendMail" />
        </el-form-item>
      </el-form>
      <span slot="footer">
        <el-button @click="runVisible=false">取消</el-button>
        <el-button type="primary" :loading="runLoading" @click="doRun">确定</el-button>
      </span>
    </el-dialog>
  </div>
  </template>

<script>
import axios from 'axios'
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
      runVisible: false,
      runDate: '',
      sendMail: false,
      runLoading: false
    }
  },
  created(){
    // 默认日期设为 2025-09-04，并自动加载
    this.queryDate = '2025-09-04'
    this.$nextTick(this.fetchPlans)
  },
  methods: {
    fetchPlans(){
      if (!this.queryDate) { this.$message.error('请选择日期'); return }
      this.loading = true
      axios.get('/webManager/selection/plans', { params: { date: this.queryDate } })
        .then(res => { if (res.data.code===0) this.rows = res.data.data || []; else this.$message.error(res.data.msg) })
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
    openRun(){ this.runVisible = true },
    doRun(){
      if (!this.runDate) { this.$message.error('请选择日期'); return }
      this.runLoading = true
      axios.post('/webManager/selection/run', { date: this.runDate, send_mail: this.sendMail })
        .then(res => { if (res.data.code===0) this.$message.success('发起成功'); else this.$message.error(res.data.msg) })
        .catch(()=> this.$message.error('发起失败'))
        .finally(()=> { this.runLoading=false; this.runVisible=false })
    },
    goKline(row){
      if(!row || !row.stock_code){ this.$message.error('无效股票'); return }
      this.$router.push({ path: '/daily/stock', query: { code: row.stock_code, name: row.stock_name||'', from: 'selection' } })
    }
  }
}
  </script>

<style scoped>
/* 简单响应式：小屏隐藏部分列可在后续补充 */
</style>


