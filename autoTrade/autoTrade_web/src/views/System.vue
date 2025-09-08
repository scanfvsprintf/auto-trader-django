<template>
  <div>
    <el-card style="margin-bottom:12px">
      <div slot="header">schema 管理</div>
      <el-form inline>
        <el-form-item label="选择 schema">
          <el-select v-model="schema" filterable placeholder="请选择或搜索" style="min-width:300px" @visible-change="(v)=>{ if(v) loadSchemas() }">
            <el-option v-for="s in schemaList" :key="s" :label="s" :value="s" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="danger" :disabled="readOnly || !schema" @click="confirmDelete">删除</el-button>
        </el-form-item>
      </el-form>
    </el-card>
    <el-card>
      <div slot="header">回测结果查询（示例）</div>
      <el-form inline>
        <el-form-item label="schema"><el-input v-model="qSchema" placeholder="测试schema"></el-input></el-form-item>
        <el-form-item><el-button type="primary" @click="query">查询</el-button></el-form-item>
      </el-form>
      <el-table :data="rows" border size="mini" style="margin-top:8px">
        <el-table-column prop="schema" label="schema" width="200" />
        <el-table-column prop="mode" label="模式" width="160" />
      </el-table>
    </el-card>
  </div>
  </template>

<script>
import axios from 'axios'
export default {
  name: 'System',
  data(){ return { schema: '', schemaList: [], qSchema: '', rows: [], readOnly: false }},
  created(){ this.readOnly = !!window.__READ_ONLY__; this.loadSchemas() },
  methods: {
    loadSchemas(){ axios.get('/webManager/system/schema').then(res=>{ if(res.data.code===0) this.schemaList = res.data.data || [] }) },
    confirmDelete(){
      if(!this.schema){ this.$message.error('请选择 schema'); return }
      this.$confirm(`确认删除 schema ${this.schema} ？此操作不可恢复！`, '提示', { type: 'warning', confirmButtonText:'确认删除', cancelButtonText:'取消' })
        .then(()=> this.deleteSchema())
        .catch(()=>{})
    },
    deleteSchema(){ if(!this.schema){this.$message.error('请选择 schema');return} axios.delete('/webManager/system/schema', { params: { name: this.schema } }).then(res=>{ if(res.data.code===0){ this.$message.success('删除成功'); this.loadSchemas(); this.schema=''; } else this.$message.error(res.data.msg) }) },
    query(){ if(!this.qSchema){this.$message.error('请输入schema');return} axios.get('/webManager/system/backtest/results', { params: { schema: this.qSchema } }).then(res=>{ if(res.data.code===0) this.rows=[res.data.data]; else this.$message.error(res.data.msg) }) }
  }
}
  </script>


