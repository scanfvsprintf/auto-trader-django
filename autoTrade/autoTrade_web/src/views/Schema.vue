<template>
  <div>
    <el-card>
      <div slot="header">Schema 管理</div>
      <el-form inline>
        <el-form-item label="选择 schema">
          <el-select v-model="schema" filterable placeholder="请选择或搜索" style="min-width:300px" @visible-change="onSchemaDropdownVisible">
            <el-option v-for="s in schemaList" :key="s" :label="s" :value="s"></el-option>
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="danger" :disabled="readOnly || !schema" @click="confirmDelete">删除</el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
  </template>

<script>
import axios from 'axios'
export default {
  name: 'Schema',
  data(){ return { schema: '', schemaList: [], readOnly: false } },
  created(){ this.readOnly = !!window.__READ_ONLY__; this.loadSchemas() },
  methods: {
    loadSchemas(){ axios.get('/webManager/system/schema').then(res=>{ if(res.data.code===0) this.schemaList = res.data.data || [] }) },
    onSchemaDropdownVisible(v){ if(v){ this.loadSchemas() } },
    confirmDelete(){
      if(!this.schema){ this.$message.error('请选择 schema'); return }
      this.$confirm(`确认删除 schema ${this.schema} ？此操作不可恢复！`, '提示', { type: 'warning', confirmButtonText:'确认删除', cancelButtonText:'取消' })
        .then(()=> this.deleteSchema())
        .catch(()=>{})
    },
    deleteSchema(){ if(!this.schema){this.$message.error('请选择 schema');return} axios.delete('/webManager/system/schema', { params: { name: this.schema } }).then(res=>{ if(res.data.code===0){ this.$message.success('删除成功'); this.loadSchemas(); this.schema=''; } else this.$message.error(res.data.msg) }) }
  }
}
  </script>

<style scoped>
</style>


