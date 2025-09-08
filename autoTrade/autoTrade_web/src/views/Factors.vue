<template>
  <div>
    <el-row :gutter="12">
      <el-col :xs="24" :sm="12">
        <el-card>
          <div slot="header">因子参数</div>
          <el-button type="primary" size="mini" :disabled="readOnly" @click="saveParam">保存</el-button>
          <el-table :data="params" size="mini" border style="margin-top:8px">
            <el-table-column prop="param_name" label="名称" width="160" />
            <el-table-column prop="param_value" label="值" />
            <el-table-column prop="group_name" label="分组" width="120" />
            <el-table-column prop="description" label="说明" />
          </el-table>
        </el-card>
      </el-col>
      <el-col :xs="24" :sm="12">
        <el-card>
          <div slot="header">因子定义</div>
          <el-button type="primary" size="mini" :disabled="readOnly" @click="saveDef">保存</el-button>
          <el-table :data="defs" size="mini" border style="margin-top:8px">
            <el-table-column prop="factor_code" label="代码" width="160" />
            <el-table-column prop="factor_name" label="名称" />
            <el-table-column prop="direction" label="方向" width="100" />
            <el-table-column prop="is_active" label="启用" width="80" />
            <el-table-column prop="description" label="说明" />
          </el-table>
        </el-card>
      </el-col>
    </el-row>
  </div>
  </template>

<script>
import axios from 'axios'
export default {
  name: 'Factors',
  data(){ return { params: [], defs: [], readOnly: false }},
  created(){ this.load() },
  methods: {
    load(){
      this.readOnly = !!window.__READ_ONLY__
      axios.get('/webManager/factors/params').then(res=>{ if(res.data.code===0) this.params=res.data.data||[] })
      axios.get('/webManager/factors/definitions').then(res=>{ if(res.data.code===0) this.defs=res.data.data||[] })
    },
    saveParam(){ this.$message.info('请在后续版本提供编辑表单，这里仅展示读取。') },
    saveDef(){ this.$message.info('请在后续版本提供编辑表单，这里仅展示读取。') }
  }
}
  </script>


