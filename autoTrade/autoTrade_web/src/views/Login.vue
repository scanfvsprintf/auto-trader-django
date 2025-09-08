<template>
  <div style="height:100%;display:flex;align-items:center;justify-content:center;">
    <el-card style="width:320px">
      <div slot="header">登录</div>
      <el-form label-width="60px">
        <el-form-item label="用户名">
          <el-input v-model="username"></el-input>
        </el-form-item>
        <el-form-item label="密码">
          <el-input v-model="password" type="password"></el-input>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="login">登录</el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
  </template>

<script>
export default {
  name: 'Login',
  data(){ return { username: '', password: '' } },
  methods: {
    login(){
      if ((this.username === 'xiangyangxin' || this.username === 'xiangmei') && this.password === '123456') {
        localStorage.setItem('authedUser', this.username)
        this.$message.success('登录成功')
        this.$router.replace('/')
      } else {
        this.$message.error('用户名或密码错误')
      }
    }
  }
}
  </script>


