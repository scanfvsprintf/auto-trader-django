<template>
  <div class="ai-config-root" v-loading="loading" element-loading-text="加载中...">
    <!-- 非竖屏移动端：标准布局 -->
    <div v-if="!isPortraitMobile" class="ai-config-desktop">
      <el-card>
        <div slot="header">
          <span>AI模型配置</span>
          <el-button style="float: right; padding: 3px 0" type="text" @click="refreshData">刷新</el-button>
        </div>
        
        <!-- AI源配置 -->
        <el-tabs v-model="activeTab" type="card">
          <el-tab-pane label="AI源配置" name="source">
            <div style="margin-bottom: 20px;">
              <el-button type="primary" @click="showSourceDialog = true">新增AI源</el-button>
            </div>
            
            <el-table :data="sourceList" border style="width: 100%">
              <el-table-column prop="name" label="资源名称" width="150"></el-table-column>
              <el-table-column prop="url" label="资源URL" min-width="200"></el-table-column>
              <el-table-column prop="api_key" label="API密钥" width="120">
                <template slot-scope="scope">
                  {{ scope.row.api_key ? '***' + scope.row.api_key.slice(-4) : '' }}
                </template>
              </el-table-column>
              <el-table-column prop="is_active" label="是否启用" width="100">
                <template slot-scope="scope">
                  <el-tag :type="scope.row.is_active ? 'success' : 'danger'">
                    {{ scope.row.is_active ? '启用' : '禁用' }}
                  </el-tag>
                </template>
              </el-table-column>
              <el-table-column prop="description" label="描述" min-width="150"></el-table-column>
              <el-table-column prop="created_at" label="创建时间" width="150">
                <template slot-scope="scope">
                  {{ formatDate(scope.row.created_at) }}
                </template>
              </el-table-column>
              <el-table-column label="操作" width="200">
                <template slot-scope="scope">
                  <el-button size="mini" @click="editSource(scope.row)">编辑</el-button>
                  <el-button size="mini" type="danger" @click="deleteSource(scope.row)">删除</el-button>
                </template>
              </el-table-column>
            </el-table>
          </el-tab-pane>
          
          <!-- AI模型配置 -->
          <el-tab-pane label="AI模型配置" name="model">
            <div style="margin-bottom: 20px;">
              <el-button type="primary" @click="showModelDialog = true">新增AI模型</el-button>
            </div>
            
            <el-table :data="modelList" border style="width: 100%">
              <el-table-column prop="name" label="模型名称" width="150"></el-table-column>
              <el-table-column prop="model_type" label="模型类型" width="100">
                <template slot-scope="scope">
                  {{ scope.row.model_type == 1 ? '文本模型' : scope.row.model_type == 2 ? '生图模型' : '未知类型' }}
                </template>
              </el-table-column>
              <el-table-column prop="source_name" label="关联源" width="120"></el-table-column>
              <el-table-column prop="model_id" label="模型ID" width="150"></el-table-column>
              <el-table-column prop="max_tokens" label="最大Token" width="100"></el-table-column>
              <el-table-column prop="temperature" label="温度参数" width="100"></el-table-column>
              <el-table-column prop="is_active" label="是否启用" width="100">
                <template slot-scope="scope">
                  <el-tag :type="scope.row.is_active ? 'success' : 'danger'">
                    {{ scope.row.is_active ? '启用' : '禁用' }}
                  </el-tag>
                </template>
              </el-table-column>
              <el-table-column prop="description" label="描述" min-width="150"></el-table-column>
              <el-table-column label="操作" width="250">
                <template slot-scope="scope">
                  <el-button size="mini" @click="testModel(scope.row)">测试连接</el-button>
                  <el-button size="mini" @click="editModel(scope.row)">编辑</el-button>
                  <el-button size="mini" type="danger" @click="deleteModel(scope.row)">删除</el-button>
                </template>
              </el-table-column>
            </el-table>
          </el-tab-pane>
          
          <!-- AI文本生成测试 -->
          <el-tab-pane label="AI测试" name="test">
            <el-form :model="testForm" label-width="100px">
              <el-form-item label="选择模型">
                <el-select v-model="testForm.model_name" placeholder="请选择模型" style="width: 300px;">
                  <el-option
                    v-for="model in modelList"
                    :key="model.name"
                    :label="model.name"
                    :value="model.name">
                  </el-option>
                </el-select>
              </el-form-item>
              <el-form-item label="提示词">
                <el-input
                  v-model="testForm.prompt"
                  type="textarea"
                  :rows="4"
                  placeholder="请输入提示词..."
                  style="width: 500px;">
                </el-input>
              </el-form-item>
              <el-form-item label="温度参数">
                <el-slider
                  v-model="testForm.temperature"
                  :min="0"
                  :max="2"
                  :step="0.1"
                  style="width: 300px;">
                </el-slider>
                <span style="margin-left: 10px;">{{ testForm.temperature }}</span>
              </el-form-item>
              <el-form-item label="最大Token">
                <el-input-number
                  v-model="testForm.max_tokens"
                  :min="1"
                  :max="1000000"
                  style="width: 200px;">
                </el-input-number>
              </el-form-item>
              <el-form-item>
                <el-button type="primary" :loading="testLoading" @click="generateText">生成文本</el-button>
              </el-form-item>
            </el-form>
            
            <el-card v-if="testResult" style="margin-top: 20px;">
              <div slot="header">生成结果</div>
              <div style="white-space: pre-wrap; line-height: 1.6;">{{ testResult }}</div>
            </el-card>
          </el-tab-pane>
        </el-tabs>
      </el-card>
    </div>

    <!-- 竖屏移动端：优化布局 -->
    <div v-else class="ai-config-mobile">
      <!-- 标签页切换 -->
      <div class="mobile-tabs">
        <div 
          v-for="tab in tabs" 
          :key="tab.name"
          class="mobile-tab"
          :class="{ active: activeTab === tab.name }"
          @click="activeTab = tab.name">
          {{ tab.label }}
        </div>
      </div>

      <!-- AI源配置 -->
      <div v-if="activeTab === 'source'" class="mobile-content">
        <div class="mobile-section">
          <div class="mobile-section-header">
            <span>AI源配置</span>
            <el-button size="mini" type="primary" @click="showSourceDialog = true">新增</el-button>
          </div>
          
          <div class="mobile-card-list">
            <el-card v-for="source in sourceList" :key="source.id" class="mobile-card">
              <div class="mobile-card-header">
                <div class="mobile-card-title">{{ source.name }}</div>
                <div class="mobile-card-actions">
                  <el-button size="mini" @click="editSource(source)">编辑</el-button>
                  <el-button size="mini" type="danger" @click="deleteSource(source)">删除</el-button>
                </div>
              </div>
              <div class="mobile-card-content">
                <div class="mobile-card-item">
                  <span class="label">URL:</span>
                  <span class="value">{{ source.url }}</span>
                </div>
                <div class="mobile-card-item">
                  <span class="label">API密钥:</span>
                  <span class="value">{{ source.api_key ? '***' + source.api_key.slice(-4) : '' }}</span>
                </div>
                <div class="mobile-card-item">
                  <span class="label">状态:</span>
                  <el-tag :type="source.is_active ? 'success' : 'danger'" size="mini">
                    {{ source.is_active ? '启用' : '禁用' }}
                  </el-tag>
                </div>
                <div class="mobile-card-item" v-if="source.description">
                  <span class="label">描述:</span>
                  <span class="value">{{ source.description }}</span>
                </div>
                <div class="mobile-card-item">
                  <span class="label">创建时间:</span>
                  <span class="value">{{ formatDate(source.created_at) }}</span>
                </div>
              </div>
            </el-card>
          </div>
        </div>
      </div>

      <!-- AI模型配置 -->
      <div v-if="activeTab === 'model'" class="mobile-content">
        <div class="mobile-section">
          <div class="mobile-section-header">
            <span>AI模型配置</span>
            <el-button size="mini" type="primary" @click="showModelDialog = true">新增</el-button>
          </div>
          
          <div class="mobile-card-list">
            <el-card v-for="model in modelList" :key="model.id" class="mobile-card">
              <div class="mobile-card-header">
                <div class="mobile-card-title">{{ model.name }}</div>
                <div class="mobile-card-actions">
                  <el-button size="mini" @click="testModel(model)">测试</el-button>
                  <el-button size="mini" @click="editModel(model)">编辑</el-button>
                  <el-button size="mini" type="danger" @click="deleteModel(model)">删除</el-button>
                </div>
              </div>
              <div class="mobile-card-content">
                <div class="mobile-card-item">
                  <span class="label">类型:</span>
                  <span class="value">{{ model.model_type == 1 ? '文本模型' : model.model_type == 2 ? '生图模型' : '未知类型' }}</span>
                </div>
                <div class="mobile-card-item">
                  <span class="label">关联源:</span>
                  <span class="value">{{ model.source_name }}</span>
                </div>
                <div class="mobile-card-item">
                  <span class="label">模型ID:</span>
                  <span class="value">{{ model.model_id }}</span>
                </div>
                <div class="mobile-card-item">
                  <span class="label">最大Token:</span>
                  <span class="value">{{ model.max_tokens }}</span>
                </div>
                <div class="mobile-card-item">
                  <span class="label">温度参数:</span>
                  <span class="value">{{ model.temperature }}</span>
                </div>
                <div class="mobile-card-item">
                  <span class="label">状态:</span>
                  <el-tag :type="model.is_active ? 'success' : 'danger'" size="mini">
                    {{ model.is_active ? '启用' : '禁用' }}
                  </el-tag>
                </div>
                <div class="mobile-card-item" v-if="model.description">
                  <span class="label">描述:</span>
                  <span class="value">{{ model.description }}</span>
                </div>
              </div>
            </el-card>
          </div>
        </div>
      </div>

      <!-- AI测试 -->
      <div v-if="activeTab === 'test'" class="mobile-content">
        <div class="mobile-section">
          <div class="mobile-section-header">
            <span>AI测试</span>
          </div>
          
          <el-form :model="testForm" label-position="top" class="mobile-form">
            <el-form-item label="选择模型">
              <el-select v-model="testForm.model_name" placeholder="请选择模型" style="width: 100%;">
                <el-option
                  v-for="model in modelList"
                  :key="model.name"
                  :label="model.name"
                  :value="model.name">
                </el-option>
              </el-select>
            </el-form-item>
            
            <el-form-item label="提示词">
              <el-input
                v-model="testForm.prompt"
                type="textarea"
                :rows="4"
                placeholder="请输入提示词...">
              </el-input>
            </el-form-item>
            
            <el-form-item label="温度参数">
              <div class="mobile-slider-container">
                <el-slider
                  v-model="testForm.temperature"
                  :min="0"
                  :max="2"
                  :step="0.1">
                </el-slider>
                <span class="slider-value">{{ testForm.temperature }}</span>
              </div>
            </el-form-item>
            
            <el-form-item label="最大Token">
              <el-input-number
                v-model="testForm.max_tokens"
                :min="1"
                :max="1000000"
                style="width: 100%;">
              </el-input-number>
            </el-form-item>
            
            <el-form-item>
              <el-button type="primary" :loading="testLoading" @click="generateText" style="width: 100%;">生成文本</el-button>
            </el-form-item>
          </el-form>
          
          <el-card v-if="testResult" class="mobile-result-card">
            <div slot="header">生成结果</div>
            <div class="mobile-result-content">{{ testResult }}</div>
          </el-card>
        </div>
      </div>
    </div>
    
    <!-- AI源配置对话框 -->
    <el-dialog
      :title="sourceDialogTitle"
      :visible.sync="showSourceDialog"
      :width="isPortraitMobile ? '90%' : '600px'"
      :class="{ 'ai-config-mobile': isPortraitMobile }"
      :append-to-body="isPortraitMobile"
      :close-on-click-modal="false"
      :close-on-press-escape="true"
      :destroy-on-close="true">
      <el-form :model="sourceForm" :rules="sourceRules" ref="sourceForm" label-width="100px">
        <el-form-item label="资源名称" prop="name">
          <el-input v-model="sourceForm.name" placeholder="请输入资源名称"></el-input>
        </el-form-item>
        <el-form-item label="资源URL" prop="url">
          <el-input v-model="sourceForm.url" placeholder="请输入API URL"></el-input>
        </el-form-item>
        <el-form-item label="API密钥" prop="api_key">
          <el-input v-model="sourceForm.api_key" type="password" placeholder="请输入API密钥"></el-input>
        </el-form-item>
        <el-form-item label="描述">
          <el-input v-model="sourceForm.description" type="textarea" :rows="3" placeholder="请输入描述"></el-input>
        </el-form-item>
        <el-form-item label="是否启用">
          <el-switch v-model="sourceForm.is_active"></el-switch>
        </el-form-item>
      </el-form>
      <div slot="footer" class="dialog-footer">
        <el-button @click="showSourceDialog = false">取消</el-button>
        <el-button type="primary" :loading="sourceLoading" @click="saveSource">确定</el-button>
      </div>
    </el-dialog>
    
    <!-- AI模型配置对话框 -->
    <el-dialog
      :title="modelDialogTitle"
      :visible.sync="showModelDialog"
      :width="isPortraitMobile ? '90%' : '600px'"
      :class="{ 'ai-config-mobile': isPortraitMobile }"
      :append-to-body="isPortraitMobile"
      :close-on-click-modal="false"
      :close-on-press-escape="true"
      :destroy-on-close="true">
      <el-form :model="modelForm" :rules="modelRules" ref="modelForm" label-width="100px">
        <el-form-item label="模型名称" prop="name">
          <el-input v-model="modelForm.name" placeholder="请输入模型名称"></el-input>
        </el-form-item>
        <el-form-item label="模型类型" prop="model_type">
          <el-select v-model="modelForm.model_type" placeholder="请选择模型类型">
            <el-option label="文本模型" :value="1"></el-option>
            <el-option label="生图模型" :value="2"></el-option>
          </el-select>
        </el-form-item>
        <el-form-item label="关联源" prop="source_id">
          <el-select v-model="modelForm.source_id" placeholder="请选择AI源">
            <el-option
              v-for="source in sourceList"
              :key="source.id"
              :label="source.name"
              :value="source.id">
            </el-option>
          </el-select>
        </el-form-item>
        <el-form-item label="模型ID" prop="model_id">
          <el-input v-model="modelForm.model_id" placeholder="请输入模型ID"></el-input>
        </el-form-item>
        <el-form-item label="最大Token" prop="max_tokens">
          <el-input-number v-model="modelForm.max_tokens" :min="1" :max="1000000"></el-input-number>
        </el-form-item>
        <el-form-item label="温度参数" prop="temperature">
          <el-input-number v-model="modelForm.temperature" :min="0" :max="2" :step="0.1"></el-input-number>
        </el-form-item>
        <el-form-item label="描述">
          <el-input v-model="modelForm.description" type="textarea" :rows="3" placeholder="请输入描述"></el-input>
        </el-form-item>
        <el-form-item label="是否启用">
          <el-switch v-model="modelForm.is_active"></el-switch>
        </el-form-item>
      </el-form>
      <div slot="footer" class="dialog-footer">
        <el-button @click="showModelDialog = false">取消</el-button>
        <el-button type="primary" :loading="modelLoading" @click="saveModel">确定</el-button>
      </div>
    </el-dialog>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'AiConfig',
  data() {
    return {
      loading: false,
      activeTab: 'source',
      isMobile: false,
      isPortrait: true,
      tabs: [
        { name: 'source', label: 'AI源' },
        { name: 'model', label: 'AI模型' },
        { name: 'test', label: 'AI测试' }
      ],
      
      // AI源配置相关
      sourceList: [],
      showSourceDialog: false,
      sourceLoading: false,
      sourceForm: {
        id: null,
        name: '',
        url: '',
        api_key: '',
        description: '',
        is_active: true
      },
      sourceRules: {
        name: [{ required: true, message: '请输入资源名称', trigger: 'blur' }],
        url: [{ required: true, message: '请输入资源URL', trigger: 'blur' }],
        api_key: [{ required: true, message: '请输入API密钥', trigger: 'blur' }]
      },
      
      // AI模型配置相关
      modelList: [],
      showModelDialog: false,
      modelLoading: false,
      modelForm: {
        id: null,
        name: '',
        model_type: 1,
        source_id: null,
        model_id: '',
        max_tokens: 1000,
        temperature: 0.7,
        description: '',
        is_active: true
      },
      modelRules: {
        name: [{ required: true, message: '请输入模型名称', trigger: 'blur' }],
        model_type: [{ required: true, message: '请选择模型类型', trigger: 'change' }],
        source_id: [{ required: true, message: '请选择关联源', trigger: 'change' }],
        model_id: [{ required: true, message: '请输入模型ID', trigger: 'blur' }],
        max_tokens: [{ required: true, message: '请输入最大Token数', trigger: 'blur' }],
        temperature: [{ required: true, message: '请输入温度参数', trigger: 'blur' }]
      },
      
      // AI测试相关
      testForm: {
        model_name: '',
        prompt: '',
        temperature: 0.7,
        max_tokens: 1000
      },
      testLoading: false,
      testResult: ''
    }
  },
  computed: {
    isPortraitMobile() {
      return this.isMobile && this.isPortrait
    },
    sourceDialogTitle() {
      return this.sourceForm.id ? '编辑AI源配置' : '新增AI源配置'
    },
    modelDialogTitle() {
      return this.modelForm.id ? '编辑AI模型配置' : '新增AI模型配置'
    }
  },
  created() {
    this.onResize()
    this.loadData()
    window.addEventListener('resize', this.onResize)
    window.addEventListener('orientationchange', this.onResize)
  },
  mounted() {
    this.adjustDialogHeight()
  },
  beforeDestroy() {
    window.removeEventListener('resize', this.onResize)
    window.removeEventListener('orientationchange', this.onResize)
  },
  methods: {
    onResize() {
      try {
        const w = window.innerWidth || 1024
        const h = window.innerHeight || 768
        this.isMobile = w <= 768
        this.isPortrait = h >= w
        this.adjustDialogHeight()
      } catch (e) {
        this.isMobile = false
        this.isPortrait = true
      }
    },
    
    adjustDialogHeight() {
      this.$nextTick(() => {
        if (this.isPortraitMobile) {
          const dialogs = document.querySelectorAll('.ai-config-mobile .el-dialog')
          dialogs.forEach(dialog => {
            const viewportHeight = window.innerHeight
            // 可用高度 = 整个屏幕高度 - 底部导航栏52px
            const availableHeight = viewportHeight - 52
            dialog.style.maxHeight = `${availableHeight}px`
            
            const body = dialog.querySelector('.el-dialog__body')
            if (body) {
              const headerHeight = dialog.querySelector('.el-dialog__header')?.offsetHeight || 0
              const footerHeight = dialog.querySelector('.el-dialog__footer')?.offsetHeight || 0
              const bodyMaxHeight = availableHeight - headerHeight - footerHeight - 24
              body.style.maxHeight = `${bodyMaxHeight}px`
            }
          })
        }
      })
    },
    async loadData() {
      this.loading = true
      try {
        await Promise.all([
          this.loadSourceList(),
          this.loadModelList()
        ])
      } catch (error) {
        this.$message.error('加载数据失败')
      } finally {
        this.loading = false
      }
    },
    
    async loadSourceList() {
      try {
        const response = await axios.get('/webManager/ai/source/config')
        if (response.data.code === 0) {
          this.sourceList = response.data.data || []
        } else {
          this.$message.error(response.data.msg)
        }
      } catch (error) {
        this.$message.error('加载AI源配置失败')
      }
    },
    
    async loadModelList() {
      try {
        const response = await axios.get('/webManager/ai/model/config')
        if (response.data.code === 0) {
          this.modelList = response.data.data || []
        } else {
          this.$message.error(response.data.msg)
        }
      } catch (error) {
        this.$message.error('加载AI模型配置失败')
      }
    },
    
    refreshData() {
      this.loadData()
    },
    
    // AI源配置相关方法
    editSource(row) {
      this.sourceForm = { 
        ...row,
        is_active: Boolean(row.is_active)  // 确保is_active是布尔类型
      }
      this.showSourceDialog = true
      this.$nextTick(() => {
        this.adjustDialogHeight()
      })
    },
    
    async deleteSource(row) {
      try {
        await this.$confirm('确定要删除这个AI源配置吗？', '确认删除', {
          type: 'warning'
        })
        
        const response = await axios.delete('/webManager/ai/source/config', {
          params: { id: row.id }
        })
        
        if (response.data.code === 0) {
          this.$message.success('删除成功')
          this.loadSourceList()
        } else {
          this.$message.error(response.data.msg)
        }
      } catch (error) {
        if (error !== 'cancel') {
          this.$message.error('删除失败')
        }
      }
    },
    
    async saveSource() {
      try {
        await this.$refs.sourceForm.validate()
        
        this.sourceLoading = true
        const url = this.sourceForm.id ? '/webManager/ai/source/config' : '/webManager/ai/source/config'
        const method = this.sourceForm.id ? 'put' : 'post'
        
        const response = await axios[method](url, this.sourceForm)
        
        if (response.data.code === 0) {
          this.$message.success(this.sourceForm.id ? '更新成功' : '创建成功')
          this.showSourceDialog = false
          this.resetSourceForm()
          this.loadSourceList()
        } else {
          this.$message.error(response.data.msg)
        }
      } catch (error) {
        if (error !== 'cancel') {
          this.$message.error('保存失败')
        }
      } finally {
        this.sourceLoading = false
      }
    },
    
    resetSourceForm() {
      this.sourceForm = {
        id: null,
        name: '',
        url: '',
        api_key: '',
        description: '',
        is_active: true
      }
      this.$nextTick(() => {
        this.$refs.sourceForm && this.$refs.sourceForm.clearValidate()
      })
    },
    
    // AI模型配置相关方法
    editModel(row) {
      this.modelForm = { 
        ...row,
        model_type: parseInt(row.model_type),  // 确保model_type是数字类型
        source_id: parseInt(row.source_id)     // 确保source_id是数字类型
      }
      this.showModelDialog = true
      this.$nextTick(() => {
        this.adjustDialogHeight()
      })
    },
    
    async deleteModel(row) {
      try {
        await this.$confirm('确定要删除这个AI模型配置吗？', '确认删除', {
          type: 'warning'
        })
        
        const response = await axios.delete('/webManager/ai/model/config', {
          params: { id: row.id }
        })
        
        if (response.data.code === 0) {
          this.$message.success('删除成功')
          this.loadModelList()
        } else {
          this.$message.error(response.data.msg)
        }
      } catch (error) {
        if (error !== 'cancel') {
          this.$message.error('删除失败')
        }
      }
    },
    
    async saveModel() {
      try {
        await this.$refs.modelForm.validate()
        
        this.modelLoading = true
        const url = '/webManager/ai/model/config'
        const method = this.modelForm.id ? 'put' : 'post'
        
        const response = await axios[method](url, this.modelForm)
        
        if (response.data.code === 0) {
          this.$message.success(this.modelForm.id ? '更新成功' : '创建成功')
          this.showModelDialog = false
          this.resetModelForm()
          this.loadModelList()
        } else {
          this.$message.error(response.data.msg)
        }
      } catch (error) {
        if (error !== 'cancel') {
          this.$message.error('保存失败')
        }
      } finally {
        this.modelLoading = false
      }
    },
    
    resetModelForm() {
      this.modelForm = {
        id: null,
        name: '',
        model_type: 1,
        source_id: null,
        model_id: '',
        max_tokens: 1000,
        temperature: 0.7,
        description: '',
        is_active: true
      }
      this.$nextTick(() => {
        this.$refs.modelForm && this.$refs.modelForm.clearValidate()
      })
    },
    
    // AI测试相关方法
    async testModel(row) {
      try {
        this.$message.info('正在测试连接...')
        const response = await axios.get('/webManager/ai/test/connection', {
          params: { model_name: row.name }
        })
        
        if (response.data.code === 0) {
          if (response.data.data.connected) {
            this.$message.success('连接测试成功')
          } else {
            this.$message.error('连接测试失败')
          }
        } else {
          this.$message.error(response.data.msg)
        }
      } catch (error) {
        this.$message.error('连接测试失败')
      }
    },
    
    async generateText() {
      if (!this.testForm.model_name) {
        this.$message.error('请选择模型')
        return
      }
      if (!this.testForm.prompt.trim()) {
        this.$message.error('请输入提示词')
        return
      }
      
      this.testLoading = true
      try {
        const response = await axios.post('/webManager/ai/generate/text', this.testForm)
        
        if (response.data.code === 0) {
          this.testResult = response.data.data.result
          this.$message.success('文本生成成功')
        } else {
          this.$message.error(response.data.msg)
        }
      } catch (error) {
        this.$message.error('文本生成失败')
      } finally {
        this.testLoading = false
      }
    },
    
    formatDate(dateString) {
      if (!dateString) return ''
      const date = new Date(dateString)
      return date.toLocaleString('zh-CN')
    }
  }
}
</script>

<style scoped>
.dialog-footer {
  text-align: right;
}

/* 竖屏移动端样式 */
.ai-config-mobile {
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.mobile-tabs {
  display: flex;
  background: #fff;
  border-bottom: 1px solid #e6ebf2;
  flex-shrink: 0;
}

.mobile-tab {
  flex: 1;
  padding: 12px 8px;
  text-align: center;
  font-size: 14px;
  color: #6b7280;
  border-bottom: 2px solid transparent;
  cursor: pointer;
  transition: all 0.2s;
}

.mobile-tab.active {
  color: #2563eb;
  border-bottom-color: #2563eb;
  font-weight: 600;
}

.mobile-content {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  padding-bottom: 80px; /* 为底部导航留出空间 */
}

.mobile-section {
  margin-bottom: 24px;
}

.mobile-section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
  padding: 12px 16px;
  background: #f8fafc;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
  min-height: 48px;
  gap: 8px;
}

.mobile-section-header span {
  font-size: 14px;
  font-weight: 600;
  color: #1F2937;
  flex: 1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-right: 8px;
  min-width: 0;
  line-height: 1.2;
  display: block;
}

.mobile-section-header .el-button {
  padding: 6px 10px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 500;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  transition: all 0.2s;
  flex-shrink: 0;
  min-width: 50px;
  max-width: 80px;
}

.mobile-section-header .el-button:hover {
  transform: translateY(-1px);
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
}

.mobile-card-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.mobile-card {
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.mobile-card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
  min-height: 32px;
}

.mobile-card-title {
  font-size: 15px;
  font-weight: 600;
  color: #1F2937;
  flex: 1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-right: 8px;
}

.mobile-card-actions {
  display: flex;
  gap: 4px;
  flex-shrink: 0;
}

.mobile-card-content {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.mobile-card-item {
  display: flex;
  align-items: flex-start;
  font-size: 13px;
  line-height: 1.4;
  min-height: 20px;
}

.mobile-card-item .label {
  min-width: 70px;
  color: #6b7280;
  font-weight: 500;
  flex-shrink: 0;
  margin-right: 8px;
}

.mobile-card-item .value {
  flex: 1;
  color: #374151;
  word-break: break-word;
  overflow-wrap: break-word;
  hyphens: auto;
}

.mobile-form {
  margin-top: 16px;
}

.mobile-form .el-form-item {
  margin-bottom: 20px;
}

.mobile-form .el-form-item__label {
  font-size: 14px;
  font-weight: 600;
  color: #374151;
  margin-bottom: 8px;
}

.mobile-slider-container {
  display: flex;
  align-items: center;
  gap: 12px;
}

.mobile-slider-container .el-slider {
  flex: 1;
}

.slider-value {
  min-width: 40px;
  text-align: center;
  font-size: 14px;
  color: #2563eb;
  font-weight: 600;
}

.mobile-result-card {
  margin-top: 20px;
  border-radius: 8px;
}

.mobile-result-content {
  white-space: pre-wrap;
  line-height: 1.6;
  font-size: 14px;
  color: #374151;
  max-height: 300px;
  overflow-y: auto;
}

/* 对话框在移动端的优化 */
.ai-config-mobile .el-dialog {
  width: 90% !important;
  margin: 0 auto !important;
  max-width: 400px;
  height: auto !important;
  max-height: calc(100vh - 52px) !important;
  border-radius: 12px;
  overflow: hidden;
  position: fixed !important;
  top: 50% !important;
  left: 50% !important;
  transform: translate(-50%, -50%) !important;
  z-index: 5000 !important;
  display: flex !important;
  flex-direction: column !important;
}

.ai-config-mobile .el-dialog__wrapper {
  background-color: rgba(0, 0, 0, 0.5) !important;
  position: fixed !important;
  top: 0 !important;
  left: 0 !important;
  width: 100% !important;
  height: calc(100vh - 52px) !important;
  z-index: 4999 !important;
}

.ai-config-mobile .el-dialog__header {
  padding: 12px 16px 8px;
  background: #f8fafc;
  border-bottom: 1px solid #e2e8f0;
  flex-shrink: 0;
}

.ai-config-mobile .el-dialog__title {
  font-size: 15px;
  font-weight: 600;
  color: #1e293b;
}

.ai-config-mobile .el-dialog__body {
  padding: 12px 16px;
  max-height: calc(100vh - 132px);
  overflow-y: auto;
  background: #fff;
  flex: 1;
  min-height: 0;
}

.ai-config-mobile .el-dialog__footer {
  padding: 8px 16px 12px;
  background: #f8fafc;
  border-top: 1px solid #e2e8f0;
  flex-shrink: 0;
}

.ai-config-mobile .el-form-item {
  margin-bottom: 12px;
}

.ai-config-mobile .el-form-item__label {
  font-size: 13px;
  margin-bottom: 4px;
  line-height: 1.4;
  color: #374151;
}

.ai-config-mobile .el-input,
.ai-config-mobile .el-select,
.ai-config-mobile .el-textarea {
  width: 100%;
}

.ai-config-mobile .el-input-number {
  width: 100%;
}

.ai-config-mobile .el-button {
  width: 100%;
  margin: 0 0 8px 0;
}

.ai-config-mobile .el-dialog__footer .el-button {
  width: 48%;
  margin: 0 1%;
  padding: 10px 16px;
  border-radius: 6px;
  font-weight: 500;
  z-index: 5001 !important;
  position: relative;
}

/* 确保对话框内容可以滚动 */
.ai-config-mobile .el-dialog__body::-webkit-scrollbar {
  width: 4px;
}

.ai-config-mobile .el-dialog__body::-webkit-scrollbar-track {
  background: #f1f1f1;
}

.ai-config-mobile .el-dialog__body::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 2px;
}

.ai-config-mobile .el-dialog__body::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}

/* 按钮组优化 */
.ai-config-mobile .mobile-card-actions .el-button {
  padding: 4px 8px;
  font-size: 12px;
}

/* 滚动条优化 */
.mobile-content::-webkit-scrollbar {
  width: 4px;
}

.mobile-content::-webkit-scrollbar-track {
  background: #f1f1f1;
}

.mobile-content::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 2px;
}

.mobile-content::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}
</style>
