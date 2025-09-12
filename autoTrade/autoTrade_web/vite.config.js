const vue = require('@vitejs/plugin-vue2');
const path = require('path');

module.exports = {
  plugins: [vue()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src')
    }
  },
  server: {
    port: 80,
    host: true,
    strictPort: true,
    proxy: {
      '/webManager': {
        target: process.env.VITE_API_TARGET || 'http://127.0.0.1:8081',
        changeOrigin: true,
        ws: false,
        secure: false,
        // 不改写路径，后端已挂载 /webManager
      }
    }
  },
  build: {
    outDir: 'dist'
  }
};


