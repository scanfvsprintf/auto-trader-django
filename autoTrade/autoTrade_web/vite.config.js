const vue = require('@vitejs/plugin-vue2');

module.exports = {
  plugins: [vue()],
  server: {
    port: 80,
    host: true,
    proxy: {
      '/webManager': {
        target: 'http://localhost:8081',
        changeOrigin: true
      }
    }
  },
  build: {
    outDir: 'dist'
  }
};


