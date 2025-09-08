<template>
  <router-view />
  </template>

<script>
export default { name: 'App' }
  </script>

<style>
html, body, #app { height: 100%; margin: 0; }
.layout { display: flex; flex-direction: column; height: 100%; }
.layout-header { padding: 8px 12px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; }
.layout-body { display: flex; flex: 1; min-height: 0; }
.layout-sider { width: 220px; border-right: 1px solid #eee; }
.layout-content { flex: 1; padding: 12px; overflow: auto; }
@media (max-width: 768px) { .layout-sider { width: 160px; } }
  </style>


