import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) {
            return undefined;
          }

          if (id.includes('react-dom') || id.includes('react/')) {
            return 'react-vendor';
          }

          if (id.includes('@ant-design/icons')) {
            return 'ant-icons';
          }

          if (
            id.includes('rc-table')
            || id.includes('rc-pagination')
            || id.includes('/antd/es/table/')
            || id.includes('/antd/es/pagination/')
          ) {
            return 'antd-data';
          }

          if (
            id.includes('rc-select')
            || id.includes('rc-tree-select')
            || id.includes('rc-cascader')
            || id.includes('rc-virtual-list')
            || id.includes('/antd/es/select/')
          ) {
            return 'antd-select';
          }

          if (
            id.includes('rc-dialog')
            || id.includes('rc-drawer')
            || id.includes('rc-dropdown')
            || id.includes('rc-menu')
            || id.includes('rc-notification')
            || id.includes('rc-tooltip')
            || id.includes('rc-trigger')
            || id.includes('/antd/es/modal/')
            || id.includes('/antd/es/dropdown/')
            || id.includes('/antd/es/message/')
            || id.includes('/antd/es/notification/')
            || id.includes('/antd/es/tooltip/')
            || id.includes('/antd/es/popover/')
          ) {
            return 'antd-feedback';
          }

          if (
            id.includes('rc-field-form')
            || id.includes('/antd/es/form/')
            || id.includes('/antd/es/input/')
            || id.includes('/antd/es/input-number/')
            || id.includes('/antd/es/switch/')
            || id.includes('/antd/es/slider/')
          ) {
            return 'antd-forms';
          }

          if (id.includes('antd')) {
            return 'antd-core';
          }

          return 'vendor';
        },
      },
    },
  },
});
