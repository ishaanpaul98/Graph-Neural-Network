import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'https://api.ishaanpaul.com',
        changeOrigin: true
      },
      '/auth': {
        target: 'https://api.ishaanpaul.com',
        changeOrigin: true
      }
    }
  },
  define: {
    // Expose environment variables to the client
    'process.env': {}
  }
})
