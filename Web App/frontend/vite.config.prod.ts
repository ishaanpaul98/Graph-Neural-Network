import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Production configuration for AWS Amplify
export default defineConfig({
  plugins: [react()],
  define: {
    // Expose environment variables to the client
    'process.env': {}
  }
}) 