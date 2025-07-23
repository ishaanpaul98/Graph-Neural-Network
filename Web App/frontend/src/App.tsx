import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { Container, CssBaseline, ThemeProvider, createTheme } from '@mui/material'
import UnifiedMovieForm from './components/UnifiedMovieForm'
import Recommendations from './components/Recommendations'
import AuthSuccess from './components/AuthSuccess'

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    background: {
      default: '#f5f5f5',
    },
  },
})

function MainApp() {
  const [recommendations, setRecommendations] = useState<string[]>([])

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <UnifiedMovieForm onRecommendations={setRecommendations} />
      <Recommendations recommendations={recommendations} />
    </Container>
  )
}

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Routes>
          <Route path="/" element={<MainApp />} />
          <Route path="/auth-success" element={<AuthSuccess />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Router>
    </ThemeProvider>
  )
}

export default App
