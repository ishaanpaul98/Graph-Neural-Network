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
      default: 'transparent',
      paper: 'rgba(255, 255, 255, 0.9)',
    },
    text: {
      primary: '#1a1a1a',
      secondary: '#666666',
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          backdropFilter: 'blur(10px)',
          backgroundColor: 'rgba(255, 255, 255, 0.9)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backdropFilter: 'blur(10px)',
          backgroundColor: 'rgba(255, 255, 255, 0.9)',
        },
      },
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
