import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { Container, CssBaseline, ThemeProvider, createTheme, Box, Tabs, Tab } from '@mui/material'
import MovieForm from './components/MovieForm'
import TraktMovieForm from './components/TraktMovieForm'
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
  const [activeTab, setActiveTab] = useState(0)

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue)
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={activeTab} onChange={handleTabChange} aria-label="recommendation tabs">
          <Tab label="Trakt Integration" />
          <Tab label="Original Dataset" />
        </Tabs>
      </Box>
      
      {activeTab === 0 && (
        <TraktMovieForm onRecommendations={setRecommendations} />
      )}
      
      {activeTab === 1 && (
        <MovieForm onRecommendations={setRecommendations} />
      )}
      
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
