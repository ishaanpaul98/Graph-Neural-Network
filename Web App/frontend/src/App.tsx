import { useState } from 'react'
import { Container, CssBaseline, ThemeProvider, createTheme } from '@mui/material'
import MovieForm from './components/MovieForm'
import Recommendations from './components/Recommendations'

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

function App() {
  const [recommendations, setRecommendations] = useState<string[]>([])

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <MovieForm onRecommendations={setRecommendations} />
        <Recommendations recommendations={recommendations} />
      </Container>
    </ThemeProvider>
  )
}

export default App
