/**
 * Main App component for the Media Semantic Search application.
 */


import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Container, Box, AppBar, Toolbar, Typography } from '@mui/material';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useEffect } from 'react';
import SearchBar from './components/SearchBar';
import MediaGallery from './components/MediaGallery';
import SettingsPanel from './components/SettingsPanel';
import StatusBar from './components/StatusBar';
import TaskProgressDashboard from './components/TaskProgressDashboard';
import { useSearchStore } from './store/searchStore';

// Create Material-UI theme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
      light: '#42a5f5',
      dark: '#1565c0',
    },
    secondary: {
      main: '#dc004e',
      light: '#ff5983',
      dark: '#9a0036',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
  },
  typography: {
    h4: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 500,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 500,
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 8,
          },
        },
      },
    },
  },
});

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

function App() {
  const { searchResults, isLoading, isLoadingAll, error, loadAllMedia } = useSearchStore();

  // Load all media on app initialization
  useEffect(() => {
    loadAllMedia();
  }, [loadAllMedia]);

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        
        {/* App Bar */}
        <AppBar position="static" elevation={1}>
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              Media Semantic Search
            </Typography>
            <SettingsPanel />
          </Toolbar>
        </AppBar>

        {/* Main Content */}
        <Container maxWidth="xl" sx={{ py: 3 }}>
          <Box sx={{ mb: 3 }}>
            <SearchBar />
          </Box>

          {/* Error Display */}
          {error && (
            <Box sx={{ mb: 2 }}>
              <Typography color="error" variant="body2">
                {error}
              </Typography>
            </Box>
          )}

          {/* Results Gallery */}
          <MediaGallery
            results={searchResults}
            loading={isLoading || isLoadingAll}
          />

          {/* Status Bar */}
          <StatusBar />
        </Container>

        {/* Task Progress Dashboard */}
        <TaskProgressDashboard />
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
