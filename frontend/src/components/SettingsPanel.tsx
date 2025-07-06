/**
 * Settings panel component for application configuration.
 */

import React, { useState } from 'react';
import {
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Tabs,
  Tab,
  TextField,
  Switch,
  FormControlLabel,
  Divider,
  Alert,
  CircularProgress,
  Chip,
  Stack,
} from '@mui/material';
import {
  Settings as SettingsIcon,
  Folder as FolderIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { indexingAPI, healthAPI } from '../services/api';
import { defaultIndexingRequest } from '../types/indexing';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`settings-tabpanel-${index}`}
      aria-labelledby={`settings-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const SettingsPanel: React.FC = () => {
  const [open, setOpen] = useState(false);
  const [tabValue, setTabValue] = useState(0);
  const [indexingPaths, setIndexingPaths] = useState<string[]>([]);
  const [newPath, setNewPath] = useState('');

  const queryClient = useQueryClient();

  // Health check query
  const { data: healthData, isLoading: healthLoading } = useQuery({
    queryKey: ['health'],
    queryFn: healthAPI.checkHealth,
    enabled: open,
    refetchInterval: 5000, // Refresh every 5 seconds when panel is open
  });

  // Indexing status query
  const { data: indexingStatus, isLoading: indexingLoading } = useQuery({
    queryKey: ['indexing-status'],
    queryFn: indexingAPI.getStatus,
    enabled: open,
    refetchInterval: 2000, // Refresh every 2 seconds when panel is open
  });

  // Start indexing mutation
  const startIndexingMutation = useMutation({
    mutationFn: indexingAPI.startIndexing,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['indexing-status'] });
    },
  });

  // Control indexing mutation
  const controlIndexingMutation = useMutation({
    mutationFn: ({ action, jobId }: { action: string; jobId?: string }) =>
      indexingAPI.controlIndexing(action, jobId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['indexing-status'] });
    },
  });

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleAddPath = () => {
    if (newPath.trim() && !indexingPaths.includes(newPath.trim())) {
      setIndexingPaths([...indexingPaths, newPath.trim()]);
      setNewPath('');
    }
  };

  const handleRemovePath = (pathToRemove: string) => {
    setIndexingPaths(indexingPaths.filter(path => path !== pathToRemove));
  };

  const handleStartIndexing = () => {
    if (indexingPaths.length === 0) {
      return;
    }

    const request = {
      ...defaultIndexingRequest,
      paths: indexingPaths,
    };

    startIndexingMutation.mutate(request);
  };

  const handleControlIndexing = (action: string) => {
    controlIndexingMutation.mutate({
      action,
      jobId: indexingStatus?.job_id,
    });
  };

  const getHealthStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'success';
      case 'degraded':
        return 'warning';
      case 'unhealthy':
        return 'error';
      default:
        return 'info';
    }
  };

  return (
    <>
      <IconButton color="inherit" onClick={() => setOpen(true)}>
        <SettingsIcon />
      </IconButton>

      <Dialog open={open} onClose={() => setOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Settings</DialogTitle>
        
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={handleTabChange}>
            <Tab label="Indexing" />
            <Tab label="System Status" />
            <Tab label="About" />
          </Tabs>
        </Box>

        <DialogContent sx={{ minHeight: 400 }}>
          {/* Indexing Tab */}
          <TabPanel value={tabValue} index={0}>
            <Typography variant="h6" gutterBottom>
              Media Indexing
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Configure directories to scan for media files. The system will analyze images and videos using AI to enable semantic search.
            </Typography>

            {/* Current Status */}
            {indexingLoading ? (
              <CircularProgress size={24} />
            ) : indexingStatus ? (
              <Alert 
                severity={indexingStatus.status === 'idle' ? 'info' : 'warning'}
                sx={{ mb: 2 }}
              >
                Status: {indexingStatus.status}
                {indexingStatus.progress && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    Progress: {indexingStatus.progress.processed_files} / {indexingStatus.progress.total_files} files
                  </Typography>
                )}
              </Alert>
            ) : null}

            {/* Directory Configuration */}
            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle2" gutterBottom>
                Directories to Index
              </Typography>
              
              <Stack direction="row" spacing={1} sx={{ mb: 2 }}>
                <TextField
                  fullWidth
                  size="small"
                  placeholder="/path/to/your/media/folder"
                  value={newPath}
                  onChange={(e) => setNewPath(e.target.value)}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      handleAddPath();
                    }
                  }}
                />
                <Button
                  variant="outlined"
                  startIcon={<AddIcon />}
                  onClick={handleAddPath}
                  disabled={!newPath.trim()}
                >
                  Add
                </Button>
              </Stack>

              {indexingPaths.length > 0 && (
                <Stack spacing={1}>
                  {indexingPaths.map((path, index) => (
                    <Chip
                      key={index}
                      icon={<FolderIcon />}
                      label={path}
                      onDelete={() => handleRemovePath(path)}
                      deleteIcon={<DeleteIcon />}
                      variant="outlined"
                    />
                  ))}
                </Stack>
              )}
            </Box>

            {/* Indexing Controls */}
            <Stack direction="row" spacing={2}>
              <Button
                variant="contained"
                onClick={handleStartIndexing}
                disabled={indexingPaths.length === 0 || startIndexingMutation.isPending}
                startIcon={startIndexingMutation.isPending ? <CircularProgress size={16} /> : undefined}
              >
                Start Indexing
              </Button>
              
              {indexingStatus?.status === 'processing' && (
                <>
                  <Button
                    variant="outlined"
                    onClick={() => handleControlIndexing('pause')}
                    disabled={controlIndexingMutation.isPending}
                  >
                    Pause
                  </Button>
                  <Button
                    variant="outlined"
                    color="error"
                    onClick={() => handleControlIndexing('cancel')}
                    disabled={controlIndexingMutation.isPending}
                  >
                    Cancel
                  </Button>
                </>
              )}
            </Stack>
          </TabPanel>

          {/* System Status Tab */}
          <TabPanel value={tabValue} index={1}>
            <Typography variant="h6" gutterBottom>
              System Status
            </Typography>

            {healthLoading ? (
              <CircularProgress />
            ) : healthData ? (
              <Stack spacing={2}>
                <Alert severity={getHealthStatusColor(healthData.status)}>
                  Overall Status: {healthData.status}
                </Alert>

                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Services
                  </Typography>
                  <Stack spacing={1}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography>Database</Typography>
                      <Chip
                        label={healthData.database_healthy ? 'Healthy' : 'Unhealthy'}
                        color={healthData.database_healthy ? 'success' : 'error'}
                        size="small"
                      />
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography>Ollama (AI)</Typography>
                      <Chip
                        label={healthData.ollama_healthy ? 'Healthy' : 'Unhealthy'}
                        color={healthData.ollama_healthy ? 'success' : 'error'}
                        size="small"
                      />
                    </Box>
                  </Stack>
                </Box>

                <Divider />

                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    System Metrics
                  </Typography>
                  <Stack spacing={1}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography>Memory Usage</Typography>
                      <Typography>
                        {healthData.memory_usage_mb ? `${Math.round(healthData.memory_usage_mb)} MB` : 'N/A'}
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography>CPU Usage</Typography>
                      <Typography>
                        {healthData.cpu_usage_percent ? `${healthData.cpu_usage_percent.toFixed(1)}%` : 'N/A'}
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography>Indexed Files</Typography>
                      <Typography>{healthData.indexed_files_count || 0}</Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography>Uptime</Typography>
                      <Typography>
                        {healthData.uptime_seconds ? `${Math.round(healthData.uptime_seconds / 60)} minutes` : 'N/A'}
                      </Typography>
                    </Box>
                  </Stack>
                </Box>
              </Stack>
            ) : (
              <Alert severity="error">
                Unable to fetch system status
              </Alert>
            )}
          </TabPanel>

          {/* About Tab */}
          <TabPanel value={tabValue} index={2}>
            <Typography variant="h6" gutterBottom>
              About Media Semantic Search
            </Typography>
            <Typography variant="body2" paragraph>
              Version 1.0.0
            </Typography>
            <Typography variant="body2" paragraph>
              A local media semantic search application that uses AI to analyze and search through your images and videos using natural language queries.
            </Typography>
            
            <Divider sx={{ my: 2 }} />
            
            <Typography variant="subtitle2" gutterBottom>
              Technology Stack
            </Typography>
            <Typography variant="body2" component="div">
              <ul>
                <li>Frontend: React + TypeScript + Material-UI</li>
                <li>Backend: Python + FastAPI</li>
                <li>AI Model: Gemma3:4b via Ollama</li>
                <li>Database: ChromaDB (Vector Database)</li>
              </ul>
            </Typography>
          </TabPanel>
        </DialogContent>

        <DialogActions>
          <Button onClick={() => setOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default SettingsPanel;
