/**
 * Settings panel component for application configuration.
 */

import React, { useState, useEffect } from 'react';
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
  LinearProgress,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
} from '@mui/material';
import {
  Settings as SettingsIcon,
  Folder as FolderIcon,
  InsertDriveFile as FileIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { indexingAPI, healthAPI, configAPI } from '../services/api';
import { defaultIndexingRequest } from '../types/indexing';
import PromptLibrary from './PromptLibrary';
import PromptConfiguration from './PromptConfiguration';

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
  const [clearConfirmOpen, setClearConfirmOpen] = useState(false);

  // LLM Configuration state
  const [llmConfig, setLlmConfig] = useState({
    max_video_frames: 10,
    video_frame_interval: 30,
    max_image_dimension: 1024,
    image_quality: 85,
    ollama_model: 'gemma3:4b',
    ollama_embedding_model: 'nomic-embed-text',
    ollama_timeout: 120,
    enable_advanced_analysis: false,
    confidence_threshold: 0.5,
    max_tags_per_item: 10,
  });

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
    refetchInterval: (data) => {
      // Only poll if indexing is in progress
      if (!data || data.status === 'processing' || data.status === 'scanning') {
        return 2000; // Refresh every 2 seconds when actively indexing
      }
      return false; // Stop polling when idle, completed, failed, etc.
    },
  });

  // LLM config query
  const { data: llmConfigData, isLoading: llmConfigLoading } = useQuery({
    queryKey: ['llm-config'],
    queryFn: configAPI.getLLMConfig,
    enabled: open,
  });

  // Update local state when config data changes
  useEffect(() => {
    if (llmConfigData) {
      setLlmConfig(llmConfigData);
    }
  }, [llmConfigData]);

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

  // Clear database mutation
  const clearDatabaseMutation = useMutation({
    mutationFn: indexingAPI.clearIndex,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['search'] });
      queryClient.invalidateQueries({ queryKey: ['media'] });
      queryClient.invalidateQueries({ queryKey: ['indexing-status'] });
      setClearConfirmOpen(false);
    },
    onError: (error) => {
      console.error('Failed to clear database:', error);
      // You could add a toast notification here
    },
  });

  // LLM config mutations
  const updateLLMConfigMutation = useMutation({
    mutationFn: configAPI.updateLLMConfig,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['llm-config'] });
    },
    onError: (error) => {
      console.error('Failed to update LLM config:', error);
    },
  });

  const resetLLMConfigMutation = useMutation({
    mutationFn: configAPI.resetLLMConfig,
    onSuccess: (data) => {
      if (data) {
        setLlmConfig(data);
      }
      queryClient.invalidateQueries({ queryKey: ['llm-config'] });
    },
    onError: (error) => {
      console.error('Failed to reset LLM config:', error);
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

  const getPathIcon = (path: string) => {
    // Simple heuristic: if it has a file extension, it's likely a file
    const hasExtension = /\.[a-zA-Z0-9]+$/.test(path);
    return hasExtension ? <FileIcon /> : <FolderIcon />;
  };

  const getPathType = (path: string) => {
    const hasExtension = /\.[a-zA-Z0-9]+$/.test(path);
    return hasExtension ? 'File' : 'Directory';
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

  const handleClearDatabase = () => {
    setClearConfirmOpen(true);
  };

  const handleClearConfirm = () => {
    clearDatabaseMutation.mutate();
  };

  const handleClearCancel = () => {
    setClearConfirmOpen(false);
  };

  const handleLLMConfigChange = (field: string, value: any) => {
    setLlmConfig(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSaveLLMConfig = () => {
    // Basic client-side validation
    const errors = [];

    if (llmConfig.max_video_frames < 1 || llmConfig.max_video_frames > 50) {
      errors.push('Max video frames must be between 1 and 50');
    }

    if (llmConfig.video_frame_interval < 1 || llmConfig.video_frame_interval > 300) {
      errors.push('Video frame interval must be between 1 and 300 seconds');
    }

    if (llmConfig.max_image_dimension < 256 || llmConfig.max_image_dimension > 4096) {
      errors.push('Max image dimension must be between 256 and 4096 pixels');
    }

    if (llmConfig.image_quality < 50 || llmConfig.image_quality > 100) {
      errors.push('Image quality must be between 50 and 100 percent');
    }

    if (llmConfig.ollama_timeout < 30 || llmConfig.ollama_timeout > 600) {
      errors.push('Ollama timeout must be between 30 and 600 seconds');
    }

    if (errors.length > 0) {
      console.error('Validation errors:', errors);
      // You could show these errors in a toast or alert
      return;
    }

    updateLLMConfigMutation.mutate(llmConfig);
  };

  const handleResetLLMConfig = () => {
    resetLLMConfigMutation.mutate();
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
            <Tab label="Library" />
            <Tab label="AI Settings" />
            <Tab label="Prompts" />
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
              Add directories and individual media files to your library. The system will analyze images and videos using AI to enable semantic search.
            </Typography>

            <Alert severity="info" sx={{ mb: 2 }}>
              <Typography variant="body2">
                <strong>Flexible Path Support:</strong> You can add both directories (e.g., <code>/path/to/media/folder</code>)
                and individual files (e.g., <code>/path/to/video.mp4</code>).
                Files are added to a staging area before indexing begins.
              </Typography>
            </Alert>

            {/* Current Status */}
            {indexingLoading ? (
              <CircularProgress size={24} />
            ) : indexingStatus ? (
              <Alert
                severity={
                  indexingStatus.status === 'idle' ? 'info' :
                  indexingStatus.status === 'processing' ? 'warning' :
                  indexingStatus.status === 'completed' ? 'success' :
                  indexingStatus.status === 'failed' ? 'error' : 'info'
                }
                sx={{ mb: 2 }}
              >
                <Box>
                  <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                    Status: {indexingStatus.status.charAt(0).toUpperCase() + indexingStatus.status.slice(1)}
                  </Typography>
                  {indexingStatus.progress && (
                    <>
                      <Typography variant="body2" sx={{ mt: 1 }}>
                        Progress: {indexingStatus.progress.processed_files} / {indexingStatus.progress.total_files} files
                        {indexingStatus.progress.successful_files > 0 && (
                          <span style={{ color: 'green', marginLeft: '8px' }}>
                            ✓ {indexingStatus.progress.successful_files} successful
                          </span>
                        )}
                        {indexingStatus.progress.failed_files > 0 && (
                          <span style={{ color: 'red', marginLeft: '8px' }}>
                            ✗ {indexingStatus.progress.failed_files} failed
                          </span>
                        )}
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={indexingStatus.progress.total_files > 0
                          ? (indexingStatus.progress.processed_files / indexingStatus.progress.total_files) * 100
                          : 0
                        }
                        sx={{ mt: 1, mb: 1 }}
                      />
                    </>
                  )}
                  {indexingStatus.status === 'completed' && indexingStatus.progress && (
                    <Typography variant="body2" sx={{ mt: 1, color: 'success.main' }}>
                      🎉 Indexing completed successfully! {indexingStatus.progress.successful_files} files processed.
                    </Typography>
                  )}
                  {indexingStatus.job_id && (
                    <Typography variant="caption" sx={{ mt: 1, display: 'block', opacity: 0.7 }}>
                      Job ID: {indexingStatus.job_id}
                    </Typography>
                  )}
                </Box>
              </Alert>
            ) : null}

            {/* Media Path Configuration */}
            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle2" gutterBottom>
                Staging Area - Add Files & Directories
              </Typography>

              <Stack direction="row" spacing={1} sx={{ mb: 2 }}>
                <TextField
                  fullWidth
                  size="small"
                  placeholder="/path/to/media/folder or /path/to/video.mp4"
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
                  Add to Staging
                </Button>
              </Stack>

              {indexingPaths.length > 0 && (
                <Box>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    Staged items ({indexingPaths.length}):
                  </Typography>
                  <Stack spacing={1}>
                    {indexingPaths.map((path, index) => (
                      <Chip
                        key={index}
                        icon={getPathIcon(path)}
                        label={`${getPathType(path)}: ${path}`}
                        onDelete={() => handleRemovePath(path)}
                        deleteIcon={<DeleteIcon />}
                        variant="outlined"
                        sx={{
                          justifyContent: 'flex-start',
                          '& .MuiChip-label': {
                            textAlign: 'left',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis'
                          }
                        }}
                      />
                    ))}
                  </Stack>
                </Box>
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
                {indexingPaths.length > 0
                  ? `Index ${indexingPaths.length} Item${indexingPaths.length > 1 ? 's' : ''}`
                  : 'Start Indexing'
                }
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

          {/* Library Tab */}
          <TabPanel value={tabValue} index={1}>
            <Typography variant="h6" gutterBottom>
              Media Library
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              View and manage your indexed media files. Remove files from the library or validate library integrity.
            </Typography>

            {/* Library Contents */}
            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle2" gutterBottom>
                Library Contents
              </Typography>

              <Alert severity="warning" sx={{ mb: 2 }}>
                <Typography variant="body2">
                  <strong>Database Management:</strong> Use these tools to manage your indexed media library.
                  Clearing the database will remove all analysis data and search history.
                </Typography>
              </Alert>

              {/* Database Management Actions */}
              <Stack direction="row" spacing={2} sx={{ mb: 3 }}>
                <Button
                  variant="outlined"
                  color="error"
                  onClick={handleClearDatabase}
                  disabled={clearDatabaseMutation.isPending}
                  startIcon={clearDatabaseMutation.isPending ? <CircularProgress size={16} /> : <DeleteIcon />}
                >
                  {clearDatabaseMutation.isPending ? 'Clearing...' : 'Clear All History'}
                </Button>
                <Button
                  variant="outlined"
                  disabled
                >
                  Validate Library
                </Button>
              </Stack>

              <Typography variant="body2" color="text.secondary">
                Individual files can be deleted using the context menu in search results (right-click or three-dot menu on each item).
              </Typography>
            </Box>
          </TabPanel>

          {/* AI Settings Tab */}
          <TabPanel value={tabValue} index={2}>
            <Typography variant="h6" gutterBottom>
              AI Analysis Settings
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Configure how the AI analyzes your videos and images. These settings affect frame extraction,
              model parameters, and analysis quality.
            </Typography>

            {llmConfigLoading ? (
              <CircularProgress />
            ) : (
              <Stack spacing={3}>
                {/* Error Display */}
                {updateLLMConfigMutation.isError && (
                  <Alert severity="error">
                    Failed to save settings. Please try again.
                  </Alert>
                )}

                {updateLLMConfigMutation.isSuccess && (
                  <Alert severity="success">
                    Settings saved successfully!
                  </Alert>
                )}
                {/* Video Processing Settings */}
                <Box>
                  <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
                    Video Processing
                  </Typography>

                  <Grid container spacing={3}>
                    <Grid item xs={12} sm={6}>
                      <Typography variant="body2" gutterBottom>
                        Frames to Extract: {llmConfig.max_video_frames}
                      </Typography>
                      <Slider
                        value={llmConfig.max_video_frames}
                        onChange={(_, value) => handleLLMConfigChange('max_video_frames', value)}
                        min={1}
                        max={50}
                        step={1}
                        marks={[
                          { value: 1, label: '1' },
                          { value: 10, label: '10' },
                          { value: 25, label: '25' },
                          { value: 50, label: '50' }
                        ]}
                        valueLabelDisplay="auto"
                      />
                      <Typography variant="caption" color="text.secondary">
                        Number of frames to extract from each video for analysis
                      </Typography>
                    </Grid>

                    <Grid item xs={12} sm={6}>
                      <Typography variant="body2" gutterBottom>
                        Frame Interval: {llmConfig.video_frame_interval}s
                      </Typography>
                      <Slider
                        value={llmConfig.video_frame_interval}
                        onChange={(_, value) => handleLLMConfigChange('video_frame_interval', value)}
                        min={1}
                        max={300}
                        step={5}
                        marks={[
                          { value: 1, label: '1s' },
                          { value: 30, label: '30s' },
                          { value: 60, label: '1m' },
                          { value: 300, label: '5m' }
                        ]}
                        valueLabelDisplay="auto"
                      />
                      <Typography variant="caption" color="text.secondary">
                        Extract a frame every N seconds
                      </Typography>
                    </Grid>
                  </Grid>
                </Box>

                {/* Image Processing Settings */}
                <Box>
                  <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
                    Image Processing
                  </Typography>

                  <Grid container spacing={3}>
                    <Grid item xs={12} sm={6}>
                      <Typography variant="body2" gutterBottom>
                        Max Image Dimension: {llmConfig.max_image_dimension}px
                      </Typography>
                      <Slider
                        value={llmConfig.max_image_dimension}
                        onChange={(_, value) => handleLLMConfigChange('max_image_dimension', value)}
                        min={256}
                        max={4096}
                        step={128}
                        marks={[
                          { value: 256, label: '256' },
                          { value: 1024, label: '1024' },
                          { value: 2048, label: '2048' },
                          { value: 4096, label: '4096' }
                        ]}
                        valueLabelDisplay="auto"
                      />
                      <Typography variant="caption" color="text.secondary">
                        Maximum width or height for processed images
                      </Typography>
                    </Grid>

                    <Grid item xs={12} sm={6}>
                      <Typography variant="body2" gutterBottom>
                        Image Quality: {llmConfig.image_quality}%
                      </Typography>
                      <Slider
                        value={llmConfig.image_quality}
                        onChange={(_, value) => handleLLMConfigChange('image_quality', value)}
                        min={50}
                        max={100}
                        step={5}
                        marks={[
                          { value: 50, label: '50%' },
                          { value: 75, label: '75%' },
                          { value: 85, label: '85%' },
                          { value: 100, label: '100%' }
                        ]}
                        valueLabelDisplay="auto"
                      />
                      <Typography variant="caption" color="text.secondary">
                        JPEG compression quality for processed images
                      </Typography>
                    </Grid>
                  </Grid>
                </Box>

                {/* Model Settings */}
                <Box>
                  <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
                    AI Model Configuration
                  </Typography>

                  <Grid container spacing={3}>
                    <Grid item xs={12} sm={6}>
                      <FormControl fullWidth>
                        <InputLabel>Vision Model</InputLabel>
                        <Select
                          value={llmConfig.ollama_model}
                          onChange={(e) => handleLLMConfigChange('ollama_model', e.target.value)}
                          label="Vision Model"
                        >
                          <MenuItem value="gemma3:4b">Gemma3 4B</MenuItem>
                          <MenuItem value="llava:7b">LLaVA 7B</MenuItem>
                          <MenuItem value="llava:13b">LLaVA 13B</MenuItem>
                          <MenuItem value="bakllava:7b">BakLLaVA 7B</MenuItem>
                        </Select>
                      </FormControl>
                      <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                        AI model used for image and video analysis
                      </Typography>
                    </Grid>

                    <Grid item xs={12} sm={6}>
                      <FormControl fullWidth>
                        <InputLabel>Embedding Model</InputLabel>
                        <Select
                          value={llmConfig.ollama_embedding_model}
                          onChange={(e) => handleLLMConfigChange('ollama_embedding_model', e.target.value)}
                          label="Embedding Model"
                        >
                          <MenuItem value="nomic-embed-text">Nomic Embed Text</MenuItem>
                          <MenuItem value="all-minilm">All MiniLM</MenuItem>
                          <MenuItem value="mxbai-embed-large">MxBai Embed Large</MenuItem>
                        </Select>
                      </FormControl>
                      <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                        Model used for generating search embeddings
                      </Typography>
                    </Grid>
                  </Grid>
                </Box>

                {/* Advanced Settings */}
                <Box>
                  <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
                    Advanced Settings
                  </Typography>

                  <Grid container spacing={3}>
                    <Grid item xs={12} sm={6}>
                      <Typography variant="body2" gutterBottom>
                        Request Timeout: {llmConfig.ollama_timeout}s
                      </Typography>
                      <Slider
                        value={llmConfig.ollama_timeout}
                        onChange={(_, value) => handleLLMConfigChange('ollama_timeout', value)}
                        min={30}
                        max={600}
                        step={30}
                        marks={[
                          { value: 30, label: '30s' },
                          { value: 120, label: '2m' },
                          { value: 300, label: '5m' },
                          { value: 600, label: '10m' }
                        ]}
                        valueLabelDisplay="auto"
                      />
                      <Typography variant="caption" color="text.secondary">
                        Maximum time to wait for AI model responses
                      </Typography>
                    </Grid>

                    <Grid item xs={12} sm={6}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={llmConfig.enable_advanced_analysis}
                            onChange={(e) => handleLLMConfigChange('enable_advanced_analysis', e.target.checked)}
                          />
                        }
                        label="Enable Advanced Analysis"
                      />
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                        Use more detailed prompts and analysis techniques
                      </Typography>
                    </Grid>
                  </Grid>
                </Box>

                <Divider sx={{ my: 4 }} />

                {/* Prompt Configuration */}
                <PromptConfiguration />

                {/* Action Buttons */}
                <Stack direction="row" spacing={2} sx={{ pt: 2 }}>
                  <Button
                    variant="contained"
                    onClick={handleSaveLLMConfig}
                    disabled={updateLLMConfigMutation.isPending}
                    startIcon={updateLLMConfigMutation.isPending ? <CircularProgress size={16} /> : undefined}
                  >
                    {updateLLMConfigMutation.isPending ? 'Saving...' : 'Save Settings'}
                  </Button>
                  <Button
                    variant="outlined"
                    onClick={handleResetLLMConfig}
                    disabled={resetLLMConfigMutation.isPending}
                  >
                    Reset to Defaults
                  </Button>
                </Stack>
              </Stack>
            )}
          </TabPanel>

          {/* Prompts Tab */}
          <TabPanel value={tabValue} index={3}>
            <PromptLibrary />
          </TabPanel>

          {/* System Status Tab */}
          <TabPanel value={tabValue} index={4}>
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
          <TabPanel value={tabValue} index={5}>
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

      {/* Clear Database Confirmation Dialog */}
      <Dialog
        open={clearConfirmOpen}
        onClose={handleClearCancel}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Clear All History</DialogTitle>
        <DialogContent>
          <Typography gutterBottom>
            Are you sure you want to clear all indexed video analysis data?
          </Typography>
          <Typography variant="body2" color="text.secondary">
            This action will:
          </Typography>
          <Typography variant="body2" component="div" color="text.secondary" sx={{ mt: 1 }}>
            <ul>
              <li>Remove all video analysis results and descriptions</li>
              <li>Delete all vector embeddings and search data</li>
              <li>Clear the entire search history</li>
              <li>Keep original video files unchanged on disk</li>
            </ul>
          </Typography>
          <Typography variant="body2" color="error.main" sx={{ mt: 2, fontWeight: 'bold' }}>
            This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClearCancel}>Cancel</Button>
          <Button
            onClick={handleClearConfirm}
            color="error"
            variant="contained"
            disabled={clearDatabaseMutation.isPending}
            startIcon={clearDatabaseMutation.isPending ? <CircularProgress size={16} /> : <DeleteIcon />}
          >
            {clearDatabaseMutation.isPending ? 'Clearing...' : 'Clear All History'}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default SettingsPanel;
