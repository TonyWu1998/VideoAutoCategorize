/**
 * Task Progress Dashboard component for monitoring indexing operations.
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  LinearProgress,
  Chip,
  IconButton,
  Collapse,
  Stack,
  Button,
  Tooltip,
  Alert,
  CircularProgress,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Close as CloseIcon,
  Refresh as RefreshIcon,
  Cancel as CancelIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Schedule as ScheduleIcon,
} from '@mui/icons-material';
import { indexingAPI } from '../services/api';

interface IndexingProgress {
  total_files: number;
  processed_files: number;
  successful_files: number;
  failed_files: number;
  skipped_files: number;
  current_file?: string;
  estimated_remaining_seconds?: number;

  // Frame-level progress tracking for video analysis
  current_file_frames_total?: number;
  current_file_frames_processed?: number;
  current_frame_activity?: string;
}

interface IndexingStatus {
  status: 'idle' | 'scanning' | 'processing' | 'completed' | 'failed' | 'cancelled' | 'paused';
  job_id?: string;
  started_at?: string;
  estimated_completion?: string;
  progress?: IndexingProgress;
  current_paths: string[];
  batch_size: number;
  max_concurrent: number;
  success: boolean;
  message: string;
}

const TaskProgressDashboard: React.FC = () => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [status, setStatus] = useState<IndexingStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  // Poll for status updates
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        setIsLoading(true);
        setError(null);
        const statusData = await indexingAPI.getStatus();
        setStatus(statusData);
        setLastUpdated(new Date());
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch status');
      } finally {
        setIsLoading(false);
      }
    };

    // Initial fetch
    fetchStatus();

    // Set up polling interval (every 2 seconds when active, every 10 seconds when idle)
    const pollInterval = status?.status && status.status !== 'idle' ? 2000 : 10000;
    const interval = setInterval(fetchStatus, pollInterval);

    return () => clearInterval(interval);
  }, [status?.status]);

  const handleRefresh = () => {
    setError(null);
    // Trigger immediate refresh by clearing status
    setStatus(null);
  };

  const handleCancel = async () => {
    if (!status?.job_id) return;
    
    try {
      await indexingAPI.controlIndexing('cancel', status.job_id);
      // Refresh status after cancellation
      setTimeout(handleRefresh, 1000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cancel job');
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircleIcon color="success" />;
      case 'failed':
        return <ErrorIcon color="error" />;
      case 'cancelled':
        return <CancelIcon color="warning" />;
      case 'processing':
      case 'scanning':
        return <CircularProgress size={20} />;
      default:
        return <ScheduleIcon color="action" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      case 'cancelled':
        return 'warning';
      case 'processing':
      case 'scanning':
        return 'primary';
      default:
        return 'default';
    }
  };

  const formatDuration = (seconds: number) => {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.round(seconds % 60);
    return `${minutes}m ${remainingSeconds}s`;
  };

  const calculateProgress = (progress?: IndexingProgress) => {
    if (!progress || progress.total_files === 0) return 0;

    // Base progress from completed files
    const baseProgress = progress.processed_files / progress.total_files;

    // Add fractional progress from current file's frame processing
    if (progress.current_file_frames_total &&
        progress.current_file_frames_processed &&
        progress.current_file_frames_total > 0) {

      const currentFileProgress = progress.current_file_frames_processed / progress.current_file_frames_total;
      const fractionalProgress = currentFileProgress / progress.total_files;
      return Math.min((baseProgress + fractionalProgress) * 100, 100);
    }

    return baseProgress * 100;
  };

  // Don't show dashboard if no active tasks and not expanded
  const hasActiveTasks = status && status.status !== 'idle';
  if (!hasActiveTasks && !isExpanded) {
    return null;
  }

  return (
    <Paper
      elevation={3}
      sx={{
        position: 'fixed',
        bottom: 16,
        right: 16,
        width: isExpanded ? 400 : 200,
        maxHeight: isExpanded ? 500 : 60,
        zIndex: 1300,
        transition: 'all 0.3s ease-in-out',
      }}
    >
      {/* Header */}
      <Box
        sx={{
          p: 2,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          cursor: 'pointer',
          bgcolor: hasActiveTasks ? 'primary.main' : 'grey.100',
          color: hasActiveTasks ? 'primary.contrastText' : 'text.primary',
        }}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <Stack direction="row" spacing={1} alignItems="center">
          {status && getStatusIcon(status.status)}
          <Typography variant="subtitle2" fontWeight="bold">
            {hasActiveTasks ? 'Task Progress' : 'No Active Tasks'}
          </Typography>
          {status?.progress && (
            <Chip
              label={`${status.progress.processed_files}/${status.progress.total_files}`}
              size="small"
              sx={{ 
                bgcolor: hasActiveTasks ? 'rgba(255,255,255,0.2)' : 'grey.300',
                color: hasActiveTasks ? 'inherit' : 'text.primary',
              }}
            />
          )}
        </Stack>
        
        <Stack direction="row" spacing={0.5}>
          <Tooltip title="Refresh">
            <IconButton
              size="small"
              onClick={(e) => {
                e.stopPropagation();
                handleRefresh();
              }}
              sx={{ color: 'inherit' }}
            >
              <RefreshIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <IconButton size="small" sx={{ color: 'inherit' }}>
            {isExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
          </IconButton>
        </Stack>
      </Box>

      {/* Expanded Content */}
      <Collapse in={isExpanded}>
        <Box sx={{ p: 2 }}>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {status && (
            <Stack spacing={2}>
              {/* Status Information */}
              <Box>
                <Stack direction="row" spacing={1} alignItems="center" mb={1}>
                  <Chip
                    label={status.status.toUpperCase()}
                    color={getStatusColor(status.status) as any}
                    size="small"
                  />
                  {status.job_id && (
                    <Typography variant="caption" color="text.secondary">
                      Job: {status.job_id.slice(0, 8)}...
                    </Typography>
                  )}
                </Stack>

                <Typography variant="body2" color="text.secondary">
                  {status.message}
                </Typography>
              </Box>

              {/* Progress Bar */}
              {status.progress && status.progress.total_files > 0 && (
                <Box>
                  {/* File-level Progress */}
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">
                      Files: {status.progress.processed_files}/{status.progress.total_files}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {Math.round(calculateProgress(status.progress))}%
                    </Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={calculateProgress(status.progress)}
                    sx={{ height: 8, borderRadius: 4, mb: 1 }}
                  />

                  {/* Frame-level Progress (for video analysis) */}
                  {status.progress.current_file_frames_total && status.progress.current_file_frames_total > 0 && (
                    <Box sx={{ mt: 1 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                        <Typography variant="caption" color="text.secondary">
                          Current file frames: {status.progress.current_file_frames_processed || 0}/{status.progress.current_file_frames_total}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {Math.round(((status.progress.current_file_frames_processed || 0) / status.progress.current_file_frames_total) * 100)}%
                        </Typography>
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={((status.progress.current_file_frames_processed || 0) / status.progress.current_file_frames_total) * 100}
                        sx={{
                          height: 4,
                          borderRadius: 2,
                          backgroundColor: 'rgba(0,0,0,0.1)',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: 'secondary.main'
                          }
                        }}
                      />
                    </Box>
                  )}

                  {/* Progress Details */}
                  <Stack direction="row" spacing={1} mt={1}>
                    {status.progress.successful_files > 0 && (
                      <Chip
                        label={`✓ ${status.progress.successful_files}`}
                        size="small"
                        color="success"
                        variant="outlined"
                      />
                    )}
                    {status.progress.failed_files > 0 && (
                      <Chip
                        label={`✗ ${status.progress.failed_files}`}
                        size="small"
                        color="error"
                        variant="outlined"
                      />
                    )}
                    {status.progress.skipped_files > 0 && (
                      <Chip
                        label={`⊘ ${status.progress.skipped_files}`}
                        size="small"
                        color="warning"
                        variant="outlined"
                      />
                    )}
                  </Stack>
                </Box>
              )}

              {/* Current Activity */}
              {status.progress?.current_file && (
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    {status.progress.current_frame_activity ? 'Current activity:' : 'Processing:'}
                  </Typography>

                  {/* Show frame activity if available, otherwise show file name */}
                  {status.progress.current_frame_activity ? (
                    <Typography variant="body2">
                      {status.progress.current_frame_activity}
                    </Typography>
                  ) : (
                    <Typography variant="body2" noWrap>
                      {status.progress.current_file.split('/').pop()}
                    </Typography>
                  )}

                  {/* Show file name as secondary info when showing frame activity */}
                  {status.progress.current_frame_activity && (
                    <Typography variant="caption" color="text.secondary" noWrap display="block">
                      File: {status.progress.current_file.split('/').pop()}
                    </Typography>
                  )}
                </Box>
              )}

              {/* Time Information */}
              {status.started_at && (
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Started: {new Date(status.started_at).toLocaleTimeString()}
                  </Typography>
                  {status.progress?.estimated_remaining_seconds && (
                    <Typography variant="caption" color="text.secondary" display="block">
                      Est. remaining: {formatDuration(status.progress.estimated_remaining_seconds)}
                    </Typography>
                  )}
                </Box>
              )}

              {/* Actions */}
              {status.status === 'processing' || status.status === 'scanning' ? (
                <Button
                  variant="outlined"
                  color="error"
                  size="small"
                  startIcon={<CancelIcon />}
                  onClick={handleCancel}
                  fullWidth
                >
                  Cancel Job
                </Button>
              ) : null}

              {/* Last Updated */}
              {lastUpdated && (
                <Typography variant="caption" color="text.secondary" textAlign="center">
                  Last updated: {lastUpdated.toLocaleTimeString()}
                </Typography>
              )}
            </Stack>
          )}

          {isLoading && !status && (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
              <CircularProgress size={24} />
            </Box>
          )}
        </Box>
      </Collapse>
    </Paper>
  );
};

export default TaskProgressDashboard;
