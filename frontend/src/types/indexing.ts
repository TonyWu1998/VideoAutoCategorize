/**
 * TypeScript type definitions for indexing-related data structures.
 */

export enum IndexingStatus {
  IDLE = 'idle',
  SCANNING = 'scanning',
  PROCESSING = 'processing',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
  PAUSED = 'paused'
}

export interface IndexingProgress {
  total_files: number;
  processed_files: number;
  successful_files: number;
  failed_files: number;
  skipped_files: number;
  current_file?: string;
  estimated_remaining_seconds?: number;
  files_per_second?: number;
  average_file_size?: number;
}

export interface IndexingStatusResponse {
  status: IndexingStatus;
  job_id?: string;
  started_at?: string;
  estimated_completion?: string;
  progress?: IndexingProgress;
  recent_results: IndexingResult[];
  current_paths: string[];
  batch_size: number;
  max_concurrent: number;
  success: boolean;
  message?: string;
}

export interface IndexingResult {
  file_path: string;
  file_id?: string;
  status: string;
  processing_time_seconds: number;
  description?: string;
  tags: string[];
  confidence?: number;
  error_message?: string;
  error_code?: string;
  file_size?: number;
  dimensions?: string;
  duration?: number;
}

export interface IndexingRequest {
  paths: string[];
  recursive: boolean;
  force_reindex: boolean;
  batch_size?: number;
  max_concurrent?: number;
  include_patterns: string[];
  exclude_patterns: string[];
  min_file_size?: number;
  max_file_size?: number;
}

export interface IndexingStats {
  total_indexed_files: number;
  total_processing_time: number;
  average_processing_time: number;
  total_successful: number;
  total_failed: number;
  success_rate: number;
  files_per_hour: number;
  peak_processing_rate?: number;
  images_indexed: number;
  videos_indexed: number;
  last_indexing_date?: string;
  files_indexed_today: number;
  files_indexed_this_week: number;
}

export interface FileWatcherStatus {
  enabled: boolean;
  watched_paths: string[];
  events_processed: number;
  pending_files: number;
  last_event_time?: string;
}

// Default indexing request
export const defaultIndexingRequest: IndexingRequest = {
  paths: [],
  recursive: true,
  force_reindex: false,
  batch_size: 10,
  max_concurrent: 4,
  include_patterns: [],
  exclude_patterns: [],
};

// Utility functions
export const getStatusColor = (status: IndexingStatus): 'primary' | 'secondary' | 'success' | 'error' | 'warning' => {
  switch (status) {
    case IndexingStatus.IDLE:
      return 'secondary';
    case IndexingStatus.SCANNING:
    case IndexingStatus.PROCESSING:
      return 'primary';
    case IndexingStatus.COMPLETED:
      return 'success';
    case IndexingStatus.FAILED:
      return 'error';
    case IndexingStatus.CANCELLED:
    case IndexingStatus.PAUSED:
      return 'warning';
    default:
      return 'secondary';
  }
};

export const getStatusIcon = (status: IndexingStatus): string => {
  switch (status) {
    case IndexingStatus.IDLE:
      return 'pause_circle';
    case IndexingStatus.SCANNING:
      return 'search';
    case IndexingStatus.PROCESSING:
      return 'autorenew';
    case IndexingStatus.COMPLETED:
      return 'check_circle';
    case IndexingStatus.FAILED:
      return 'error';
    case IndexingStatus.CANCELLED:
      return 'cancel';
    case IndexingStatus.PAUSED:
      return 'pause';
    default:
      return 'help';
  }
};

export const calculateProgress = (progress?: IndexingProgress): number => {
  if (!progress || progress.total_files === 0) return 0;
  return Math.round((progress.processed_files / progress.total_files) * 100);
};

export const formatProcessingTime = (seconds: number): string => {
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`;
  } else if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}m ${remainingSeconds}s`;
  } else {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  }
};
