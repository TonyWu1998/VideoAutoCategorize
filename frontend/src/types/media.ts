/**
 * TypeScript type definitions for media-related data structures.
 */

export enum MediaType {
  IMAGE = 'image',
  VIDEO = 'video',
  ALL = 'all'
}

export interface MediaMetadata {
  file_path: string;
  file_name: string;
  file_size: number;
  created_date: string;
  modified_date: string;
  media_type: MediaType;
  dimensions?: string;
  duration?: number;
  format?: string;
  ai_description?: string;
  ai_tags: string[];
  ai_confidence?: number;
  indexed_date?: string;
  index_version?: string;
}

export interface MediaItem {
  file_id: string;
  metadata: MediaMetadata;
  similarity_score?: number;
  thumbnail_url?: string;
  preview_url?: string;
}

export interface SearchFilters {
  media_type: MediaType;
  date_range?: {
    start?: string;
    end?: string;
  };
  min_similarity: number;
  max_results: number;
  min_file_size?: number;
  max_file_size?: number;
  min_width?: number;
  max_width?: number;
  min_height?: number;
  max_height?: number;
  min_duration?: number;
  max_duration?: number;
  include_tags: string[];
  exclude_tags: string[];
  include_paths: string[];
  exclude_paths: string[];
}

export interface SearchRequest {
  query: string;
  filters: SearchFilters;
  include_metadata: boolean;
  include_thumbnails: boolean;
}

export interface SearchResponse {
  query: string;
  results: MediaItem[];
  total_results: number;
  search_time_ms: number;
  filters_applied: SearchFilters;
  success: boolean;
  message?: string;
}

export interface MediaStats {
  total_files: number;
  total_size_bytes: number;
  image_count: number;
  video_count: number;
  format_breakdown: Record<string, number>;
  oldest_file?: string;
  newest_file?: string;
  last_index_date?: string;
  pending_indexing: number;
}

export interface UploadResponse {
  file_id: string;
  file_path: string;
  file_size: number;
  indexed: boolean;
  success: boolean;
  message?: string;
}

// Default search filters
export const defaultSearchFilters: SearchFilters = {
  media_type: MediaType.ALL,
  min_similarity: 0.3,
  max_results: 20,
  include_tags: [],
  exclude_tags: [],
  include_paths: [],
  exclude_paths: []
};

// Utility functions
export const formatFileSize = (bytes: number): string => {
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  if (bytes === 0) return '0 Bytes';
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
};

export const formatDuration = (seconds: number): string => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  
  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  return `${minutes}:${secs.toString().padStart(2, '0')}`;
};

export const getMediaTypeIcon = (mediaType: MediaType): string => {
  switch (mediaType) {
    case MediaType.IMAGE:
      return 'image';
    case MediaType.VIDEO:
      return 'videocam';
    default:
      return 'folder';
  }
};
