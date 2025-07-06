/**
 * API client for communicating with the backend services.
 */

import axios, { AxiosInstance, AxiosResponse } from 'axios';
import {
  SearchRequest,
  SearchResponse,
  MediaItem,
  MediaStats,
  MediaType,
  UploadResponse
} from '../types/media';
import {
  IndexingRequest,
  IndexingStatusResponse,
  IndexingStats,
  FileWatcherStatus
} from '../types/indexing';

// API configuration
const API_BASE_URL = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000';
const API_TIMEOUT = 30000; // 30 seconds

// Create axios instance
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    return response;
  },
  (error) => {
    console.error('API Response Error:', error.response?.data || error.message);
    
    // Handle common error cases
    if (error.response?.status === 404) {
      throw new Error('Resource not found');
    } else if (error.response?.status === 500) {
      throw new Error('Internal server error');
    } else if (error.code === 'ECONNREFUSED') {
      throw new Error('Cannot connect to server. Please ensure the backend is running.');
    }
    
    throw error;
  }
);

// Search API
export const searchAPI = {
  /**
   * Perform semantic search on media files
   */
  semanticSearch: async (request: SearchRequest): Promise<SearchResponse> => {
    const response = await apiClient.post<SearchResponse>('/api/search', request);
    return response.data;
  },

  /**
   * Simple search with query parameters
   */
  simpleSearch: async (
    query: string,
    mediaType: MediaType = MediaType.ALL,
    limit: number = 20,
    minSimilarity: number = 0.3
  ): Promise<SearchResponse> => {
    const response = await apiClient.get<SearchResponse>('/api/search', {
      params: {
        q: query,
        media_type: mediaType,
        limit,
        min_similarity: minSimilarity,
      },
    });
    return response.data;
  },

  /**
   * Get search suggestions
   */
  getSuggestions: async (query: string, limit: number = 10): Promise<string[]> => {
    const response = await apiClient.get('/api/search/suggestions', {
      params: { q: query, limit },
    });
    return response.data.suggestions;
  },

  /**
   * Find similar media files
   */
  findSimilar: async (
    fileId: string,
    limit: number = 10,
    minSimilarity: number = 0.5
  ): Promise<MediaItem[]> => {
    const response = await apiClient.get(`/api/search/similar/${fileId}`, {
      params: { limit, min_similarity: minSimilarity },
    });
    return response.data.similar_items;
  },

  /**
   * Get popular tags
   */
  getPopularTags: async (limit: number = 50, minFrequency: number = 2) => {
    const response = await apiClient.get('/api/search/tags', {
      params: { limit, min_frequency: minFrequency },
    });
    return response.data.tags;
  },

  /**
   * Clear search cache
   */
  clearCache: async (): Promise<void> => {
    await apiClient.delete('/api/search/cache');
  },
};

// Media API
export const mediaAPI = {
  /**
   * Get media file URL
   */
  getMediaUrl: (fileId: string): string => {
    return `${API_BASE_URL}/api/media/${fileId}`;
  },

  /**
   * Get thumbnail URL
   */
  getThumbnailUrl: (fileId: string, size: number = 200): string => {
    return `${API_BASE_URL}/api/media/${fileId}/thumbnail?size=${size}`;
  },

  /**
   * Get media metadata
   */
  getMetadata: async (fileId: string) => {
    const response = await apiClient.get(`/api/media/${fileId}/metadata`);
    return response.data;
  },

  /**
   * List media files
   */
  listFiles: async (
    mediaType: MediaType = MediaType.ALL,
    limit: number = 50,
    offset: number = 0,
    sortBy: string = 'created_date',
    sortOrder: string = 'desc'
  ): Promise<MediaItem[]> => {
    const response = await apiClient.get('/api/media', {
      params: {
        media_type: mediaType,
        limit,
        offset,
        sort_by: sortBy,
        sort_order: sortOrder,
      },
    });
    return response.data;
  },

  /**
   * Upload media file
   */
  uploadFile: async (file: File, autoIndex: boolean = true): Promise<UploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post<UploadResponse>('/api/media/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      params: {
        auto_index: autoIndex,
      },
    });
    return response.data;
  },

  /**
   * Delete media files
   */
  deleteFiles: async (
    fileIds: string[],
    deleteFromDisk: boolean = false,
    force: boolean = false
  ) => {
    const response = await apiClient.delete('/api/media', {
      data: {
        file_ids: fileIds,
        delete_from_disk: deleteFromDisk,
        force,
      },
    });
    return response.data;
  },

  /**
   * Get media statistics
   */
  getStats: async (): Promise<MediaStats> => {
    const response = await apiClient.get<MediaStats>('/api/media/stats');
    return response.data;
  },

  /**
   * Update media metadata
   */
  updateMetadata: async (
    fileId: string,
    description?: string,
    tags?: string[]
  ): Promise<void> => {
    await apiClient.post(`/api/media/${fileId}/update-metadata`, {
      description,
      tags,
    });
  },

  /**
   * Regenerate thumbnail
   */
  regenerateThumbnail: async (fileId: string): Promise<void> => {
    await apiClient.post(`/api/media/${fileId}/regenerate-thumbnail`);
  },

  /**
   * Get download URL
   */
  getDownloadUrl: (fileId: string): string => {
    return `${API_BASE_URL}/api/media/${fileId}/download`;
  },

  /**
   * Get supported formats
   */
  getSupportedFormats: async () => {
    const response = await apiClient.get('/api/media/formats/supported');
    return response.data;
  },
};

// Indexing API
export const indexingAPI = {
  /**
   * Start indexing operation
   */
  startIndexing: async (request: IndexingRequest): Promise<void> => {
    await apiClient.post('/api/index/start', request);
  },

  /**
   * Get indexing status
   */
  getStatus: async (): Promise<IndexingStatusResponse> => {
    const response = await apiClient.get<IndexingStatusResponse>('/api/index/status');
    return response.data;
  },

  /**
   * Control indexing operation
   */
  controlIndexing: async (action: string, jobId?: string): Promise<void> => {
    await apiClient.post('/api/index/control', {
      action,
      job_id: jobId,
    });
  },

  /**
   * Get indexing history
   */
  getHistory: async (limit: number = 20, offset: number = 0) => {
    const response = await apiClient.get('/api/index/history', {
      params: { limit, offset },
    });
    return response.data;
  },

  /**
   * Get indexing statistics
   */
  getStats: async (): Promise<IndexingStats> => {
    const response = await apiClient.get<IndexingStats>('/api/index/stats');
    return response.data;
  },

  /**
   * Reindex specific files
   */
  reindexFiles: async (fileIds: string[]): Promise<void> => {
    await apiClient.post('/api/index/reindex', { file_ids: fileIds });
  },

  /**
   * Get file watcher status
   */
  getWatcherStatus: async (): Promise<FileWatcherStatus> => {
    const response = await apiClient.get<FileWatcherStatus>('/api/index/watcher');
    return response.data;
  },

  /**
   * Start file watcher
   */
  startWatcher: async (paths: string[]): Promise<void> => {
    await apiClient.post('/api/index/watcher/start', { paths });
  },

  /**
   * Stop file watcher
   */
  stopWatcher: async (): Promise<void> => {
    await apiClient.post('/api/index/watcher/stop');
  },

  /**
   * Clear index
   */
  clearIndex: async (): Promise<void> => {
    await apiClient.delete('/api/index/clear');
  },

  /**
   * Validate index
   */
  validateIndex: async () => {
    const response = await apiClient.get('/api/index/validate');
    return response.data;
  },
};

// Health API
export const healthAPI = {
  /**
   * Check application health
   */
  checkHealth: async () => {
    const response = await apiClient.get('/api/health');
    return response.data;
  },

  /**
   * Simple health check
   */
  simpleHealthCheck: async () => {
    const response = await apiClient.get('/api/health/simple');
    return response.data;
  },

  /**
   * Check database health
   */
  checkDatabase: async () => {
    const response = await apiClient.get('/api/health/database');
    return response.data;
  },

  /**
   * Check Ollama health
   */
  checkOllama: async () => {
    const response = await apiClient.get('/api/health/ollama');
    return response.data;
  },
};

// Export the configured axios instance for custom requests
export { apiClient };

// Export default API object
export default {
  search: searchAPI,
  media: mediaAPI,
  indexing: indexingAPI,
  health: healthAPI,
};
