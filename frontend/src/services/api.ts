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
import {
  PromptTemplate,
  PromptTemplateRequest,
  PromptConfiguration,
  PromptConfigurationRequest,
  PromptValidationRequest,
  PromptValidationResponse,
  PromptTestRequest,
  PromptTestResponse,
  MediaType as PromptMediaType
} from '../types/prompts';

// Ollama Models API Types
export interface OllamaModelInfo {
  name: string;
  size?: string;
  modified_at?: string;
  digest?: string;
  details?: Record<string, any>;
}

export interface AvailableModelsResponse {
  success: boolean;
  models: OllamaModelInfo[];
  vision_models: string[];
  embedding_models: string[];
  total_count: number;
  ollama_connected: boolean;
  message?: string;
}

// API configuration
const API_BASE_URL = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000';
const API_TIMEOUT = 30000; // 30 seconds for regular operations
const LONG_RUNNING_TIMEOUT = 0; // No timeout for long-running operations like video analysis

// Create axios instance for regular operations
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Create axios instance for long-running operations (no timeout)
const longRunningApiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: LONG_RUNNING_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Setup interceptors for both clients
const setupInterceptors = (client: AxiosInstance) => {
  // Request interceptor for logging
  client.interceptors.request.use(
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
  client.interceptors.response.use(
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
};

// Apply interceptors to both clients
setupInterceptors(apiClient);
setupInterceptors(longRunningApiClient);

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

  /**
   * Get video frame URLs
   */
  getVideoFrames: async (fileId: string, frameCount: number = 5, size: number = 300) => {
    const response = await apiClient.get(`/api/media/${fileId}/frames`, {
      params: { frame_count: frameCount, size }
    });
    return response.data;
  },

  /**
   * Get video frame URL
   */
  getVideoFrameUrl: (fileId: string, frameIndex: number, size: number = 300): string => {
    return `${API_BASE_URL}/api/media/${fileId}/frame/${frameIndex}?size=${size}`;
  },

  // =============================================================================
  // LIBRARY MANAGEMENT
  // =============================================================================

  /**
   * Get all files in the media library
   */
  getLibraryContents: async (
    mediaType?: MediaType,
    limit: number = 100,
    offset: number = 0,
    sortBy: string = 'created_date',
    sortOrder: string = 'desc'
  ): Promise<MediaItem[]> => {
    const response = await apiClient.get<MediaItem[]>('/api/media/library', {
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
   * Remove files from the library
   */
  removeFromLibrary: async (
    fileIds: string[],
    deleteFromDisk: boolean = false,
    force: boolean = false
  ) => {
    const response = await apiClient.delete('/api/media/library', {
      params: {
        file_ids: fileIds,
        delete_from_disk: deleteFromDisk,
        force,
      },
    });
    return response.data;
  },

  /**
   * Get library statistics
   */
  getLibraryStats: async (): Promise<MediaStats> => {
    const response = await apiClient.get<MediaStats>('/api/media/library/stats');
    return response.data;
  },

  /**
   * Validate library integrity
   */
  validateLibrary: async () => {
    const response = await apiClient.post('/api/media/library/validate');
    return response.data;
  },
};

// Indexing API
export const indexingAPI = {
  /**
   * Start indexing operation (uses long-running client for no timeout)
   */
  startIndexing: async (request: IndexingRequest): Promise<void> => {
    await longRunningApiClient.post('/api/index/start', request);
  },

  /**
   * Get indexing status (uses regular client for quick status checks)
   */
  getStatus: async (): Promise<IndexingStatusResponse> => {
    const response = await apiClient.get<IndexingStatusResponse>('/api/index/status');
    return response.data;
  },

  /**
   * Control indexing operation (uses regular client for control commands)
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
   * Reindex specific files (uses long-running client for no timeout)
   */
  reindexFiles: async (fileIds: string[]): Promise<void> => {
    await longRunningApiClient.post('/api/index/reindex', fileIds);
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

// Config API
export const configAPI = {
  /**
   * Get current LLM configuration
   */
  getLLMConfig: async () => {
    const response = await apiClient.get('/api/config/llm');
    return response.data;
  },

  /**
   * Update LLM configuration
   */
  updateLLMConfig: async (config: any) => {
    const response = await apiClient.put('/api/config/llm', config);
    return response.data;
  },

  /**
   * Reset LLM configuration to defaults
   */
  resetLLMConfig: async () => {
    const response = await apiClient.post('/api/config/llm/reset');
    return response.data;
  },

  /**
   * Get available Ollama models
   */
  getAvailableModels: async (): Promise<AvailableModelsResponse> => {
    const response = await apiClient.get('/api/config/ollama/models');
    return response.data;
  },
};

// Export the configured axios instance for custom requests
export { apiClient };

// Prompt Management API
export const promptAPI = {
  /**
   * List all prompt templates
   */
  listTemplates: async (mediaType?: PromptMediaType): Promise<PromptTemplate[]> => {
    const params = mediaType ? { media_type: mediaType } : {};
    const response = await apiClient.get('/api/config/prompts/', { params });
    return response.data.templates;
  },

  /**
   * Get a specific prompt template
   */
  getTemplate: async (templateId: string): Promise<PromptTemplate> => {
    const response = await apiClient.get(`/api/config/prompts/${templateId}`);
    return response.data;
  },

  /**
   * Create a new prompt template
   */
  createTemplate: async (request: PromptTemplateRequest): Promise<PromptTemplate> => {
    const response = await apiClient.post('/api/config/prompts/', request);
    return response.data;
  },

  /**
   * Update an existing prompt template
   */
  updateTemplate: async (templateId: string, request: PromptTemplateRequest): Promise<PromptTemplate> => {
    const response = await apiClient.put(`/api/config/prompts/${templateId}`, request);
    return response.data;
  },

  /**
   * Delete a prompt template
   */
  deleteTemplate: async (templateId: string): Promise<void> => {
    await apiClient.delete(`/api/config/prompts/${templateId}`);
  },

  /**
   * Get active prompt configuration
   */
  getActiveConfiguration: async (): Promise<PromptConfiguration> => {
    const response = await apiClient.get('/api/config/prompts/config/active');
    return response.data;
  },

  /**
   * Update active prompt configuration
   */
  updateActiveConfiguration: async (request: PromptConfigurationRequest): Promise<PromptConfiguration> => {
    const response = await apiClient.put('/api/config/prompts/config/active', request);
    return response.data;
  },

  /**
   * Validate a prompt template
   */
  validatePrompt: async (request: PromptValidationRequest): Promise<PromptValidationResponse> => {
    const response = await apiClient.post('/api/config/prompts/validate', request);
    return response.data;
  },

  /**
   * Test a prompt template
   */
  testPrompt: async (request: PromptTestRequest): Promise<PromptTestResponse> => {
    const response = await apiClient.post('/api/config/prompts/test', request);
    return response.data;
  },
};

// Export default API object
export default {
  search: searchAPI,
  media: mediaAPI,
  indexing: indexingAPI,
  health: healthAPI,
  config: configAPI,
  prompts: promptAPI,
};
