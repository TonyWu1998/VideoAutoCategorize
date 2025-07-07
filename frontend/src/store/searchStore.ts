/**
 * Zustand store for search state management.
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import {
  MediaItem,
  SearchFilters,
  SearchResponse,
  MediaType,
  defaultSearchFilters,
} from '../types/media';
import { searchAPI, mediaAPI, indexingAPI } from '../services/api';

interface SearchState {
  // Search state
  query: string;
  searchResults: MediaItem[];
  allMediaItems: MediaItem[]; // All media items for filtering
  isLoading: boolean;
  isLoadingAll: boolean; // Loading state for initial load
  error: string | null;
  lastSearchTime: number;
  totalResults: number;

  // Filters
  filters: SearchFilters;

  // UI state
  selectedItems: Set<string>;
  viewMode: 'grid' | 'list';
  sortBy: 'relevance' | 'date' | 'name' | 'size';
  sortOrder: 'asc' | 'desc';

  // Search suggestions
  suggestions: string[];
  showSuggestions: boolean;

  // Indexing monitoring
  isMonitoringIndexing: boolean;
  lastIndexingStatus: string | null;
  
  // Actions
  setQuery: (query: string) => void;
  setFilters: (filters: Partial<SearchFilters>) => void;
  performSearch: (query?: string) => Promise<void>;
  loadAllMedia: () => Promise<void>; // Load all media for default display
  filterMedia: (query: string) => void; // Filter existing media
  clearResults: () => void;
  clearError: () => void;

  // Selection actions
  selectItem: (fileId: string) => void;
  deselectItem: (fileId: string) => void;
  selectAll: () => void;
  clearSelection: () => void;

  // UI actions
  setViewMode: (mode: 'grid' | 'list') => void;
  setSorting: (sortBy: string, sortOrder: 'asc' | 'desc') => void;

  // Suggestions
  getSuggestions: (query: string) => Promise<void>;
  hideSuggestions: () => void;

  // Similar search
  findSimilar: (fileId: string) => Promise<void>;

  // Data refresh methods
  refreshData: () => Promise<void>;
  removeItemFromState: (fileId: string) => void;
  clearAllData: () => void;

  // Indexing status monitoring
  startIndexingMonitor: () => void;
  stopIndexingMonitor: () => void;
  checkIndexingStatusOnce: () => Promise<void>;
  isMonitoringIndexing: boolean;
}

export const useSearchStore = create<SearchState>()(
  devtools(
    (set, get) => ({
      // Initial state
      query: '',
      searchResults: [],
      allMediaItems: [], // Store all media for filtering
      isLoading: false,
      isLoadingAll: false, // Loading state for initial load
      error: null,
      lastSearchTime: 0,
      totalResults: 0,

      filters: { ...defaultSearchFilters },

      selectedItems: new Set(),
      viewMode: 'grid',
      sortBy: 'relevance',
      sortOrder: 'desc',

      suggestions: [],
      showSuggestions: false,

      // Indexing monitoring
      isMonitoringIndexing: false,
      lastIndexingStatus: null,
      
      // Actions
      setQuery: (query: string) => {
        set({ query, error: null });

        // If query is empty, show all media, otherwise filter
        if (query.trim() === '') {
          const state = get();
          set({ searchResults: state.allMediaItems });
        } else {
          // Filter existing media or perform search
          get().filterMedia(query);
        }

        // Get suggestions if query is not empty
        if (query.trim().length > 2) {
          get().getSuggestions(query);
        } else {
          set({ suggestions: [], showSuggestions: false });
        }
      },
      
      setFilters: (newFilters: Partial<SearchFilters>) => {
        set((state) => ({
          filters: { ...state.filters, ...newFilters },
          error: null,
        }));
      },
      
      loadAllMedia: async () => {
        set({ isLoadingAll: true, error: null });

        try {
          // Load all media files from the library
          const allMedia = await mediaAPI.listFiles(
            MediaType.ALL,
            200, // Maximum allowed by backend
            0,
            'created_date',
            'desc'
          );

          set({
            allMediaItems: allMedia,
            searchResults: allMedia, // Show all by default
            totalResults: allMedia.length,
            isLoadingAll: false,
          });

        } catch (error) {
          console.error('Failed to load all media:', error);
          set({
            allMediaItems: [],
            searchResults: [],
            totalResults: 0,
            isLoadingAll: false,
            error: error instanceof Error ? error.message : 'Failed to load media',
          });
        }
      },

      filterMedia: (query: string) => {
        const state = get();
        const queryLower = query.toLowerCase();

        // Filter allMediaItems based on query
        const filtered = state.allMediaItems.filter(item => {
          const fileName = item.metadata.file_name.toLowerCase();
          const description = item.metadata.ai_description?.toLowerCase() || '';
          const tags = item.metadata.ai_tags.join(' ').toLowerCase();

          return fileName.includes(queryLower) ||
                 description.includes(queryLower) ||
                 tags.includes(queryLower);
        });

        set({
          searchResults: filtered,
          totalResults: filtered.length,
        });
      },

      performSearch: async (searchQuery?: string) => {
        const state = get();
        const query = searchQuery || state.query;

        if (!query.trim()) {
          // If no query, show all media
          set({ searchResults: state.allMediaItems, totalResults: state.allMediaItems.length });
          return;
        }

        set({
          isLoading: true,
          error: null,
          showSuggestions: false,
          selectedItems: new Set()
        });

        try {
          const searchRequest = {
            query: query.trim(),
            filters: state.filters,
            include_metadata: true,
            include_thumbnails: true,
          };

          const response: SearchResponse = await searchAPI.semanticSearch(searchRequest);

          let results = response.results;

          // Apply client-side sorting if not relevance-based
          if (state.sortBy !== 'relevance') {
            results = [...results].sort((a, b) => {
              let aValue: any, bValue: any;

              switch (state.sortBy) {
                case 'date':
                  aValue = new Date(a.metadata.created_date);
                  bValue = new Date(b.metadata.created_date);
                  break;
                case 'name':
                  aValue = a.metadata.file_name.toLowerCase();
                  bValue = b.metadata.file_name.toLowerCase();
                  break;
                case 'size':
                  aValue = a.metadata.file_size;
                  bValue = b.metadata.file_size;
                  break;
                default:
                  return 0;
              }

              if (aValue < bValue) return state.sortOrder === 'asc' ? -1 : 1;
              if (aValue > bValue) return state.sortOrder === 'asc' ? 1 : -1;
              return 0;
            });
          }

          set({
            searchResults: results,
            totalResults: response.total_results,
            lastSearchTime: response.search_time_ms,
            isLoading: false,
            query: query.trim(),
          });

        } catch (error) {
          console.error('Search failed:', error);
          set({
            searchResults: [],
            totalResults: 0,
            isLoading: false,
            error: error instanceof Error ? error.message : 'Search failed',
          });
        }
      },
      
      clearResults: () => {
        const state = get();
        set({
          searchResults: state.allMediaItems, // Reset to show all media
          totalResults: state.allMediaItems.length,
          query: '',
          error: null,
          selectedItems: new Set(),
          suggestions: [],
          showSuggestions: false,
        });
      },
      
      clearError: () => {
        set({ error: null });
      },
      
      // Selection actions
      selectItem: (fileId: string) => {
        set((state) => ({
          selectedItems: new Set([...state.selectedItems, fileId]),
        }));
      },
      
      deselectItem: (fileId: string) => {
        set((state) => {
          const newSelection = new Set(state.selectedItems);
          newSelection.delete(fileId);
          return { selectedItems: newSelection };
        });
      },
      
      selectAll: () => {
        set((state) => ({
          selectedItems: new Set(state.searchResults.map(item => item.file_id)),
        }));
      },
      
      clearSelection: () => {
        set({ selectedItems: new Set() });
      },
      
      // UI actions
      setViewMode: (mode: 'grid' | 'list') => {
        set({ viewMode: mode });
      },
      
      setSorting: (sortBy: string, sortOrder: 'asc' | 'desc') => {
        set({ sortBy: sortBy as any, sortOrder });
        
        // Re-sort current results
        const state = get();
        if (state.searchResults.length > 0) {
          // Trigger re-search to apply new sorting
          state.performSearch();
        }
      },
      
      // Suggestions
      getSuggestions: async (query: string) => {
        try {
          const suggestions = await searchAPI.getSuggestions(query, 8);
          set({ suggestions, showSuggestions: suggestions.length > 0 });
        } catch (error) {
          console.error('Failed to get suggestions:', error);
          set({ suggestions: [], showSuggestions: false });
        }
      },
      
      hideSuggestions: () => {
        set({ showSuggestions: false });
      },
      
      // Similar search
      findSimilar: async (fileId: string) => {
        set({ isLoading: true, error: null });

        try {
          const similarItems = await searchAPI.findSimilar(fileId, 20, 0.4);

          set({
            searchResults: similarItems,
            totalResults: similarItems.length,
            isLoading: false,
            query: `Similar to selected file`,
            selectedItems: new Set(),
          });

        } catch (error) {
          console.error('Similar search failed:', error);
          set({
            isLoading: false,
            error: error instanceof Error ? error.message : 'Similar search failed',
          });
        }
      },

      // Data refresh methods
      refreshData: async () => {
        // Reload all media data from the backend
        await get().loadAllMedia();
      },

      removeItemFromState: (fileId: string) => {
        const state = get();

        // Remove from both allMediaItems and searchResults
        const updatedAllMedia = state.allMediaItems.filter(item => item.file_id !== fileId);
        const updatedSearchResults = state.searchResults.filter(item => item.file_id !== fileId);

        // Remove from selected items if present
        const updatedSelectedItems = new Set(state.selectedItems);
        updatedSelectedItems.delete(fileId);

        set({
          allMediaItems: updatedAllMedia,
          searchResults: updatedSearchResults,
          totalResults: updatedSearchResults.length,
          selectedItems: updatedSelectedItems,
        });
      },

      clearAllData: () => {
        // Clear all data when database is cleared
        set({
          allMediaItems: [],
          searchResults: [],
          totalResults: 0,
          query: '',
          selectedItems: new Set(),
          error: null,
          suggestions: [],
          showSuggestions: false,
        });
      },

      // Indexing status monitoring
      startIndexingMonitor: () => {
        const state = get();
        if (state.isMonitoringIndexing) {
          return; // Already monitoring
        }

        set({ isMonitoringIndexing: true });

        const checkIndexingStatus = async () => {
          try {
            const status = await indexingAPI.getStatus();
            const currentState = get();

            // Check if indexing just completed or failed
            const wasProcessing = currentState.lastIndexingStatus === 'processing' ||
                                currentState.lastIndexingStatus === 'scanning';
            const isNowComplete = status.status === 'completed' || status.status === 'idle';
            const isNowFailed = status.status === 'failed';

            // Also check for progress completion (in case status is stuck)
            const progressComplete = status.progress &&
                                   status.progress.total_files > 0 &&
                                   status.progress.processed_files >= status.progress.total_files;

            if (wasProcessing && (isNowComplete || isNowFailed || progressComplete)) {
              // Indexing finished (either completed or failed), refresh data
              console.log(`Indexing ${status.status} (progress: ${status.progress?.processed_files}/${status.progress?.total_files}), refreshing media data...`);
              try {
                await currentState.refreshData();
                console.log('Media data refreshed successfully after indexing completion');
              } catch (refreshError) {
                console.error('Failed to refresh data after indexing completion:', refreshError);
              }
            }

            // Update last known status
            set({ lastIndexingStatus: status.status });

            // Continue monitoring if still indexing or scanning (and progress not complete)
            const shouldContinueMonitoring = currentState.isMonitoringIndexing &&
                                           (status.status === 'processing' || status.status === 'scanning') &&
                                           !progressComplete;

            if (shouldContinueMonitoring) {
              setTimeout(checkIndexingStatus, 2000); // Check every 2 seconds
            } else {
              // Stop monitoring when indexing is done (completed, failed, idle, or progress complete)
              console.log(`Stopping indexing monitor. Final status: ${status.status}, progress: ${status.progress?.processed_files}/${status.progress?.total_files}`);
              set({ isMonitoringIndexing: false });
            }
          } catch (error) {
            console.error('Failed to check indexing status:', error);
            // Continue monitoring even on error, but with longer interval
            const currentState = get();
            if (currentState.isMonitoringIndexing) {
              setTimeout(checkIndexingStatus, 5000); // Check every 5 seconds on error
            }
          }
        };

        // Start monitoring immediately
        checkIndexingStatus();
      },

      stopIndexingMonitor: () => {
        set({ isMonitoringIndexing: false, lastIndexingStatus: null });
      },

      checkIndexingStatusOnce: async () => {
        try {
          const status = await indexingAPI.getStatus();
          const currentState = get();

          // If indexing is active and we're not already monitoring, start monitoring
          if ((status.status === 'processing' || status.status === 'scanning') &&
              !currentState.isMonitoringIndexing) {
            console.log('Detected active indexing, starting monitor...');
            currentState.startIndexingMonitor();
          }

          // Update status
          set({ lastIndexingStatus: status.status });
        } catch (error) {
          console.error('Failed to check indexing status:', error);
        }
      },
    }),
    {
      name: 'search-store',
    }
  )
);
