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
import { searchAPI, mediaAPI } from '../services/api';

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
    }),
    {
      name: 'search-store',
    }
  )
);
