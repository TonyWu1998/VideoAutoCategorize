/**
 * Search bar component with suggestions and filters.
 */

import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  TextField,
  Button,
  Paper,
  List,
  ListItem,
  ListItemText,
  Chip,
  Stack,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Typography,
  Collapse,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Search as SearchIcon,
  FilterList as FilterIcon,
  Clear as ClearIcon,

} from '@mui/icons-material';
import { useSearchStore } from '../store/searchStore';
import { MediaType } from '../types/media';

const SearchBar: React.FC = () => {
  const {
    query,
    filters,
    suggestions,
    showSuggestions,
    isLoading,
    setQuery,
    setFilters,
    performSearch,
    clearResults,
    hideSuggestions,
  } = useSearchStore();

  const [showFilters, setShowFilters] = useState(false);
  const [localQuery, setLocalQuery] = useState(query);
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Update local query when store query changes
  useEffect(() => {
    setLocalQuery(query);
  }, [query]);

  const handleSearch = async () => {
    setQuery(localQuery);
    await performSearch(localQuery);
    hideSuggestions();
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter') {
      handleSearch();
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setLocalQuery(suggestion);
    setQuery(suggestion);
    performSearch(suggestion);
    hideSuggestions();
  };

  const handleClear = () => {
    setLocalQuery('');
    clearResults();
    hideSuggestions();
    searchInputRef.current?.focus();
  };

  const handleQueryChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newQuery = event.target.value;
    setLocalQuery(newQuery);
    setQuery(newQuery);
  };

  const handleFilterChange = (filterName: string, value: any) => {
    setFilters({ [filterName]: value });
  };

  return (
    <Box sx={{ position: 'relative' }}>
      {/* Main Search Bar */}
      <Paper elevation={2} sx={{ p: 2, borderRadius: 2 }}>
        <Stack direction="row" spacing={2} alignItems="center">
          <TextField
            ref={searchInputRef}
            fullWidth
            variant="outlined"
            placeholder="Search your media... (e.g., 'sunset beach', 'family gathering', 'red car')"
            value={localQuery}
            onChange={handleQueryChange}
            onKeyPress={handleKeyPress}
            onFocus={() => {
              if (suggestions.length > 0) {
                // Show suggestions when focusing if we have them
              }
            }}
            InputProps={{
              startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />,
              endAdornment: localQuery && (
                <IconButton size="small" onClick={handleClear}>
                  <ClearIcon />
                </IconButton>
              ),
            }}
          />
          
          <Button
            variant="contained"
            onClick={handleSearch}
            disabled={isLoading || !localQuery.trim()}
            sx={{ minWidth: 120 }}
          >
            {isLoading ? 'Searching...' : 'Search'}
          </Button>
          
          <Tooltip title="Show Filters">
            <IconButton
              onClick={() => setShowFilters(!showFilters)}
              color={showFilters ? 'primary' : 'default'}
            >
              <FilterIcon />
            </IconButton>
          </Tooltip>
        </Stack>

        {/* Search Suggestions */}
        {showSuggestions && suggestions.length > 0 && (
          <Paper
            elevation={4}
            sx={{
              position: 'absolute',
              top: '100%',
              left: 0,
              right: 0,
              zIndex: 1000,
              mt: 1,
              maxHeight: 200,
              overflow: 'auto',
            }}
          >
            <List dense>
              {suggestions.map((suggestion, index) => (
                <ListItem
                  key={index}
                  button
                  onClick={() => handleSuggestionClick(suggestion)}
                  sx={{
                    '&:hover': {
                      backgroundColor: 'action.hover',
                    },
                  }}
                >
                  <SearchIcon sx={{ mr: 1, color: 'text.secondary', fontSize: 16 }} />
                  <ListItemText primary={suggestion} />
                </ListItem>
              ))}
            </List>
          </Paper>
        )}

        {/* Filters Panel */}
        <Collapse in={showFilters}>
          <Box sx={{ mt: 2, pt: 2, borderTop: 1, borderColor: 'divider' }}>
            <Stack spacing={3}>
              {/* Media Type Filter */}
              <Stack direction="row" spacing={2} alignItems="center">
                <FormControl size="small" sx={{ minWidth: 120 }}>
                  <InputLabel>Media Type</InputLabel>
                  <Select
                    value={filters.media_type}
                    label="Media Type"
                    onChange={(e) => handleFilterChange('media_type', e.target.value)}
                  >
                    <MenuItem value={MediaType.ALL}>All</MenuItem>
                    <MenuItem value={MediaType.IMAGE}>Images</MenuItem>
                    <MenuItem value={MediaType.VIDEO}>Videos</MenuItem>
                  </Select>
                </FormControl>

                <FormControl size="small" sx={{ minWidth: 120 }}>
                  <InputLabel>Max Results</InputLabel>
                  <Select
                    value={filters.max_results}
                    label="Max Results"
                    onChange={(e) => handleFilterChange('max_results', e.target.value)}
                  >
                    <MenuItem value={10}>10</MenuItem>
                    <MenuItem value={20}>20</MenuItem>
                    <MenuItem value={50}>50</MenuItem>
                    <MenuItem value={100}>100</MenuItem>
                  </Select>
                </FormControl>
              </Stack>

              {/* Similarity Threshold */}
              <Box>
                <Typography gutterBottom variant="body2" color="text.secondary">
                  Similarity Threshold: {Math.round(filters.min_similarity * 100)}%
                </Typography>
                <Slider
                  value={filters.min_similarity}
                  onChange={(_, value) => handleFilterChange('min_similarity', value)}
                  min={0.1}
                  max={1.0}
                  step={0.1}
                  marks={[
                    { value: 0.1, label: '10%' },
                    { value: 0.5, label: '50%' },
                    { value: 1.0, label: '100%' },
                  ]}
                  sx={{ width: 300 }}
                />
              </Box>

              {/* Active Filters Display */}
              {(filters.media_type !== MediaType.ALL || 
                filters.min_similarity !== 0.3 || 
                filters.max_results !== 20) && (
                <Box>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Active Filters:
                  </Typography>
                  <Stack direction="row" spacing={1} flexWrap="wrap">
                    {filters.media_type !== MediaType.ALL && (
                      <Chip
                        label={`Type: ${filters.media_type}`}
                        size="small"
                        onDelete={() => handleFilterChange('media_type', MediaType.ALL)}
                      />
                    )}
                    {filters.min_similarity !== 0.3 && (
                      <Chip
                        label={`Min Similarity: ${Math.round(filters.min_similarity * 100)}%`}
                        size="small"
                        onDelete={() => handleFilterChange('min_similarity', 0.3)}
                      />
                    )}
                    {filters.max_results !== 20 && (
                      <Chip
                        label={`Max Results: ${filters.max_results}`}
                        size="small"
                        onDelete={() => handleFilterChange('max_results', 20)}
                      />
                    )}
                  </Stack>
                </Box>
              )}
            </Stack>
          </Box>
        </Collapse>
      </Paper>
    </Box>
  );
};

export default SearchBar;
