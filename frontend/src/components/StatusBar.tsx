/**
 * Status bar component showing search results and system information.
 */

import React from 'react';
import {
  Box,
  Typography,
  Chip,
  Stack,
  LinearProgress,
  Tooltip,
} from '@mui/material';
import {
  Search as SearchIcon,
  Timer as TimerIcon,
  PhotoLibrary as PhotoIcon,
} from '@mui/icons-material';
import { useSearchStore } from '../store/searchStore';

const StatusBar: React.FC = () => {
  const {
    searchResults,
    totalResults,
    lastSearchTime,
    query,
    isLoading,
    selectedItems,
  } = useSearchStore();

  if (!query && searchResults.length === 0 && !isLoading) {
    return null;
  }

  return (
    <Box
      sx={{
        position: 'fixed',
        bottom: 0,
        left: 0,
        right: 0,
        backgroundColor: 'background.paper',
        borderTop: 1,
        borderColor: 'divider',
        p: 2,
        zIndex: 1000,
      }}
    >
      {isLoading && (
        <LinearProgress sx={{ mb: 1 }} />
      )}
      
      <Stack
        direction="row"
        spacing={2}
        alignItems="center"
        justifyContent="space-between"
        flexWrap="wrap"
      >
        {/* Search Results Info */}
        <Stack direction="row" spacing={2} alignItems="center">
          {query && (
            <Tooltip title="Current search query">
              <Chip
                icon={<SearchIcon />}
                label={`"${query}"`}
                variant="outlined"
                size="small"
              />
            </Tooltip>
          )}

          {searchResults.length > 0 && (
            <Tooltip title="Search results">
              <Chip
                icon={<PhotoIcon />}
                label={`${searchResults.length} ${totalResults > searchResults.length ? `of ${totalResults}` : ''} results`}
                color="primary"
                size="small"
              />
            </Tooltip>
          )}

          {lastSearchTime > 0 && (
            <Tooltip title="Search execution time">
              <Chip
                icon={<TimerIcon />}
                label={`${lastSearchTime.toFixed(0)}ms`}
                variant="outlined"
                size="small"
              />
            </Tooltip>
          )}
        </Stack>

        {/* Selection Info */}
        {selectedItems.size > 0 && (
          <Stack direction="row" spacing={1} alignItems="center">
            <Typography variant="body2" color="text.secondary">
              {selectedItems.size} selected
            </Typography>
          </Stack>
        )}

        {/* Loading Status */}
        {isLoading && (
          <Typography variant="body2" color="text.secondary">
            Searching...
          </Typography>
        )}
      </Stack>
    </Box>
  );
};

export default StatusBar;
