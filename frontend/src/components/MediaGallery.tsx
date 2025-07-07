/**
 * Media gallery component for displaying search results.
 */

import React, { useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardMedia,
  CardContent,
  Typography,
  Chip,
  Stack,
  IconButton,
  Skeleton,
  Paper,
  Checkbox,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  CircularProgress,
  Snackbar,
  Alert,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Image as ImageIcon,
  MoreVert as MoreVertIcon,
  Download as DownloadIcon,
  Info as InfoIcon,
  FindInPage as SimilarIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { MediaItem, MediaType, formatFileSize, formatDuration } from '../types/media';
import { useSearchStore } from '../store/searchStore';
import { mediaAPI, indexingAPI } from '../services/api';
import MediaViewer from './MediaViewer';
import { useMutation, useQueryClient } from '@tanstack/react-query';

interface MediaGalleryProps {
  results: MediaItem[];
  loading: boolean;
}

const MediaGallery: React.FC<MediaGalleryProps> = ({ results, loading }) => {
  const {
    selectedItems,
    viewMode,
    query,
    selectItem,
    deselectItem,
    findSimilar,
    removeItemFromState,
    refreshData,
    checkIndexingStatusOnce,
  } = useSearchStore();

  const [selectedMedia, setSelectedMedia] = useState<MediaItem | null>(null);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [menuFileId, setMenuFileId] = useState<string | null>(null);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [itemToDelete, setItemToDelete] = useState<MediaItem | null>(null);
  const [reindexingItems, setReindexingItems] = useState<Set<string>>(new Set());
  const [snackbar, setSnackbar] = useState<{open: boolean, message: string, severity: 'success' | 'error'}>({
    open: false,
    message: '',
    severity: 'success'
  });

  const queryClient = useQueryClient();

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: async (fileId: string) => {
      await mediaAPI.deleteFiles([fileId], false, false);
    },
    onSuccess: (_, fileId) => {
      // Remove item from store state immediately
      removeItemFromState(fileId);

      // Invalidate and refetch search results (for other components)
      queryClient.invalidateQueries({ queryKey: ['search'] });
      queryClient.invalidateQueries({ queryKey: ['media'] });
      setDeleteConfirmOpen(false);
      setItemToDelete(null);
    },
    onError: (error) => {
      console.error('Failed to delete file:', error);
      // You could add a toast notification here
    },
  });

  // Reindex mutation
  const reindexMutation = useMutation({
    mutationFn: async (fileIds: string[]) => {
      return await indexingAPI.reindexFiles(fileIds);
    },
    onSuccess: async (_, fileIds) => {
      // Show success message
      setSnackbar({
        open: true,
        message: `Successfully started reindexing ${fileIds.length} file${fileIds.length > 1 ? 's' : ''}`,
        severity: 'success'
      });

      // Check if indexing is now active and start monitoring if needed
      try {
        await checkIndexingStatusOnce();
      } catch (error) {
        console.error('Failed to check indexing status after reindexing:', error);
      }

      // Remove from reindexing set after a delay to show completion
      setTimeout(() => {
        setReindexingItems(prev => {
          const newSet = new Set(prev);
          fileIds.forEach(id => newSet.delete(id));
          return newSet;
        });
      }, 3000); // Increased delay to allow for indexing to start

      // Also invalidate React Query cache for other components
      queryClient.invalidateQueries({ queryKey: ['search'] });
      queryClient.invalidateQueries({ queryKey: ['media'] });
    },
    onError: (error, fileIds) => {
      console.error('Failed to reindex files:', error);

      // Show error message
      setSnackbar({
        open: true,
        message: `Failed to reindex files: ${error instanceof Error ? error.message : 'Unknown error'}`,
        severity: 'error'
      });

      // Remove from reindexing set on error
      setReindexingItems(prev => {
        const newSet = new Set(prev);
        fileIds.forEach(id => newSet.delete(id));
        return newSet;
      });
    },
  });

  const handleItemClick = (item: MediaItem) => {
    setSelectedMedia(item);
  };

  const handleSelectionChange = (fileId: string, checked: boolean) => {
    if (checked) {
      selectItem(fileId);
    } else {
      deselectItem(fileId);
    }
  };

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, fileId: string) => {
    event.stopPropagation();
    setAnchorEl(event.currentTarget);
    setMenuFileId(fileId);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
    setMenuFileId(null);
  };

  const handleDownload = (item: MediaItem) => {
    const downloadUrl = mediaAPI.getDownloadUrl(item.file_id);
    window.open(downloadUrl, '_blank');
    handleMenuClose();
  };

  const handleFindSimilar = (fileId: string) => {
    findSimilar(fileId);
    handleMenuClose();
  };

  const handleDeleteClick = (item: MediaItem) => {
    setItemToDelete(item);
    setDeleteConfirmOpen(true);
    handleMenuClose();
  };

  const handleDeleteConfirm = () => {
    if (itemToDelete) {
      deleteMutation.mutate(itemToDelete.file_id);
    }
  };

  const handleDeleteCancel = () => {
    setDeleteConfirmOpen(false);
    setItemToDelete(null);
  };

  const handleReindex = (fileId: string) => {
    setReindexingItems(prev => new Set([...prev, fileId]));
    reindexMutation.mutate([fileId]);
    setAnchorEl(null);
    setMenuFileId(null);
  };

  const handleBatchReindex = () => {
    const fileIds = Array.from(selectedItems);
    if (fileIds.length > 0) {
      setReindexingItems(prev => new Set([...prev, ...fileIds]));
      reindexMutation.mutate(fileIds);
    }
  };

  const renderLoadingSkeleton = () => (
    <Grid container spacing={2}>
      {Array.from({ length: 8 }).map((_, index) => (
        <Grid item xs={12} sm={6} md={4} lg={3} key={index}>
          <Card>
            <Skeleton variant="rectangular" height={200} />
            <CardContent>
              <Skeleton variant="text" />
              <Skeleton variant="text" width="60%" />
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
  );

  const renderMediaCard = (item: MediaItem) => {
    const isSelected = selectedItems.has(item.file_id);
    const isVideo = item.metadata.media_type === MediaType.VIDEO;
    const isReindexing = reindexingItems.has(item.file_id);
    const thumbnailUrl = mediaAPI.getThumbnailUrl(item.file_id, 300);

    return (
      <Card
        key={item.file_id}
        sx={{
          cursor: 'pointer',
          transition: 'all 0.2s ease-in-out',
          border: isSelected ? 2 : 0,
          borderColor: 'primary.main',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: 4,
          },
        }}
        onClick={() => handleItemClick(item)}
      >
        {/* Media Preview */}
        <Box sx={{ position: 'relative' }}>
          <CardMedia
            component="img"
            height="200"
            image={thumbnailUrl}
            alt={item.metadata.file_name}
            sx={{
              objectFit: 'cover',
              backgroundColor: 'grey.100',
            }}
            onError={(e) => {
              // Fallback to a placeholder or icon
              const target = e.target as HTMLImageElement;
              target.style.display = 'none';
            }}
          />
          
          {/* Video Play Overlay */}
          {isVideo && (
            <Box
              sx={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                backgroundColor: 'rgba(0, 0, 0, 0.6)',
                borderRadius: '50%',
                p: 1,
              }}
            >
              <PlayIcon sx={{ color: 'white', fontSize: 32 }} />
            </Box>
          )}

          {/* Reindexing Overlay */}
          {isReindexing && (
            <Box
              sx={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                backgroundColor: 'rgba(0, 0, 0, 0.7)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexDirection: 'column',
                gap: 1,
              }}
            >
              <CircularProgress size={32} sx={{ color: 'white' }} />
              <Typography variant="caption" sx={{ color: 'white', fontWeight: 'bold' }}>
                Reindexing...
              </Typography>
            </Box>
          )}

          {/* Selection Checkbox */}
          <Checkbox
            checked={isSelected}
            onChange={(e) => {
              e.stopPropagation();
              handleSelectionChange(item.file_id, e.target.checked);
            }}
            sx={{
              position: 'absolute',
              top: 8,
              left: 8,
              backgroundColor: 'rgba(255, 255, 255, 0.8)',
              '&:hover': {
                backgroundColor: 'rgba(255, 255, 255, 0.9)',
              },
            }}
          />

          {/* Actions Menu */}
          <IconButton
            size="small"
            onClick={(e) => handleMenuOpen(e, item.file_id)}
            sx={{
              position: 'absolute',
              top: 8,
              right: 8,
              backgroundColor: 'rgba(255, 255, 255, 0.8)',
              '&:hover': {
                backgroundColor: 'rgba(255, 255, 255, 0.9)',
              },
            }}
          >
            <MoreVertIcon />
          </IconButton>

          {/* Media Type Badge */}
          <Chip
            icon={isVideo ? <PlayIcon /> : <ImageIcon />}
            label={isVideo ? 'Video' : 'Image'}
            size="small"
            sx={{
              position: 'absolute',
              bottom: 8,
              left: 8,
              backgroundColor: 'rgba(0, 0, 0, 0.7)',
              color: 'white',
            }}
          />

          {/* Similarity Score */}
          {item.similarity_score && (
            <Chip
              label={`${Math.round(item.similarity_score * 100)}%`}
              size="small"
              color="primary"
              sx={{
                position: 'absolute',
                bottom: 8,
                right: 8,
              }}
            />
          )}
        </Box>

        {/* Card Content */}
        <CardContent sx={{ p: 2 }}>
          <Typography
            variant="subtitle2"
            noWrap
            sx={{ fontWeight: 500, mb: 1 }}
            title={item.metadata.file_name}
          >
            {item.metadata.file_name}
          </Typography>

          {/* File Info */}
          <Stack direction="row" spacing={1} sx={{ mb: 1 }}>
            <Typography variant="caption" color="text.secondary">
              {formatFileSize(item.metadata.file_size)}
            </Typography>
            {item.metadata.dimensions && (
              <Typography variant="caption" color="text.secondary">
                • {item.metadata.dimensions}
              </Typography>
            )}
            {item.metadata.duration && (
              <Typography variant="caption" color="text.secondary">
                • {formatDuration(item.metadata.duration)}
              </Typography>
            )}
          </Stack>

          {/* AI Description */}
          {item.metadata.ai_description && (
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{
                display: '-webkit-box',
                WebkitLineClamp: 2,
                WebkitBoxOrient: 'vertical',
                overflow: 'hidden',
                lineHeight: 1.3,
              }}
            >
              {item.metadata.ai_description}
            </Typography>
          )}

          {/* Tags */}
          {item.metadata.ai_tags.length > 0 && (
            <Stack direction="row" spacing={0.5} sx={{ mt: 1, flexWrap: 'wrap' }}>
              {item.metadata.ai_tags.slice(0, 3).map((tag, index) => (
                <Chip
                  key={index}
                  label={tag}
                  size="small"
                  variant="outlined"
                  sx={{ fontSize: '0.7rem', height: 20 }}
                />
              ))}
              {item.metadata.ai_tags.length > 3 && (
                <Chip
                  label={`+${item.metadata.ai_tags.length - 3}`}
                  size="small"
                  variant="outlined"
                  sx={{ fontSize: '0.7rem', height: 20 }}
                />
              )}
            </Stack>
          )}

          {/* Indexing Status */}
          {item.metadata.indexed_date && (
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
              Indexed: {new Date(item.metadata.indexed_date).toLocaleDateString()}
              {item.metadata.index_version && ` (v${item.metadata.index_version})`}
            </Typography>
          )}
        </CardContent>
      </Card>
    );
  };

  if (loading) {
    return renderLoadingSkeleton();
  }

  if (results.length === 0) {
    const isSearching = query.trim().length > 0;

    return (
      <Paper
        sx={{
          p: 4,
          textAlign: 'center',
          backgroundColor: 'grey.50',
        }}
      >
        <ImageIcon sx={{ fontSize: 64, color: 'grey.400', mb: 2 }} />
        <Typography variant="h6" color="text.secondary" gutterBottom>
          {isSearching ? 'No matching media files found' : 'No media files in library'}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {isSearching
            ? 'Try adjusting your search query or filters'
            : 'Add some media files to your library to get started'
          }
        </Typography>
      </Paper>
    );
  }

  return (
    <Box>
      {/* Batch Operations Toolbar */}
      {selectedItems.size > 0 && (
        <Paper
          elevation={2}
          sx={{
            p: 2,
            mb: 2,
            display: 'flex',
            alignItems: 'center',
            gap: 2,
            backgroundColor: 'primary.50'
          }}
        >
          <Typography variant="body2" sx={{ flexGrow: 1 }}>
            {selectedItems.size} item{selectedItems.size > 1 ? 's' : ''} selected
          </Typography>
          <Button
            variant="outlined"
            size="small"
            startIcon={<RefreshIcon />}
            onClick={handleBatchReindex}
            disabled={reindexMutation.isPending}
          >
            Reindex Selected
          </Button>
        </Paper>
      )}

      {/* Results Grid */}
      <Grid container spacing={2}>
        {results.map((item) => (
          <Grid item xs={12} sm={6} md={4} lg={3} key={item.file_id}>
            {renderMediaCard(item)}
          </Grid>
        ))}
      </Grid>

      {/* Context Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => {
          const item = results.find(r => r.file_id === menuFileId);
          if (item) handleItemClick(item);
          handleMenuClose();
        }}>
          <ListItemIcon>
            <InfoIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>View Details</ListItemText>
        </MenuItem>
        
        <MenuItem onClick={() => {
          const item = results.find(r => r.file_id === menuFileId);
          if (item) handleDownload(item);
        }}>
          <ListItemIcon>
            <DownloadIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Download</ListItemText>
        </MenuItem>
        
        <MenuItem onClick={() => {
          if (menuFileId) handleFindSimilar(menuFileId);
        }}>
          <ListItemIcon>
            <SimilarIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Find Similar</ListItemText>
        </MenuItem>

        <MenuItem
          onClick={() => {
            if (menuFileId) handleReindex(menuFileId);
          }}
          disabled={menuFileId ? reindexingItems.has(menuFileId) : false}
        >
          <ListItemIcon>
            <RefreshIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>
            {menuFileId && reindexingItems.has(menuFileId) ? 'Reindexing...' : 'Reindex'}
          </ListItemText>
        </MenuItem>

        <MenuItem
          onClick={() => {
            const item = results.find(r => r.file_id === menuFileId);
            if (item) handleDeleteClick(item);
          }}
          sx={{ color: 'error.main' }}
        >
          <ListItemIcon>
            <DeleteIcon fontSize="small" sx={{ color: 'error.main' }} />
          </ListItemIcon>
          <ListItemText>Delete from Library</ListItemText>
        </MenuItem>
      </Menu>

      {/* Media Viewer Modal */}
      {selectedMedia && (
        <MediaViewer
          media={selectedMedia}
          open={Boolean(selectedMedia)}
          onClose={() => setSelectedMedia(null)}
        />
      )}

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteConfirmOpen}
        onClose={handleDeleteCancel}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Delete Media File</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete "{itemToDelete?.metadata.file_name}" from the library?
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            This will remove the file from search results and delete its analysis data.
            The original file will remain on disk.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDeleteCancel}>Cancel</Button>
          <Button
            onClick={handleDeleteConfirm}
            color="error"
            variant="contained"
            disabled={deleteMutation.isPending}
            startIcon={deleteMutation.isPending ? <CircularProgress size={16} /> : <DeleteIcon />}
          >
            {deleteMutation.isPending ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar for feedback */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar(prev => ({ ...prev, open: false }))}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert
          onClose={() => setSnackbar(prev => ({ ...prev, open: false }))}
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default MediaGallery;
