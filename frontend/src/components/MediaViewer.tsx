/**
 * Media viewer modal component for detailed media display.
 */

import React from 'react';
import {
  Dialog,
  DialogContent,
  DialogTitle,
  IconButton,
  Box,
  Typography,
  Chip,
  Stack,
  Divider,
  Grid,
  Button,
} from '@mui/material';
import {
  Close as CloseIcon,
  Download as DownloadIcon,
  FindInPage as SimilarIcon,
  PlayArrow as PlayIcon,
} from '@mui/icons-material';
import { MediaItem, MediaType, formatFileSize, formatDuration } from '../types/media';
import { mediaAPI } from '../services/api';
import { useSearchStore } from '../store/searchStore';

interface MediaViewerProps {
  media: MediaItem;
  open: boolean;
  onClose: () => void;
}

const MediaViewer: React.FC<MediaViewerProps> = ({ media, open, onClose }) => {
  const { findSimilar } = useSearchStore();

  const isVideo = media.metadata.media_type === MediaType.VIDEO;
  const mediaUrl = mediaAPI.getMediaUrl(media.file_id);

  const handleDownload = () => {
    const downloadUrl = mediaAPI.getDownloadUrl(media.file_id);
    window.open(downloadUrl, '_blank');
  };

  const handleFindSimilar = () => {
    findSimilar(media.file_id);
    onClose();
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      PaperProps={{
        sx: { height: '90vh' }
      }}
    >
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Typography variant="h6" noWrap sx={{ flex: 1, mr: 2 }}>
          {media.metadata.file_name}
        </Typography>
        <IconButton onClick={onClose}>
          <CloseIcon />
        </IconButton>
      </DialogTitle>

      <DialogContent sx={{ p: 0 }}>
        <Grid container sx={{ height: '100%' }}>
          {/* Media Display */}
          <Grid item xs={12} md={8} sx={{ position: 'relative', backgroundColor: 'black' }}>
            <Box
              sx={{
                width: '100%',
                height: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: 400,
              }}
            >
              {isVideo ? (
                <video
                  controls
                  style={{
                    maxWidth: '100%',
                    maxHeight: '100%',
                    objectFit: 'contain',
                  }}
                  poster={mediaAPI.getThumbnailUrl(media.file_id, 800)}
                >
                  <source src={mediaUrl} type={`video/${media.metadata.format?.toLowerCase()}`} />
                  Your browser does not support the video tag.
                </video>
              ) : (
                <img
                  src={mediaUrl}
                  alt={media.metadata.file_name}
                  style={{
                    maxWidth: '100%',
                    maxHeight: '100%',
                    objectFit: 'contain',
                  }}
                />
              )}
            </Box>

            {/* Media Type Indicator */}
            <Chip
              icon={isVideo ? <PlayIcon /> : undefined}
              label={isVideo ? 'Video' : 'Image'}
              sx={{
                position: 'absolute',
                top: 16,
                left: 16,
                backgroundColor: 'rgba(0, 0, 0, 0.7)',
                color: 'white',
              }}
            />

            {/* Similarity Score */}
            {media.similarity_score && (
              <Chip
                label={`${Math.round(media.similarity_score * 100)}% match`}
                color="primary"
                sx={{
                  position: 'absolute',
                  top: 16,
                  right: 16,
                }}
              />
            )}
          </Grid>

          {/* Metadata Panel */}
          <Grid item xs={12} md={4} sx={{ p: 3, backgroundColor: 'grey.50' }}>
            <Stack spacing={3}>
              {/* Actions */}
              <Stack direction="row" spacing={1}>
                <Button
                  variant="outlined"
                  startIcon={<DownloadIcon />}
                  onClick={handleDownload}
                  size="small"
                >
                  Download
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<SimilarIcon />}
                  onClick={handleFindSimilar}
                  size="small"
                >
                  Find Similar
                </Button>
              </Stack>

              <Divider />

              {/* File Information */}
              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  File Information
                </Typography>
                <Stack spacing={1}>
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      Size
                    </Typography>
                    <Typography variant="body2">
                      {formatFileSize(media.metadata.file_size)}
                    </Typography>
                  </Box>

                  {media.metadata.dimensions && (
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        Dimensions
                      </Typography>
                      <Typography variant="body2">
                        {media.metadata.dimensions}
                      </Typography>
                    </Box>
                  )}

                  {media.metadata.duration && (
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        Duration
                      </Typography>
                      <Typography variant="body2">
                        {formatDuration(media.metadata.duration)}
                      </Typography>
                    </Box>
                  )}

                  {media.metadata.format && (
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        Format
                      </Typography>
                      <Typography variant="body2">
                        {media.metadata.format}
                      </Typography>
                    </Box>
                  )}

                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      Created
                    </Typography>
                    <Typography variant="body2">
                      {new Date(media.metadata.created_date).toLocaleDateString()}
                    </Typography>
                  </Box>
                </Stack>
              </Box>

              <Divider />

              {/* AI Analysis */}
              {media.metadata.ai_description && (
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    AI Description
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {media.metadata.ai_description}
                  </Typography>
                  {media.metadata.ai_confidence && (
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                      Confidence: {Math.round(media.metadata.ai_confidence * 100)}%
                    </Typography>
                  )}
                </Box>
              )}

              {/* Tags */}
              {media.metadata.ai_tags.length > 0 && (
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Tags
                  </Typography>
                  <Stack direction="row" spacing={0.5} sx={{ flexWrap: 'wrap', gap: 0.5 }}>
                    {media.metadata.ai_tags.map((tag, index) => (
                      <Chip
                        key={index}
                        label={tag}
                        size="small"
                        variant="outlined"
                        sx={{ fontSize: '0.75rem' }}
                      />
                    ))}
                  </Stack>
                </Box>
              )}

              <Divider />

              {/* Technical Details */}
              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  Technical Details
                </Typography>
                <Stack spacing={1}>
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      File Path
                    </Typography>
                    <Typography variant="body2" sx={{ wordBreak: 'break-all', fontSize: '0.8rem' }}>
                      {media.metadata.file_path}
                    </Typography>
                  </Box>

                  {media.metadata.indexed_date && (
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        Indexed
                      </Typography>
                      <Typography variant="body2">
                        {new Date(media.metadata.indexed_date).toLocaleDateString()}
                      </Typography>
                    </Box>
                  )}

                  {media.metadata.index_version && (
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        Index Version
                      </Typography>
                      <Typography variant="body2">
                        {media.metadata.index_version}
                      </Typography>
                    </Box>
                  )}
                </Stack>
              </Box>
            </Stack>
          </Grid>
        </Grid>
      </DialogContent>
    </Dialog>
  );
};

export default MediaViewer;
