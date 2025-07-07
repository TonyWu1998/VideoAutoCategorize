/**
 * Prompt configuration component for AI Settings tab.
 */

import React, { useState } from 'react';
import {
  Box,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Alert,
  Stack,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Divider,
} from '@mui/material';
import {
  Settings as SettingsIcon,
  CheckCircle as ActiveIcon,
  Edit as EditIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

import { promptAPI } from '../services/api';
import {
  PromptTemplate,
  PromptConfiguration,
  MediaType,
  getMediaTypeLabel,
  formatDate,
} from '../types/prompts';

const PromptConfigurationComponent: React.FC = () => {
  const queryClient = useQueryClient();
  const [error, setError] = useState<string | null>(null);

  // Queries
  const { data: templates = [], isLoading: templatesLoading } = useQuery({
    queryKey: ['promptTemplates'],
    queryFn: () => promptAPI.listTemplates(),
  });

  const { data: configuration, isLoading: configLoading } = useQuery({
    queryKey: ['promptConfiguration'],
    queryFn: promptAPI.getActiveConfiguration,
  });

  // Mutations
  const updateConfigMutation = useMutation({
    mutationFn: promptAPI.updateActiveConfiguration,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['promptConfiguration'] });
      queryClient.invalidateQueries({ queryKey: ['promptTemplates'] });
      setError(null);
    },
    onError: (error: any) => {
      setError(error.response?.data?.detail || 'Failed to update prompt configuration');
    },
  });

  const handlePromptChange = async (mediaType: MediaType, templateId: string) => {
    try {
      const updateData = {
        active_image_prompt_id: mediaType === MediaType.IMAGE ? templateId : configuration?.active_image_prompt_id,
        active_video_prompt_id: mediaType === MediaType.VIDEO_FRAME ? templateId : configuration?.active_video_prompt_id,
      };

      await updateConfigMutation.mutateAsync(updateData);
    } catch (error) {
      // Error handling is done in mutation callback
    }
  };

  const getTemplatesForMediaType = (mediaType: MediaType) => {
    return templates.filter(template => template.media_type === mediaType);
  };

  const getActiveTemplate = (mediaType: MediaType): PromptTemplate | undefined => {
    if (mediaType === MediaType.IMAGE) {
      return configuration?.active_image_prompt;
    } else {
      return configuration?.active_video_prompt;
    }
  };

  const getActiveTemplateId = (mediaType: MediaType): string => {
    if (mediaType === MediaType.IMAGE) {
      return configuration?.active_image_prompt_id || '';
    } else {
      return configuration?.active_video_prompt_id || '';
    }
  };

  if (templatesLoading || configLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <SettingsIcon />
        Active Prompt Configuration
      </Typography>
      
      <Typography variant="body2" color="text.secondary" paragraph>
        Select which prompt templates to use for analyzing different media types. 
        These prompts control how the AI analyzes and describes your images and videos.
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Stack spacing={3}>
        {/* Image Prompts */}
        <Box>
          <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
            Image Analysis Prompt
          </Typography>
          
          <Stack spacing={2}>
            <FormControl fullWidth>
              <InputLabel>Active Image Prompt</InputLabel>
              <Select
                value={getActiveTemplateId(MediaType.IMAGE)}
                onChange={(e) => handlePromptChange(MediaType.IMAGE, e.target.value)}
                label="Active Image Prompt"
                disabled={updateConfigMutation.isPending}
              >
                {getTemplatesForMediaType(MediaType.IMAGE).map((template) => (
                  <MenuItem key={template.template_id} value={template.template_id}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                      <Typography>{template.name}</Typography>
                      {template.is_default && (
                        <Chip label="Default" size="small" color="primary" />
                      )}
                      {template.is_active && (
                        <ActiveIcon color="success" fontSize="small" />
                      )}
                    </Box>
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {/* Current Image Prompt Details */}
            {getActiveTemplate(MediaType.IMAGE) && (
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="subtitle2" gutterBottom>
                    Current: {getActiveTemplate(MediaType.IMAGE)?.name}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    {getActiveTemplate(MediaType.IMAGE)?.description}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Last modified: {formatDate(getActiveTemplate(MediaType.IMAGE)?.modified_date || '')}
                  </Typography>
                </CardContent>
              </Card>
            )}
          </Stack>
        </Box>

        <Divider />

        {/* Video Prompts */}
        <Box>
          <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
            Video Frame Analysis Prompt
          </Typography>
          
          <Stack spacing={2}>
            <FormControl fullWidth>
              <InputLabel>Active Video Prompt</InputLabel>
              <Select
                value={getActiveTemplateId(MediaType.VIDEO_FRAME)}
                onChange={(e) => handlePromptChange(MediaType.VIDEO_FRAME, e.target.value)}
                label="Active Video Prompt"
                disabled={updateConfigMutation.isPending}
              >
                {getTemplatesForMediaType(MediaType.VIDEO_FRAME).map((template) => (
                  <MenuItem key={template.template_id} value={template.template_id}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                      <Typography>{template.name}</Typography>
                      {template.is_default && (
                        <Chip label="Default" size="small" color="primary" />
                      )}
                      {template.is_active && (
                        <ActiveIcon color="success" fontSize="small" />
                      )}
                    </Box>
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {/* Current Video Prompt Details */}
            {getActiveTemplate(MediaType.VIDEO_FRAME) && (
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="subtitle2" gutterBottom>
                    Current: {getActiveTemplate(MediaType.VIDEO_FRAME)?.name}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    {getActiveTemplate(MediaType.VIDEO_FRAME)?.description}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Last modified: {formatDate(getActiveTemplate(MediaType.VIDEO_FRAME)?.modified_date || '')}
                  </Typography>
                </CardContent>
              </Card>
            )}
          </Stack>
        </Box>

        {/* Configuration Info */}
        {configuration && (
          <Alert severity="info">
            <Typography variant="body2">
              <strong>Configuration last updated:</strong> {formatDate(configuration.last_updated)} by {configuration.updated_by}
            </Typography>
          </Alert>
        )}

        {/* Quick Actions */}
        <Box>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Need to create custom prompts or modify existing ones? Use the Prompts tab to manage your prompt library.
          </Typography>
        </Box>
      </Stack>
    </Box>
  );
};

export default PromptConfigurationComponent;
