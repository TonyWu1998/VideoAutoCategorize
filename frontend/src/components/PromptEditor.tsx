/**
 * Prompt editor component for creating and editing LLM prompts.
 */

import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Box,
  Typography,
  Alert,
  Chip,
  Stack,
  Paper,
  Divider,
  CircularProgress,
} from '@mui/material';
import {
  Save as SaveIcon,
  Close as CloseIcon,
  CheckCircle as ValidIcon,
  Error as ErrorIcon,
} from '@mui/icons-material';
import { useMutation, useQueryClient } from '@tanstack/react-query';

import { promptAPI } from '../services/api';
import {
  PromptTemplate,
  PromptTemplateRequest,
  MediaType,
  PromptValidationResponse,
  getMediaTypeLabel,
} from '../types/prompts';

interface PromptEditorProps {
  open: boolean;
  onClose: () => void;
  template?: PromptTemplate;
  mode: 'create' | 'edit';
}

const PromptEditor: React.FC<PromptEditorProps> = ({
  open,
  onClose,
  template,
  mode,
}) => {
  const queryClient = useQueryClient();

  // Form state
  const [formData, setFormData] = useState<PromptTemplateRequest>({
    name: '',
    description: '',
    media_type: MediaType.IMAGE,
    prompt_text: '',
    version: '1.0',
  });

  // Validation state
  const [validation, setValidation] = useState<PromptValidationResponse | null>(null);
  const [validationLoading, setValidationLoading] = useState(false);

  // Error state
  const [error, setError] = useState<string | null>(null);

  // Initialize form data when template changes
  useEffect(() => {
    if (template && mode === 'edit') {
      setFormData({
        name: template.name,
        description: template.description,
        media_type: template.media_type,
        prompt_text: template.prompt_text,
        version: template.version,
      });
    } else {
      setFormData({
        name: '',
        description: '',
        media_type: MediaType.IMAGE,
        prompt_text: '',
        version: '1.0',
      });
    }
    setValidation(null);
    setError(null);
  }, [template, mode, open]);

  // Create mutation
  const createMutation = useMutation({
    mutationFn: promptAPI.createTemplate,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['promptTemplates'] });
      queryClient.invalidateQueries({ queryKey: ['promptConfiguration'] });
      onClose();
    },
    onError: (error: any) => {
      setError(error.response?.data?.detail || 'Failed to create prompt template');
    },
  });

  // Update mutation
  const updateMutation = useMutation({
    mutationFn: ({ templateId, request }: { templateId: string; request: PromptTemplateRequest }) =>
      promptAPI.updateTemplate(templateId, request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['promptTemplates'] });
      queryClient.invalidateQueries({ queryKey: ['promptConfiguration'] });
      onClose();
    },
    onError: (error: any) => {
      setError(error.response?.data?.detail || 'Failed to update prompt template');
    },
  });

  // Validation mutation
  const validateMutation = useMutation({
    mutationFn: promptAPI.validatePrompt,
    onSuccess: (data) => {
      setValidation(data);
    },
    onError: (error: any) => {
      setError(error.response?.data?.detail || 'Failed to validate prompt');
    },
  });

  const handleInputChange = (field: keyof PromptTemplateRequest, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    setError(null);
    
    // Clear validation when prompt text changes
    if (field === 'prompt_text' || field === 'media_type') {
      setValidation(null);
    }
  };

  const handleValidate = async () => {
    if (!formData.prompt_text.trim()) {
      setError('Please enter prompt text to validate');
      return;
    }

    setValidationLoading(true);
    try {
      await validateMutation.mutateAsync({
        prompt_text: formData.prompt_text,
        media_type: formData.media_type,
      });
    } finally {
      setValidationLoading(false);
    }
  };

  const handleSave = async () => {
    if (!formData.name.trim()) {
      setError('Please enter a template name');
      return;
    }

    if (!formData.description.trim()) {
      setError('Please enter a template description');
      return;
    }

    if (!formData.prompt_text.trim()) {
      setError('Please enter prompt text');
      return;
    }

    try {
      if (mode === 'create') {
        await createMutation.mutateAsync(formData);
      } else if (template) {
        await updateMutation.mutateAsync({
          templateId: template.template_id,
          request: formData,
        });
      }
    } catch (error) {
      // Error handling is done in mutation callbacks
    }
  };

  const isLoading = createMutation.isPending || updateMutation.isPending;
  const canSave = formData.name.trim() && formData.description.trim() && formData.prompt_text.trim();

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: { height: '80vh' }
      }}
    >
      <DialogTitle>
        {mode === 'create' ? 'Create New Prompt Template' : 'Edit Prompt Template'}
      </DialogTitle>

      <DialogContent sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {/* Basic Information */}
        <Box>
          <Typography variant="h6" gutterBottom>
            Basic Information
          </Typography>
          
          <Stack spacing={2}>
            <TextField
              label="Template Name"
              value={formData.name}
              onChange={(e) => handleInputChange('name', e.target.value)}
              fullWidth
              required
              placeholder="e.g., Detailed Image Analysis"
            />

            <TextField
              label="Description"
              value={formData.description}
              onChange={(e) => handleInputChange('description', e.target.value)}
              fullWidth
              required
              multiline
              rows={2}
              placeholder="Describe what this prompt template does..."
            />

            <FormControl fullWidth required>
              <InputLabel>Media Type</InputLabel>
              <Select
                value={formData.media_type}
                onChange={(e) => handleInputChange('media_type', e.target.value)}
                label="Media Type"
              >
                <MenuItem value={MediaType.IMAGE}>
                  {getMediaTypeLabel(MediaType.IMAGE)}
                </MenuItem>
                <MenuItem value={MediaType.VIDEO_FRAME}>
                  {getMediaTypeLabel(MediaType.VIDEO_FRAME)}
                </MenuItem>
              </Select>
            </FormControl>

            <TextField
              label="Version"
              value={formData.version}
              onChange={(e) => handleInputChange('version', e.target.value)}
              fullWidth
              placeholder="1.0"
            />
          </Stack>
        </Box>

        <Divider />

        {/* Prompt Text */}
        <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
            <Typography variant="h6">
              Prompt Text
            </Typography>
            <Button
              variant="outlined"
              size="small"
              onClick={handleValidate}
              disabled={!formData.prompt_text.trim() || validationLoading}
              startIcon={validationLoading ? <CircularProgress size={16} /> : undefined}
            >
              Validate
            </Button>
          </Box>

          <TextField
            value={formData.prompt_text}
            onChange={(e) => handleInputChange('prompt_text', e.target.value)}
            fullWidth
            required
            multiline
            rows={8}
            placeholder="Enter your prompt text here..."
            sx={{ flex: 1 }}
          />

          {/* Validation Results */}
          {validation && (
            <Paper sx={{ mt: 2, p: 2 }}>
              <Stack spacing={1}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  {validation.is_valid ? (
                    <ValidIcon color="success" />
                  ) : (
                    <ErrorIcon color="error" />
                  )}
                  <Typography variant="subtitle2">
                    {validation.is_valid ? 'Valid Prompt' : 'Validation Issues'}
                  </Typography>
                  {validation.estimated_tokens && (
                    <Chip
                      label={`~${validation.estimated_tokens} tokens`}
                      size="small"
                      variant="outlined"
                    />
                  )}
                </Box>

                {validation.validation_errors.length > 0 && (
                  <Box>
                    <Typography variant="body2" color="error" gutterBottom>
                      Errors:
                    </Typography>
                    {validation.validation_errors.map((error, index) => (
                      <Alert key={index} severity="error" sx={{ mb: 1 }}>
                        {error}
                      </Alert>
                    ))}
                  </Box>
                )}

                {validation.suggestions.length > 0 && (
                  <Box>
                    <Typography variant="body2" color="info.main" gutterBottom>
                      Suggestions:
                    </Typography>
                    {validation.suggestions.map((suggestion, index) => (
                      <Alert key={index} severity="info" sx={{ mb: 1 }}>
                        {suggestion}
                      </Alert>
                    ))}
                  </Box>
                )}
              </Stack>
            </Paper>
          )}
        </Box>

        {/* Error Display */}
        {error && (
          <Alert severity="error" onClose={() => setError(null)}>
            {error}
          </Alert>
        )}
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose} startIcon={<CloseIcon />}>
          Cancel
        </Button>
        <Button
          onClick={handleSave}
          variant="contained"
          disabled={!canSave || isLoading}
          startIcon={isLoading ? <CircularProgress size={16} /> : <SaveIcon />}
        >
          {mode === 'create' ? 'Create' : 'Save Changes'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default PromptEditor;
