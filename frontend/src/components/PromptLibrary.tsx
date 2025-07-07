/**
 * Prompt library component for managing prompt templates.
 */

import React, { useState } from 'react';
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  CardActions,
  Grid,
  Chip,
  IconButton,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Stack,
  FormControl,
  InputLabel,
  Select,
  Tooltip,
  CircularProgress,
} from '@mui/material';
import {
  Add as AddIcon,
  MoreVert as MoreVertIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  CheckCircle as ActiveIcon,
  RadioButtonUnchecked as InactiveIcon,
  Star as DefaultIcon,
  Person as UserIcon,
  Computer as SystemIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

import { promptAPI } from '../services/api';
import {
  PromptTemplate,
  MediaType,
  PromptAuthor,
  getMediaTypeLabel,
  getAuthorLabel,
  formatDate,
} from '../types/prompts';
import PromptEditor from './PromptEditor';

const PromptLibrary: React.FC = () => {
  const queryClient = useQueryClient();

  // State
  const [selectedTemplate, setSelectedTemplate] = useState<PromptTemplate | null>(null);
  const [editorOpen, setEditorOpen] = useState(false);
  const [editorMode, setEditorMode] = useState<'create' | 'edit'>('create');
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [templateToDelete, setTemplateToDelete] = useState<PromptTemplate | null>(null);
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
  const [filterMediaType, setFilterMediaType] = useState<MediaType | 'all'>('all');

  // Queries
  const { data: templates = [], isLoading, error } = useQuery({
    queryKey: ['promptTemplates', filterMediaType],
    queryFn: () => promptAPI.listTemplates(filterMediaType === 'all' ? undefined : filterMediaType),
  });

  // Mutations
  const deleteMutation = useMutation({
    mutationFn: promptAPI.deleteTemplate,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['promptTemplates'] });
      queryClient.invalidateQueries({ queryKey: ['promptConfiguration'] });
      setDeleteDialogOpen(false);
      setTemplateToDelete(null);
    },
  });

  const handleCreateNew = () => {
    setSelectedTemplate(null);
    setEditorMode('create');
    setEditorOpen(true);
  };

  const handleEdit = (template: PromptTemplate) => {
    setSelectedTemplate(template);
    setEditorMode('edit');
    setEditorOpen(true);
    setMenuAnchor(null);
  };

  const handleDelete = (template: PromptTemplate) => {
    setTemplateToDelete(template);
    setDeleteDialogOpen(true);
    setMenuAnchor(null);
  };

  const confirmDelete = async () => {
    if (templateToDelete) {
      await deleteMutation.mutateAsync(templateToDelete.template_id);
    }
  };

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, template: PromptTemplate) => {
    event.stopPropagation();
    setSelectedTemplate(template);
    setMenuAnchor(event.currentTarget);
  };

  const handleMenuClose = () => {
    setMenuAnchor(null);
    setSelectedTemplate(null);
  };

  const filteredTemplates = templates.filter(template => {
    if (filterMediaType === 'all') return true;
    return template.media_type === filterMediaType;
  });

  // Group templates by media type for better organization
  const groupedTemplates = filteredTemplates.reduce((acc, template) => {
    const key = template.media_type;
    if (!acc[key]) acc[key] = [];
    acc[key].push(template);
    return acc;
  }, {} as Record<MediaType, PromptTemplate[]>);

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        Failed to load prompt templates. Please try again.
      </Alert>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5">
          Prompt Library
        </Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={handleCreateNew}
        >
          Create New Prompt
        </Button>
      </Box>

      {/* Filter */}
      <Box sx={{ mb: 3 }}>
        <FormControl size="small" sx={{ minWidth: 200 }}>
          <InputLabel>Filter by Media Type</InputLabel>
          <Select
            value={filterMediaType}
            onChange={(e) => setFilterMediaType(e.target.value as MediaType | 'all')}
            label="Filter by Media Type"
          >
            <MenuItem value="all">All Types</MenuItem>
            <MenuItem value={MediaType.IMAGE}>{getMediaTypeLabel(MediaType.IMAGE)}</MenuItem>
            <MenuItem value={MediaType.VIDEO_FRAME}>{getMediaTypeLabel(MediaType.VIDEO_FRAME)}</MenuItem>
          </Select>
        </FormControl>
      </Box>

      {/* Templates Grid */}
      {filteredTemplates.length === 0 ? (
        <Alert severity="info">
          No prompt templates found. Create your first custom prompt to get started.
        </Alert>
      ) : (
        <Grid container spacing={2}>
          {filteredTemplates.map((template) => (
            <Grid item xs={12} md={6} lg={4} key={template.template_id}>
              <Card
                sx={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  border: template.is_active ? 2 : 1,
                  borderColor: template.is_active ? 'primary.main' : 'divider',
                }}
              >
                <CardContent sx={{ flex: 1 }}>
                  {/* Header */}
                  <Box sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'flex-start',
                    mb: 1,
                    minWidth: 0  // Allow flex items to shrink below their content size
                  }}>
                    <Box sx={{
                      flex: 1,
                      minWidth: 0,  // Allow text to truncate properly
                      mr: 1  // Add margin to ensure space between text and button
                    }}>
                      <Typography variant="h6" component="div" noWrap>
                        {template.name}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                        {template.description}
                      </Typography>
                    </Box>
                    <IconButton
                      size="small"
                      onClick={(e) => handleMenuOpen(e, template)}
                      disabled={template.is_default && template.author === PromptAuthor.SYSTEM}
                      sx={{ flexShrink: 0 }}  // Prevent the button from shrinking
                    >
                      <MoreVertIcon />
                    </IconButton>
                  </Box>

                  {/* Chips */}
                  <Stack direction="row" spacing={1} sx={{ mb: 2, flexWrap: 'wrap', gap: 0.5 }}>
                    <Chip
                      label={getMediaTypeLabel(template.media_type)}
                      size="small"
                      variant="outlined"
                    />
                    <Chip
                      label={getAuthorLabel(template.author)}
                      size="small"
                      icon={template.author === PromptAuthor.SYSTEM ? <SystemIcon /> : <UserIcon />}
                      color={template.author === PromptAuthor.SYSTEM ? 'primary' : 'default'}
                    />
                    {template.is_default && (
                      <Chip
                        label="Default"
                        size="small"
                        icon={<DefaultIcon />}
                        color="warning"
                      />
                    )}
                    {template.is_active && (
                      <Chip
                        label="Active"
                        size="small"
                        icon={<ActiveIcon />}
                        color="success"
                      />
                    )}
                  </Stack>

                  {/* Prompt Preview */}
                  <Typography
                    variant="body2"
                    color="text.secondary"
                    sx={{
                      display: '-webkit-box',
                      WebkitLineClamp: 3,
                      WebkitBoxOrient: 'vertical',
                      overflow: 'hidden',
                      fontFamily: 'monospace',
                      fontSize: '0.75rem',
                      backgroundColor: 'grey.50',
                      p: 1,
                      borderRadius: 1,
                    }}
                  >
                    {template.prompt_text}
                  </Typography>
                </CardContent>

                <CardActions sx={{ justifyContent: 'space-between', px: 2, pb: 2 }}>
                  <Typography variant="caption" color="text.secondary">
                    v{template.version} • {formatDate(template.modified_date)}
                  </Typography>
                  <Box>
                    {template.is_active ? (
                      <Tooltip title="Currently Active">
                        <ActiveIcon color="success" fontSize="small" />
                      </Tooltip>
                    ) : (
                      <InactiveIcon color="disabled" fontSize="small" />
                    )}
                  </Box>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Context Menu */}
      <Menu
        anchorEl={menuAnchor}
        open={Boolean(menuAnchor)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => selectedTemplate && handleEdit(selectedTemplate)}>
          <ListItemIcon>
            <EditIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Edit</ListItemText>
        </MenuItem>
        {selectedTemplate && !selectedTemplate.is_default && (
          <MenuItem onClick={() => selectedTemplate && handleDelete(selectedTemplate)}>
            <ListItemIcon>
              <DeleteIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText>Delete</ListItemText>
          </MenuItem>
        )}
      </Menu>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Delete Prompt Template</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete the prompt template "{templateToDelete?.name}"?
            This action cannot be undone.
          </Typography>
          {templateToDelete?.is_active && (
            <Alert severity="warning" sx={{ mt: 2 }}>
              This template is currently active. Deleting it may affect media analysis.
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={confirmDelete}
            color="error"
            variant="contained"
            disabled={deleteMutation.isPending}
          >
            {deleteMutation.isPending ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Prompt Editor */}
      <PromptEditor
        open={editorOpen}
        onClose={() => setEditorOpen(false)}
        template={selectedTemplate || undefined}
        mode={editorMode}
      />
    </Box>
  );
};

export default PromptLibrary;
