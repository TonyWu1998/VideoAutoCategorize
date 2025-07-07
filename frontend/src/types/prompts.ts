/**
 * TypeScript types for prompt management.
 */

export enum MediaType {
  IMAGE = 'image',
  VIDEO_FRAME = 'video_frame',
}

export enum PromptAuthor {
  SYSTEM = 'system',
  USER = 'user',
}

export interface PromptTemplate {
  template_id: string;
  name: string;
  description: string;
  media_type: MediaType;
  prompt_text: string;
  is_default: boolean;
  is_active: boolean;
  version: string;
  author: PromptAuthor;
  created_date: string;
  modified_date: string;
}

export interface PromptTemplateRequest {
  name: string;
  description: string;
  media_type: MediaType;
  prompt_text: string;
  version?: string;
}

export interface PromptConfiguration {
  active_image_prompt_id?: string;
  active_video_prompt_id?: string;
  active_image_prompt?: PromptTemplate;
  active_video_prompt?: PromptTemplate;
  last_updated: string;
  updated_by: string;
}

export interface PromptConfigurationRequest {
  active_image_prompt_id?: string;
  active_video_prompt_id?: string;
}

export interface PromptValidationRequest {
  prompt_text: string;
  media_type: MediaType;
}

export interface PromptValidationResponse {
  is_valid: boolean;
  validation_errors: string[];
  suggestions: string[];
  estimated_tokens?: number;
}

export interface PromptTestRequest {
  prompt_text: string;
  media_type: MediaType;
  sample_image_path?: string;
}

export interface PromptTestResponse {
  success: boolean;
  test_result?: any;
  error_message?: string;
  execution_time_ms?: number;
}

// Helper functions
export const getMediaTypeLabel = (mediaType: MediaType): string => {
  switch (mediaType) {
    case MediaType.IMAGE:
      return 'Images';
    case MediaType.VIDEO_FRAME:
      return 'Video Frames';
    default:
      return 'Unknown';
  }
};

export const getAuthorLabel = (author: PromptAuthor): string => {
  switch (author) {
    case PromptAuthor.SYSTEM:
      return 'System Default';
    case PromptAuthor.USER:
      return 'Custom';
    default:
      return 'Unknown';
  }
};

export const formatDate = (dateString: string): string => {
  return new Date(dateString).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};
