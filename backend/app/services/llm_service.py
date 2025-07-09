"""
LLM service for AI-powered media analysis using Ollama.
"""

import ollama
from typing import List, Dict, Any, Optional
import base64
import json
import logging
from PIL import Image
import io
import cv2
import numpy as np
from pathlib import Path
import asyncio
import time

from app.config import settings
from app.models.prompts import MediaType

logger = logging.getLogger(__name__)


class LLMService:
    """
    Service for AI-powered analysis of images and videos using Ollama.

    Provides methods for generating descriptions, tags, and embeddings
    for media files using local LLM models.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        """Implement singleton pattern to ensure single LLM service instance."""
        if cls._instance is None:
            logger.info("🔍 Creating NEW LLM service singleton instance")
            cls._instance = super(LLMService, cls).__new__(cls)
            cls._instance._initialized = False  # Ensure instance-level flag
        else:
            logger.info("🔍 Returning EXISTING LLM service singleton instance")
        return cls._instance

    def __init__(self):
        """Initialize the LLM service with Ollama client (only once)."""
        if hasattr(self, '_initialized') and self._initialized:
            logger.info("🔍 LLM service __init__ called but already initialized, skipping")
            return

        try:
            logger.info("🔍 LLM service __init__ starting initialization...")
            logger.info(f"🔍 Settings object ID in __init__: {id(settings)}")
            logger.info(f"🔍 Initial OLLAMA_MODEL value: {settings.OLLAMA_MODEL}")

            self.client = ollama.Client(host=settings.OLLAMA_BASE_URL)
            # Don't cache the model - use property to get current setting
            self.timeout = settings.OLLAMA_TIMEOUT

            # Track the model and base URL we were initialized with for refresh detection
            self._initialized_model = settings.OLLAMA_MODEL
            self._initialized_base_url = settings.OLLAMA_BASE_URL

            # Initialize prompt manager (lazy loading to avoid circular imports)
            self._prompt_manager = None

            # Add cancellation support
            self._cancelled_jobs = set()

            # Test connection
            self._test_connection()

            logger.info(f"LLM service initialized with model: {self.model}")
            logger.info(f"🔍 LLM service initialization complete, _initialized = True")
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {e}")
            raise

    @classmethod
    def refresh_if_model_changed(cls):
        """
        Check if the model or base URL has changed and refresh the singleton if needed.
        This should be called when model settings are updated via API.
        """
        if cls._instance is not None and hasattr(cls._instance, '_initialized_model'):
            current_model = settings.OLLAMA_MODEL
            current_base_url = settings.OLLAMA_BASE_URL
            initialized_model = cls._instance._initialized_model
            initialized_base_url = getattr(cls._instance, '_initialized_base_url', 'http://localhost:11434')

            model_changed = current_model != initialized_model
            base_url_changed = current_base_url != initialized_base_url

            if model_changed or base_url_changed:
                changes = []
                if model_changed:
                    changes.append(f"model: '{initialized_model}' → '{current_model}'")
                if base_url_changed:
                    changes.append(f"base_url: '{initialized_base_url}' → '{current_base_url}'")

                logger.info(f"🔄 Configuration changed ({', '.join(changes)}) - refreshing LLM service")
                cls._instance._refresh_internal()
            else:
                logger.info(f"🔍 Configuration unchanged (model: '{current_model}', base_url: '{current_base_url}') - no refresh needed")
        else:
            logger.info("🔍 No LLM service instance to refresh")

    def _refresh_internal(self):
        """
        Internal method to refresh the LLM service configuration.
        Updates client settings and re-tests connection.
        """
        try:
            logger.info("🔄 Refreshing LLM service internal configuration...")

            # Update client configuration
            self.client = ollama.Client(host=settings.OLLAMA_BASE_URL)
            self.timeout = settings.OLLAMA_TIMEOUT

            # Update the tracked model and base URL
            old_model = self._initialized_model
            old_base_url = getattr(self, '_initialized_base_url', 'http://localhost:11434')
            self._initialized_model = settings.OLLAMA_MODEL
            self._initialized_base_url = settings.OLLAMA_BASE_URL

            # Test connection with new model
            self._test_connection()

            logger.info(f"🔄 LLM service refreshed: model '{old_model}' → '{self._initialized_model}', base_url '{old_base_url}' → '{self._initialized_base_url}'")

        except Exception as e:
            logger.error(f"Failed to refresh LLM service: {e}")
            # Don't raise exception to avoid breaking existing functionality
            logger.warning("Continuing with previous configuration")

    @property
    def prompt_manager(self):
        """Lazy load prompt manager to avoid circular imports."""
        if self._prompt_manager is None:
            from app.services.prompt_manager import PromptManager
            self._prompt_manager = PromptManager()
        return self._prompt_manager

    @property
    def model(self):
        """Get the current Ollama model from settings (dynamic, not cached)."""
        current_model = settings.OLLAMA_MODEL
        logger.info(f"🔍 LLM service model property accessed, returning: {current_model}")
        logger.info(f"🔍 Settings object ID: {id(settings)}, OLLAMA_MODEL ID: {id(settings.OLLAMA_MODEL)}")
        return current_model

    def cancel_job(self, job_id: str):
        """Cancel a specific job by ID."""
        self._cancelled_jobs.add(job_id)
        logger.info(f"🛑 LLM job {job_id} marked for cancellation")

    def is_job_cancelled(self, job_id: str) -> bool:
        """Check if a job has been cancelled."""
        return job_id in self._cancelled_jobs

    def clear_cancelled_job(self, job_id: str):
        """Clear a job from the cancelled set."""
        self._cancelled_jobs.discard(job_id)
    
    async def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze an image and return description, tags, and metadata.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing analysis results
        """
        try:
            logger.info(f"🖼️ Starting IMAGE analysis for: {image_path}")
            start_time = time.time()

            # Prepare image for analysis
            image_base64 = await self._prepare_image(image_path)
            logger.info(f"🖼️ Image prepared successfully for analysis")

            # Generate description and tags
            logger.info(f"🖼️ Calling LLM for image analysis")
            analysis_result = await self._analyze_with_llm(image_base64, "image")
            logger.info(f"🖼️ LLM analysis result: {analysis_result}")

            # Generate embedding
            logger.info(f"🖼️ Generating embedding for description: {analysis_result['description'][:100]}...")
            embedding = await self._generate_embedding(analysis_result["description"])
            logger.info(f"🖼️ Embedding generated successfully (length: {len(embedding) if embedding else 0})")

            processing_time = time.time() - start_time
            logger.info(f"🖼️ Image analysis completed successfully in {processing_time:.2f}s")

            result = {
                "description": analysis_result["description"],
                "tags": analysis_result["tags"],
                "confidence": analysis_result["confidence"],
                "embedding": embedding,
                "processing_time": processing_time,
                "model_used": self.model
            }
            logger.info(f"🖼️ Returning image analysis result with {len(result['tags'])} tags")
            return result

        except Exception as e:
            logger.error(f"🖼️ Failed to analyze image {image_path}: {e}")
            error_result = {
                "description": f"Error analyzing image: {str(e)}",
                "tags": [],
                "confidence": 0.0,
                "embedding": None,
                "processing_time": 0.0,
                "model_used": self.model
            }
            logger.error(f"🖼️ Returning error result: {error_result}")
            return error_result
    
    async def analyze_video(self, video_path: str, progress_callback=None,
                           store_frames: bool = True, video_file_id: Optional[str] = None,
                           job_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a video by extracting key frames and analyzing them.

        Args:
            video_path: Path to the video file
            progress_callback: Optional callback function for frame progress updates
                             Signature: callback(current_frame: int, total_frames: int, activity: str)
            store_frames: Whether to store individual frame analysis results (default: True)
            video_file_id: ID of the video file for frame storage (required if store_frames=True)

        Returns:
            Dictionary containing analysis results with frame storage information
        """
        try:
            logger.info(f"🎬 Starting video analysis for: {video_path}")
            start_time = time.time()

            # Extract key frames from video with timing information
            logger.info(f"🎬 Extracting frames from video: {video_path}")
            if store_frames:
                frame_data = await self._extract_video_frames_with_timing(video_path)
                frames = [frame['base64'] for frame in frame_data]
                logger.info(f"🎬 Extracted {len(frames)} frames with timing data from video")
            else:
                frames = await self._extract_video_frames(video_path)
                frame_data = [{'base64': frame, 'timestamp': i * settings.VIDEO_FRAME_INTERVAL, 'index': i}
                             for i, frame in enumerate(frames)]
                logger.info(f"🎬 Extracted {len(frames)} frames from video")

            if not frames:
                logger.error(f"🎬 No frames could be extracted from video: {video_path}")
                raise ValueError("No frames could be extracted from video")
            
            # Analyze each frame
            logger.info(f"🎬 Starting analysis of {len(frames)} frames")
            frame_analyses = []
            frame_documents = []

            # Report initial progress
            if progress_callback:
                progress_callback(0, len(frames), "Starting frame analysis")

            for i, frame_info in enumerate(frame_data):
                try:
                    # Check for cancellation
                    if job_id and self.is_job_cancelled(job_id):
                        logger.info(f"🛑 Video analysis cancelled for job {job_id}")
                        break

                    current_frame = i + 1
                    frame_base64 = frame_info['base64']
                    timestamp = frame_info['timestamp']
                    frame_index = frame_info['index']

                    logger.info(f"🎬 Analyzing frame {current_frame}/{len(frames)} at {timestamp:.1f}s")

                    # Update progress callback
                    if progress_callback:
                        progress_callback(i, len(frames), f"Analyzing frame {current_frame}")

                    analysis = await self._analyze_with_llm(frame_base64, "video_frame", job_id)
                    frame_analyses.append(analysis)

                    # Store individual frame if requested
                    if store_frames and video_file_id:
                        frame_document = await self._create_frame_document(
                            video_file_id, frame_index, timestamp, analysis
                        )
                        if frame_document:
                            frame_documents.append(frame_document)

                    logger.info(f"🎬 Successfully analyzed frame {current_frame}/{len(frames)}")

                    # Update progress callback after completion
                    if progress_callback:
                        progress_callback(current_frame, len(frames), f"Completed frame {current_frame}")

                except Exception as e:
                    logger.error(f"🎬 Failed to analyze frame {i}: {e}")
                    continue

            if not frame_analyses:
                logger.error(f"🎬 No frames could be analyzed for video: {video_path}")
                raise ValueError("No frames could be analyzed")
            
            # Combine results from all frames
            logger.info(f"🎬 Combining analysis from {len(frame_analyses)} frames")
            if progress_callback:
                progress_callback(len(frames), len(frames), "Combining frame analyses")

            combined_result = await self._combine_frame_analyses(frame_analyses)
            logger.info(f"🎬 Combined description: {combined_result['description'][:100]}...")

            # Generate embedding from combined description
            logger.info(f"🎬 Generating embedding for video description")
            if progress_callback:
                progress_callback(len(frames), len(frames), "Generating embedding")

            embedding = await self._generate_embedding(combined_result["description"])

            # Store frame documents if requested
            stored_frame_count = 0
            if store_frames and frame_documents:
                if progress_callback:
                    progress_callback(len(frames), len(frames), "Storing frame analysis data")

                stored_frame_count = await self._store_frame_documents(frame_documents)
                logger.info(f"🎬 Stored {stored_frame_count} frame documents")

            processing_time = time.time() - start_time
            logger.info(f"🎬 Video analysis completed successfully in {processing_time:.2f}s")

            result = {
                "description": combined_result["description"],
                "tags": combined_result["tags"],
                "confidence": combined_result["confidence"],
                "embedding": embedding,
                "processing_time": processing_time,
                "frames_analyzed": len(frame_analyses),
                "frames_stored": stored_frame_count,
                "frame_storage_enabled": store_frames,
                "model_used": self.model
            }
            logger.info(f"🎬 Returning video analysis result with {len(result['tags'])} tags")
            logger.info(f"🎬 Final video analysis result: {result}")
            return result

        except Exception as e:
            logger.error(f"🎬 Failed to analyze video {video_path}: {e}")
            error_result = {
                "description": f"Error analyzing video: {str(e)}",
                "tags": [],
                "confidence": 0.0,
                "embedding": None,
                "processing_time": 0.0,
                "frames_analyzed": 0,
                "model_used": self.model
            }
            logger.error(f"🎬 Returning video error result: {error_result}")
            return error_result
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        
        Args:
            query: Natural language search query
            
        Returns:
            Vector embedding for the query
        """
        try:
            embedding = await self._generate_embedding(query)
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise
    
    async def _prepare_image(self, image_path: str) -> str:
        """
        Prepare image for LLM analysis by resizing and encoding to base64.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if too large to save processing time and memory
                max_dimension = settings.MAX_IMAGE_DIMENSION
                if img.width > max_dimension or img.height > max_dimension:
                    img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                
                # Save to bytes buffer
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=settings.IMAGE_QUALITY)
                
                # Encode to base64
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                return img_base64
                
        except Exception as e:
            logger.error(f"Failed to prepare image {image_path}: {e}")
            raise
    
    async def _extract_video_frames(self, video_path: str) -> List[str]:
        """
        Extract key frames from video for analysis.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of base64 encoded frame images
        """
        try:
            frames = []
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Calculate frame extraction interval based on time
            interval_seconds = settings.VIDEO_FRAME_INTERVAL

            if duration > 0:
                # Extract frames at regular time intervals
                frame_interval = max(1, int(fps * interval_seconds))
                frame_indices = list(range(0, total_frames, frame_interval))

                # Log the calculated frame extraction details
                calculated_frames = len(frame_indices)
                logger.info(f"Video duration: {duration:.2f}s, FPS: {fps:.2f}, "
                           f"Interval: {interval_seconds}s, Calculated frames: {calculated_frames}")
            else:
                # Fallback: extract first frame only if duration cannot be determined
                frame_indices = [0] if total_frames > 0 else []
                logger.warning(f"Could not determine video duration, extracting first frame only")
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                
                # Resize if necessary
                max_dimension = settings.MAX_IMAGE_DIMENSION
                if pil_image.width > max_dimension or pil_image.height > max_dimension:
                    pil_image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                
                # Convert to base64
                buffer = io.BytesIO()
                pil_image.save(buffer, format='JPEG', quality=settings.IMAGE_QUALITY)
                frame_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                frames.append(frame_base64)
            
            cap.release()
            
            logger.debug(f"Extracted {len(frames)} frames from video")
            return frames
            
        except Exception as e:
            logger.error(f"Failed to extract frames from video {video_path}: {e}")
            return []
    
    async def _analyze_with_llm(self, image_base64: str, media_type: str, job_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze image with LLM to generate description and tags.

        Args:
            image_base64: Base64 encoded image
            media_type: Type of media being analyzed

        Returns:
            Dictionary with description, tags, and confidence
        """
        try:
            # Check for cancellation before starting
            if job_id and self.is_job_cancelled(job_id):
                logger.info(f"🛑 LLM analysis cancelled for job {job_id}")
                return self._create_fallback_result("Analysis cancelled")

            logger.info(f"🤖 Starting LLM analysis for media_type: {media_type}")

            # Get analysis prompt from prompt manager or use default
            prompt = await self._get_analysis_prompt(media_type)
            logger.info(f"🤖 Using prompt for {media_type} analysis")

            # Call Ollama API with options optimized for complex JSON responses
            logger.info(f"🤖 Calling Ollama API with model: {self.model}")
            response = await asyncio.to_thread(
                self.client.generate,
                model=self.model,
                prompt=prompt,
                images=[image_base64],
                options={
                    "temperature": 0.3,  # Conservative temperature
                    "top_p": 0.9,
                    "num_predict": 50000,  # Increased token limit for complex JSON responses
                    "repeat_penalty": 1.1  # Reduce repetition
                }
            )

            logger.info(f"🤖 Raw LLM response: {response.get('response', 'No response')[:50000]}...")

            # Validate response
            raw_response = response.get('response', '')
            if not raw_response or len(raw_response.strip()) < 10:
                logger.warning(f"🤖 LLM returned empty or very short response")
                return self._create_fallback_result("Empty response from LLM")

            # Parse response with enhanced error handling
            analysis_result = self._parse_llm_response(raw_response)
            logger.info(f"🤖 Parsed LLM analysis result: {analysis_result}")

            # Validate parsed result
            if not self._validate_analysis_result(analysis_result):
                logger.warning(f"🤖 Analysis result failed validation, using fallback")
                return self._create_fallback_result("Invalid analysis result structure")

            return analysis_result

        except Exception as e:
            logger.error(f"🤖 LLM analysis failed for media_type {media_type}: {e}")
            return self._create_fallback_result(f"LLM analysis exception: {str(e)}")

    async def _get_analysis_prompt(self, media_type: str) -> str:
        """
        Get the appropriate analysis prompt for the given media type.

        First tries to get a custom prompt from the prompt manager,
        falls back to default hardcoded prompts if none configured.

        Args:
            media_type: Type of media being analyzed

        Returns:
            Prompt text to use for analysis
        """
        try:
            # Map media_type to MediaType enum
            if media_type == "video_frame":
                prompt_media_type = MediaType.VIDEO_FRAME
            else:
                prompt_media_type = MediaType.IMAGE

            # Try to get custom prompt
            custom_prompt = await self.prompt_manager.get_active_prompt_text(prompt_media_type)

            if custom_prompt:
                logger.info(f"🤖 Using custom prompt for {media_type} (length: {len(custom_prompt)} chars)")
                logger.debug(f"🤖 Custom prompt preview: {custom_prompt[:100]}...")
                return custom_prompt

            # Fall back to default prompts
            logger.info(f"🤖 Using default prompt for {media_type} (no custom prompt configured)")
            default_prompt = self._get_default_prompt(media_type)
            logger.debug(f"🤖 Default prompt preview: {default_prompt[:100]}...")
            return default_prompt

        except Exception as e:
            logger.warning(f"Failed to get custom prompt for {media_type}, using default: {e}")
            return self._get_default_prompt(media_type)

    def _get_default_prompt(self, media_type: str) -> str:
        """
        Get the default hardcoded prompt for the given media type.

        Args:
            media_type: Type of media being analyzed

        Returns:
            Default prompt text
        """
        if media_type == "video_frame":
            return """Analyze this video frame and provide a JSON response with the following structure:

{
  "description": "A detailed 2-3 sentence description of what you see in the frame",
  "objects": ["list", "of", "key", "objects"],
  "setting": "location or environment description",
  "mood": "atmosphere or emotional tone",
  "tags": ["relevant", "descriptive", "keywords"]
}

IMPORTANT: Respond ONLY with valid JSON. Do not include markdown code blocks, explanations, or any text outside the JSON object."""
        else:
            return """Analyze this image and provide a JSON response with the following structure:

{
  "description": "A detailed 2-3 sentence description of what you see in the image",
  "objects": ["list", "of", "key", "objects", "people", "subjects"],
  "setting": "location or environment description",
  "mood": "atmosphere or emotional tone",
  "colors": "dominant colors and visual style",
  "tags": ["relevant", "descriptive", "keywords"]
}

IMPORTANT: Respond ONLY with valid JSON. Do not include markdown code blocks, explanations, or any text outside the JSON object."""
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate vector embedding for text using Ollama.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Vector embedding as list of floats
        """
        try:
            # Use Ollama's embeddings API with dedicated embedding model
            response = await asyncio.to_thread(
                self.client.embeddings,
                model=settings.OLLAMA_EMBEDDING_MODEL,
                prompt=text
            )

            return response['embedding']
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * 768  # Typical embedding dimension

    async def extract_tags(self, description: str) -> List[str]:
        """
        Extract relevant tags from an AI-generated description.

        Args:
            description: The AI-generated description text

        Returns:
            List of extracted tags
        """
        if not description:
            return []

        try:
            # Use LLM to extract structured tags from the description
            prompt = f"""
Extract 5-10 relevant tags from this image description. Return only the tags as a comma-separated list.

Description: {description}

Tags:"""

            response = await asyncio.to_thread(
                self.client.generate,
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "max_tokens": 100,
                    "stop": ["\n", ".", "Description:"]
                }
            )

            # Extract tags from response
            tags_text = response.get('response', '').strip()
            if tags_text:
                # Split by comma and clean up
                tags = [tag.strip().lower() for tag in tags_text.split(',')]
                tags = [tag for tag in tags if tag and len(tag) > 2 and len(tag) < 20]
                return tags[:10]  # Limit to 10 tags

            # Fallback: extract keywords from description
            words = description.lower().replace(',', ' ').replace('.', ' ').split()
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'this', 'that', 'these', 'those'}
            tags = [word for word in words if len(word) > 3 and word not in common_words]
            return list(set(tags))[:8]  # Limit to 8 tags

        except Exception as e:
            logger.warning(f"Failed to extract tags from description: {e}")
            # Simple fallback
            words = description.lower().split()
            return [word for word in words if len(word) > 4][:5]

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response and extract structured information.
        Handles various response formats including markdown-wrapped JSON.

        Args:
            response: Raw LLM response text

        Returns:
            Structured analysis result
        """
        try:
            logger.info(f"🔧 Parsing LLM response: {response[:100]}...")

            # Clean the response - remove markdown code blocks and extra text
            cleaned_response = self._clean_llm_response(response)
            logger.info(f"🔧 Cleaned response: {cleaned_response[:100]}...")

            # Try to parse as JSON first
            if cleaned_response.strip().startswith('{'):
                try:
                    data = json.loads(cleaned_response)
                    logger.info(f"🔧 Successfully parsed JSON: {data}")

                    # Check if this is a specialized adult content analysis response
                    if self._is_specialized_adult_analysis(data):
                        logger.info(f"🔧 Detected specialized adult analysis format")
                        return self._parse_specialized_adult_analysis(data)

                    # Handle standard format
                    description = data.get('description', '')
                    tags = []

                    # Extract tags from various fields
                    for field in ['objects', 'setting', 'mood', 'colors', 'tags']:
                        if field in data:
                            value = data[field]
                            if isinstance(value, list):
                                tags.extend(value)
                            elif isinstance(value, str) and value:
                                tags.append(value)

                    # Clean and deduplicate tags
                    tags = list(set([tag.strip().lower() for tag in tags if tag.strip()]))

                    result = {
                        "description": description,
                        "tags": tags,
                        "confidence": 0.8  # Assume good confidence for structured response
                    }
                    logger.info(f"🔧 Parsed structured result: {result}")
                    return result

                except json.JSONDecodeError as e:
                    logger.warning(f"🔧 JSON parsing failed: {e}")
                    logger.warning(f"🔧 Problematic JSON content: {cleaned_response[:200]}...")

                    # Try to fix common JSON issues before falling back
                    fixed_json = self._attempt_json_repair(cleaned_response)
                    if fixed_json:
                        try:
                            data = json.loads(fixed_json)
                            logger.info(f"🔧 Successfully parsed repaired JSON")

                            # Check if this is a specialized adult content analysis response
                            if self._is_specialized_adult_analysis(data):
                                logger.info(f"🔧 Detected specialized adult analysis format in repaired JSON")
                                return self._parse_specialized_adult_analysis(data)

                            # Handle standard format for repaired JSON
                            description = data.get('description', '')
                            tags = []
                            for field in ['objects', 'setting', 'mood', 'colors', 'tags']:
                                if field in data:
                                    value = data[field]
                                    if isinstance(value, list):
                                        tags.extend(value)
                                    elif isinstance(value, str) and value:
                                        tags.append(value)

                            tags = list(set([tag.strip().lower() for tag in tags if tag.strip()]))

                            return {
                                "description": description,
                                "tags": tags,
                                "confidence": 0.7  # Lower confidence for repaired JSON
                            }

                        except json.JSONDecodeError:
                            logger.warning(f"🔧 JSON repair attempt failed, falling back to unstructured parsing")

                    # Fall through to unstructured parsing

            # Fallback: parse unstructured response with enhanced extraction
            logger.info(f"🔧 Using enhanced unstructured parsing fallback")
            result = self._parse_unstructured_response_enhanced(cleaned_response, response)
            logger.info(f"🔧 Enhanced unstructured parsing result: {result}")
            return result

        except Exception as e:
            logger.error(f"🔧 Failed to parse LLM response: {e}")
            # Enhanced fallback that tries to extract meaningful content
            fallback_result = self._create_enhanced_fallback_result(response)
            logger.error(f"🔧 Using enhanced fallback result: {fallback_result}")
            return fallback_result

    def _clean_llm_response(self, response: str) -> str:
        """
        Clean LLM response by removing markdown code blocks and extra text.
        Enhanced to handle truncated responses and extract partial JSON.

        Args:
            response: Raw LLM response

        Returns:
            Cleaned response text
        """
        if not response:
            return ""

        logger.info(f"🔧 Cleaning response (length: {len(response)})")

        # Check for truncation indicators
        is_truncated = response.endswith('...') or response.endswith('…')
        if is_truncated:
            logger.warning(f"🔧 Response appears to be truncated")
            # Remove truncation indicators
            response = response.rstrip('.…').strip()

        # Remove common prefixes that LLMs add
        prefixes_to_remove = [
            "```json",
            "```",
            "Here's the JSON analysis:",
            "Okay, here's the JSON analysis of the image you provided:",
            "Here is the analysis:",
            "Analysis:",
            "JSON:",
            "Based on the image, here's the analysis:",
            "Looking at this image, here's the analysis:",
        ]

        cleaned = response.strip()

        # Remove prefixes
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()

        # Remove trailing markdown blocks
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        # Find JSON content between braces
        start_brace = cleaned.find('{')

        if start_brace != -1:
            # Look for the last complete closing brace
            end_brace = cleaned.rfind('}')

            if end_brace != -1 and end_brace > start_brace:
                json_content = cleaned[start_brace:end_brace + 1]
                logger.info(f"🔧 Extracted complete JSON content: {json_content[:100]}...")
                return json_content
            else:
                # Handle truncated JSON - try to find a reasonable stopping point
                logger.warning(f"🔧 JSON appears incomplete, attempting to extract partial content")

                # Try to find the last complete field or object
                partial_json = cleaned[start_brace:]

                # Look for common patterns that might indicate a good stopping point
                # Try to find the last complete quoted string or number
                last_quote = partial_json.rfind('"')
                last_comma = partial_json.rfind(',')
                last_bracket = partial_json.rfind(']')

                # Find the best stopping point
                stop_points = [i for i in [last_quote, last_comma, last_bracket] if i > 0]

                if stop_points:
                    best_stop = max(stop_points)
                    # Add closing brace if we found a reasonable stopping point
                    if best_stop > 10:  # Only if we have substantial content
                        truncated_json = partial_json[:best_stop + 1]
                        # Try to balance braces
                        open_braces = truncated_json.count('{')
                        close_braces = truncated_json.count('}')

                        # Add missing closing braces
                        missing_braces = open_braces - close_braces
                        if missing_braces > 0:
                            truncated_json += '}' * missing_braces

                        logger.info(f"🔧 Reconstructed truncated JSON: {truncated_json[:100]}...")
                        return truncated_json

        logger.info(f"🔧 No JSON braces found or unable to extract, returning cleaned text")
        return cleaned

    def _is_specialized_adult_analysis(self, data: Dict[str, Any]) -> bool:
        """
        Check if the JSON response is from a specialized adult content analysis prompt.

        Supports three specialized prompt formats:
        1. detailed_adult_analysis - Comprehensive analysis with participants, clothing, scenario
        2. fetish_focused_analysis - Fetish-specific analysis with categories and focus areas
        3. clothing_accessory_analysis - Detailed clothing and accessory inventory

        Args:
            data: Parsed JSON data

        Returns:
            True if this is a specialized adult analysis format
        """
        # Check for specialized adult analysis structure indicators
        detailed_analysis_indicators = [
            'participants',
            'clothing_accessories',
            'scenario_type',
            'visual_elements',
            'mood_atmosphere'
        ]

        fetish_analysis_indicators = [
            'fetish_categories',
            'clothing_focus',
            'scenario_elements',
            'power_dynamics'
        ]

        clothing_analysis_indicators = [
            'clothing_inventory',
            'accessories',
            'styling_elements',
            'special_focus_items'
        ]

        # Check for any specialized fields from any format
        all_indicators = detailed_analysis_indicators + fetish_analysis_indicators + clothing_analysis_indicators

        for indicator in all_indicators:
            if indicator in data:
                logger.info(f"🔧 Found specialized indicator: {indicator}")
                return True

        # Check for nested structures that indicate specialized format
        if 'participants' in data and isinstance(data['participants'], dict):
            return True

        # Check for complex clothing structures
        if 'clothing_focus' in data and isinstance(data['clothing_focus'], dict):
            return True

        if 'clothing_inventory' in data and isinstance(data['clothing_inventory'], dict):
            return True

        return False

    def _parse_specialized_adult_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse specialized adult content analysis JSON response.

        Handles three specialized prompt formats:
        1. detailed_adult_analysis - Comprehensive analysis with participants, clothing, scenario
        2. fetish_focused_analysis - Fetish-specific analysis with categories and focus areas
        3. clothing_accessory_analysis - Detailed clothing and accessory inventory

        Args:
            data: Parsed JSON data from specialized prompt

        Returns:
            Structured analysis result compatible with system expectations
        """
        try:
            logger.info(f"🔧 Parsing specialized adult analysis data")

            # Extract description
            description = data.get('description', '')

            # Initialize tags list
            tags = []

            # 1. DETAILED ADULT ANALYSIS FORMAT
            # Extract tags from direct tag sources
            tag_sources = [
                ('tags', lambda x: x if isinstance(x, list) else []),
                ('scenario_type', lambda x: x if isinstance(x, list) else [x] if isinstance(x, str) else []),
                ('fetish_categories', lambda x: x if isinstance(x, list) else []),
            ]

            for field, processor in tag_sources:
                if field in data:
                    processed_tags = processor(data[field])
                    tags.extend(processed_tags)

            # Extract from participants structure
            if 'participants' in data and isinstance(data['participants'], dict):
                participants = data['participants']
                for subfield in ['characteristics', 'roles']:
                    if subfield in participants and isinstance(participants[subfield], list):
                        tags.extend(participants[subfield])

            # Extract from clothing_accessories structure
            if 'clothing_accessories' in data and isinstance(data['clothing_accessories'], dict):
                clothing_acc = data['clothing_accessories']
                for subfield in ['notable_items', 'special_focus']:
                    if subfield in clothing_acc and isinstance(clothing_acc[subfield], list):
                        tags.extend(clothing_acc[subfield])
                if 'clothing_style' in clothing_acc and isinstance(clothing_acc['clothing_style'], str):
                    tags.append(clothing_acc['clothing_style'])

            # Extract from visual_elements structure
            if 'visual_elements' in data and isinstance(data['visual_elements'], dict):
                visual = data['visual_elements']
                for subfield in ['focus_areas']:
                    if subfield in visual and isinstance(visual[subfield], list):
                        tags.extend(visual[subfield])
                for subfield in ['lighting', 'camera_angle']:
                    if subfield in visual and isinstance(visual[subfield], str):
                        tags.append(visual[subfield])

            # 2. FETISH FOCUSED ANALYSIS FORMAT
            # Extract from clothing_focus structure
            if 'clothing_focus' in data and isinstance(data['clothing_focus'], dict):
                clothing_focus = data['clothing_focus']
                for category, items in clothing_focus.items():
                    if isinstance(items, list):
                        tags.extend(items)
                    elif isinstance(items, str) and items:
                        tags.append(items)

            # Extract from scenario_elements structure
            if 'scenario_elements' in data and isinstance(data['scenario_elements'], dict):
                scenario = data['scenario_elements']
                for subfield in ['activities', 'props_environment']:
                    if subfield in scenario and isinstance(scenario[subfield], list):
                        tags.extend(scenario[subfield])
                if 'power_dynamics' in scenario and isinstance(scenario['power_dynamics'], str):
                    tags.append(scenario['power_dynamics'])

            # Extract from visual_emphasis structure
            if 'visual_emphasis' in data and isinstance(data['visual_emphasis'], dict):
                visual_emp = data['visual_emphasis']
                for subfield in ['secondary_elements']:
                    if subfield in visual_emp and isinstance(visual_emp[subfield], list):
                        tags.extend(visual_emp[subfield])
                for subfield in ['primary_focus', 'camera_work']:
                    if subfield in visual_emp and isinstance(visual_emp[subfield], str):
                        tags.append(visual_emp[subfield])

            # 3. CLOTHING ACCESSORY ANALYSIS FORMAT
            # Extract from clothing_inventory structure
            if 'clothing_inventory' in data and isinstance(data['clothing_inventory'], dict):
                inventory = data['clothing_inventory']
                for category, items in inventory.items():
                    if isinstance(items, list):
                        tags.extend(items)
                    elif isinstance(items, str) and items:
                        tags.append(items)

            # Extract from accessories structure
            if 'accessories' in data and isinstance(data['accessories'], dict):
                accessories = data['accessories']
                for category, items in accessories.items():
                    if isinstance(items, list):
                        tags.extend(items)
                    elif isinstance(items, str) and items:
                        tags.append(items)

            # Extract from styling_elements structure
            if 'styling_elements' in data and isinstance(data['styling_elements'], dict):
                styling = data['styling_elements']
                for subfield in ['fit_style', 'condition']:
                    if subfield in styling and isinstance(styling[subfield], list):
                        tags.extend(styling[subfield])
                for subfield in ['overall_style', 'color_scheme']:
                    if subfield in styling and isinstance(styling[subfield], str):
                        tags.append(styling[subfield])

            # Extract from special_focus_items structure
            if 'special_focus_items' in data and isinstance(data['special_focus_items'], dict):
                for category, description in data['special_focus_items'].items():
                    if isinstance(description, str) and description and description.lower() not in ['none', 'n/a', '', 'not present']:
                        # Extract meaningful content from descriptions
                        if len(description) > 10:  # Substantial description
                            # Add the category as a tag if it's meaningful
                            if category in ['white_socks', 'uniforms', 'fetish_items', 'signature_pieces']:
                                tags.append(category.replace('_', ' '))
                            # Extract key terms from the description
                            import re
                            key_terms = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+(?:\s+[a-zA-Z]+)*', description)
                            for term in key_terms:
                                if len(term.strip()) > 2:
                                    tags.append(term.strip())

            # COMMON EXTRACTION FOR ALL FORMATS
            # Extract setting and mood information
            for field in ['setting', 'mood_atmosphere']:
                if field in data and isinstance(data[field], str) and data[field]:
                    tags.append(data[field])

            # Clean and process tags
            cleaned_tags = self._clean_and_process_tags(tags)

            result = {
                "description": description or "Specialized adult content analysis completed",
                "tags": cleaned_tags,
                "confidence": 0.9  # High confidence for specialized structured response
            }

            logger.info(f"🔧 Specialized parsing result: description length={len(result['description'])}, tags count={len(result['tags'])}")
            logger.debug(f"🔧 Extracted tags: {result['tags'][:10]}...")  # Log first 10 tags

            return result

        except Exception as e:
            logger.error(f"🔧 Failed to parse specialized adult analysis: {e}")
            # Fallback to basic parsing
            return {
                "description": data.get('description', 'Specialized analysis completed'),
                "tags": data.get('tags', []) if isinstance(data.get('tags'), list) else [],
                "confidence": 0.7
            }

    def _clean_and_process_tags(self, tags: List[str]) -> List[str]:
        """
        Clean and process tags extracted from specialized adult content analysis.

        Args:
            tags: Raw list of tags from various sources

        Returns:
            Cleaned and deduplicated list of tags
        """
        cleaned_tags = []

        # Common words to filter out
        common_words = {
            'the', 'and', 'or', 'with', 'for', 'in', 'on', 'at', 'to', 'of', 'a', 'an',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'description', 'analysis', 'image', 'video', 'frame', 'content', 'scene'
        }

        for tag in tags:
            if isinstance(tag, str):
                # Clean the tag
                clean_tag = tag.strip().lower()

                # Remove quotes and special characters but preserve Chinese characters
                clean_tag = clean_tag.replace('"', '').replace("'", "").replace('(', '').replace(')', '')

                # Skip empty tags, very short tags, and common words
                if (clean_tag and
                    len(clean_tag) > 2 and
                    clean_tag not in common_words and
                    not clean_tag.isdigit()):  # Skip pure numbers

                    # Handle Chinese/English mixed content
                    # Keep Chinese characters and meaningful English terms
                    import re
                    if re.search(r'[\u4e00-\u9fff]', clean_tag) or len(clean_tag) >= 3:
                        cleaned_tags.append(clean_tag)

        # Remove duplicates while preserving order
        seen = set()
        final_tags = []
        for tag in cleaned_tags:
            if tag not in seen:
                seen.add(tag)
                final_tags.append(tag)

        # Limit to reasonable number of tags (increased for specialized content)
        final_tags = final_tags[:30]

        return final_tags

    def _attempt_json_repair(self, json_str: str) -> Optional[str]:
        """
        Attempt to repair common JSON formatting issues.

        Args:
            json_str: Potentially malformed JSON string

        Returns:
            Repaired JSON string or None if repair failed
        """
        try:
            logger.info(f"🔧 Attempting JSON repair on: {json_str[:100]}...")

            # Common fixes for malformed JSON
            repaired = json_str.strip()

            # Handle truncated strings - find incomplete quoted strings and close them
            import re

            # Find the last quote and check if it's properly closed
            quote_positions = [m.start() for m in re.finditer(r'"', repaired)]
            if quote_positions and len(quote_positions) % 2 != 0:
                # Odd number of quotes means we have an unclosed string
                last_quote_pos = quote_positions[-1]
                # Find the end of the incomplete content
                remaining = repaired[last_quote_pos + 1:]

                # Look for a reasonable place to close the string
                # Find the last word or meaningful content
                truncation_patterns = [r'[^"]*?(?=\s*[,}\]]|$)', r'[^"]*?(?=\s*$)']

                for pattern in truncation_patterns:
                    match = re.search(pattern, remaining)
                    if match:
                        # Close the string at the end of meaningful content
                        close_pos = last_quote_pos + 1 + match.end()
                        repaired = repaired[:close_pos] + '"' + repaired[close_pos:]
                        break
                else:
                    # Fallback: just close the quote
                    repaired += '"'

            # Fix missing closing brackets/braces
            open_braces = repaired.count('{')
            close_braces = repaired.count('}')
            if open_braces > close_braces:
                repaired += '}' * (open_braces - close_braces)

            open_brackets = repaired.count('[')
            close_brackets = repaired.count(']')
            if open_brackets > close_brackets:
                repaired += ']' * (open_brackets - close_brackets)

            # Fix trailing commas before closing braces/brackets
            repaired = re.sub(r',\s*}', '}', repaired)
            repaired = re.sub(r',\s*]', ']', repaired)

            # Fix common Chinese/English mixed content issues
            repaired = repaired.replace('，', ',')  # Chinese comma to English comma
            repaired = repaired.replace('：', ':')  # Chinese colon to English colon

            # Handle incomplete array elements - if we end with a comma and incomplete content
            # Remove trailing incomplete elements
            repaired = re.sub(r',\s*"[^"]*$', '', repaired)

            # Try to validate the repaired JSON
            json.loads(repaired)
            logger.info(f"🔧 JSON repair successful")
            return repaired

        except Exception as e:
            logger.warning(f"🔧 JSON repair failed: {e}")
            return None

    def _parse_unstructured_response_enhanced(self, cleaned_response: str, original_response: str) -> Dict[str, Any]:
        """
        Enhanced unstructured response parsing with better content extraction.

        Args:
            cleaned_response: Cleaned response text
            original_response: Original raw response

        Returns:
            Structured analysis result
        """
        logger.info(f"🔧 Enhanced unstructured parsing")

        # Try to extract meaningful content from the response
        description = ""
        tags = []

        # Look for description-like content
        lines = cleaned_response.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 20:  # Substantial content
                # Skip lines that look like field names or JSON structure
                if not any(indicator in line.lower() for indicator in ['description', 'tags', 'participants', 'clothing']):
                    if not description:
                        description = line
                        break

        # If no description found in cleaned response, try original
        if not description:
            orig_lines = original_response.split('\n')
            for line in orig_lines:
                line = line.strip()
                if line and len(line) > 20 and not line.startswith('{') and not line.startswith('"'):
                    description = line
                    break

        # Extract potential tags using multiple strategies
        text_to_analyze = cleaned_response + " " + original_response

        # Strategy 1: Look for quoted strings (potential tags)
        import re
        quoted_strings = re.findall(r'"([^"]+)"', text_to_analyze)
        for string in quoted_strings:
            if len(string) > 2 and len(string) < 30:  # Reasonable tag length
                tags.append(string.lower())

        # Strategy 2: Look for comma-separated values
        comma_separated = re.findall(r'[,，]\s*([^,，\n\r"]+)', text_to_analyze)
        for item in comma_separated:
            clean_item = item.strip().lower()
            if len(clean_item) > 2 and len(clean_item) < 30:
                tags.append(clean_item)

        # Strategy 3: Extract Chinese/English mixed content
        chinese_english_pattern = r'[\u4e00-\u9fff]+|[a-zA-Z]+(?:\s+[a-zA-Z]+)*'
        matches = re.findall(chinese_english_pattern, text_to_analyze)
        for match in matches:
            clean_match = match.strip().lower()
            if len(clean_match) > 2 and len(clean_match) < 30:
                tags.append(clean_match)

        # Clean and deduplicate tags
        cleaned_tags = []
        common_words = {'the', 'and', 'or', 'with', 'for', 'in', 'on', 'at', 'description', 'tags', 'analysis', 'image', 'video'}

        for tag in tags:
            clean_tag = tag.strip().lower()
            if clean_tag and clean_tag not in common_words and len(clean_tag) > 2:
                cleaned_tags.append(clean_tag)

        # Remove duplicates while preserving order
        final_tags = []
        seen = set()
        for tag in cleaned_tags:
            if tag not in seen:
                seen.add(tag)
                final_tags.append(tag)

        # Limit tags
        final_tags = final_tags[:15]

        return {
            "description": description or "Content analysis completed",
            "tags": final_tags,
            "confidence": 0.5  # Medium confidence for enhanced unstructured parsing
        }

    def _create_enhanced_fallback_result(self, response: str) -> Dict[str, Any]:
        """
        Create an enhanced fallback result when all parsing fails.

        Args:
            response: Original response text

        Returns:
            Basic analysis result with extracted content
        """
        # Try to extract any meaningful text
        if response:
            # Clean up the response for description
            lines = response.split('\n')
            meaningful_lines = [line.strip() for line in lines if line.strip() and len(line.strip()) > 10]

            if meaningful_lines:
                description = meaningful_lines[0][:200]  # First meaningful line, truncated
            else:
                description = response[:100].strip()
        else:
            description = "Analysis completed"

        return {
            "description": description,
            "tags": [],
            "confidence": 0.2  # Low confidence for fallback
        }

    def _parse_unstructured_response(self, response: str) -> Dict[str, Any]:
        """
        Parse unstructured LLM response as fallback.

        Args:
            response: Cleaned response text

        Returns:
            Structured analysis result
        """
        lines = response.strip().split('\n')
        description = ""
        tags = []

        for line in lines:
            line = line.strip()
            if line and not line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '*')):
                if not description and len(line) > 10:  # Use first substantial line as description
                    description = line
                else:
                    # Extract potential tags from the line
                    words = line.replace(',', ' ').replace('.', ' ').replace('"', ' ').split()
                    potential_tags = [word.strip().lower() for word in words if len(word) > 2]
                    tags.extend(potential_tags)

        # Clean tags - remove common words and duplicates
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'this', 'that', 'these', 'those', 'you', 'see', 'can', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'have', 'has', 'had', 'been', 'being', 'do', 'does', 'did', 'done', 'get', 'got', 'make', 'made', 'take', 'took', 'come', 'came', 'go', 'went', 'know', 'knew', 'think', 'thought', 'say', 'said', 'tell', 'told', 'give', 'gave', 'find', 'found', 'use', 'used', 'work', 'worked', 'call', 'called', 'try', 'tried', 'ask', 'asked', 'need', 'needed', 'feel', 'felt', 'become', 'became', 'leave', 'left', 'put', 'keep', 'kept', 'let', 'seem', 'seemed', 'turn', 'turned', 'start', 'started', 'show', 'showed', 'hear', 'heard', 'play', 'played', 'run', 'ran', 'move', 'moved', 'live', 'lived', 'believe', 'believed', 'hold', 'held', 'bring', 'brought', 'happen', 'happened', 'write', 'wrote', 'provide', 'provided', 'sit', 'sat', 'stand', 'stood', 'lose', 'lost', 'pay', 'paid', 'meet', 'met', 'include', 'included', 'continue', 'continued', 'set', 'follow', 'followed', 'stop', 'stopped', 'create', 'created', 'speak', 'spoke', 'read', 'allow', 'allowed', 'add', 'added', 'spend', 'spent', 'grow', 'grew', 'open', 'opened', 'walk', 'walked', 'win', 'won', 'offer', 'offered', 'remember', 'remembered', 'love', 'loved', 'consider', 'considered', 'appear', 'appeared', 'buy', 'bought', 'wait', 'waited', 'serve', 'served', 'die', 'died', 'send', 'sent', 'expect', 'expected', 'build', 'built', 'stay', 'stayed', 'fall', 'fell', 'cut', 'reach', 'reached', 'kill', 'killed', 'remain', 'remained'}

        cleaned_tags = []
        for tag in tags:
            if tag not in common_words and len(tag) > 2 and len(tag) < 20:
                cleaned_tags.append(tag)

        # Remove duplicates and limit
        cleaned_tags = list(set(cleaned_tags))[:10]

        return {
            "description": description or "Analysis completed",
            "tags": cleaned_tags,
            "confidence": 0.6  # Lower confidence for unstructured response
        }

    def _validate_analysis_result(self, result: Dict[str, Any]) -> bool:
        """
        Validate that analysis result has required structure.

        Args:
            result: Analysis result to validate

        Returns:
            True if valid, False otherwise
        """
        required_keys = ['description', 'tags', 'confidence']

        # Check required keys exist
        for key in required_keys:
            if key not in result:
                logger.warning(f"🔧 Missing required key: {key}")
                return False

        # Check data types
        if not isinstance(result['description'], str):
            logger.warning(f"🔧 Description is not a string: {type(result['description'])}")
            return False

        if not isinstance(result['tags'], list):
            logger.warning(f"🔧 Tags is not a list: {type(result['tags'])}")
            return False

        if not isinstance(result['confidence'], (int, float)):
            logger.warning(f"🔧 Confidence is not a number: {type(result['confidence'])}")
            return False

        # Check reasonable values
        if len(result['description']) < 5:
            logger.warning(f"🔧 Description too short: {len(result['description'])}")
            return False

        if result['confidence'] < 0 or result['confidence'] > 1:
            logger.warning(f"🔧 Confidence out of range: {result['confidence']}")
            return False

        logger.info(f"🔧 Analysis result validation passed")
        return True

    def _create_fallback_result(self, error_message: str) -> Dict[str, Any]:
        """
        Create a fallback analysis result for error cases.

        Args:
            error_message: Description of the error

        Returns:
            Fallback analysis result
        """
        fallback_result = {
            "description": f"Analysis incomplete: {error_message}",
            "tags": ["analysis-error", "needs-retry"],
            "confidence": 0.1
        }
        logger.info(f"🔧 Created fallback result: {fallback_result}")
        return fallback_result

    async def _combine_frame_analyses(self, frame_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enhanced combination of analysis results from multiple video frames.

        Uses intelligent aggregation to preserve important information while
        creating a coherent video-level description.

        Args:
            frame_analyses: List of frame analysis results

        Returns:
            Enhanced combined analysis result with frame statistics
        """
        if not frame_analyses:
            return {"description": "", "tags": [], "confidence": 0.0}

        # Enhanced description combination
        descriptions = [analysis["description"] for analysis in frame_analyses if analysis["description"]]

        if len(descriptions) <= 3:
            # Use all descriptions for short videos
            combined_description = " ".join(descriptions)
        else:
            # For longer videos, use strategic sampling
            # Take first, middle, and last descriptions, plus highest confidence ones
            confidences = [analysis.get("confidence", 0) for analysis in frame_analyses]
            high_conf_indices = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)[:2]

            selected_descriptions = []
            selected_descriptions.append(descriptions[0])  # First frame
            if len(descriptions) > 2:
                selected_descriptions.append(descriptions[len(descriptions)//2])  # Middle frame
                selected_descriptions.append(descriptions[-1])  # Last frame

            # Add high-confidence descriptions if not already included
            for idx in high_conf_indices:
                if idx < len(descriptions) and descriptions[idx] not in selected_descriptions:
                    selected_descriptions.append(descriptions[idx])
                    if len(selected_descriptions) >= 5:  # Limit to 5 descriptions
                        break

            combined_description = " ".join(selected_descriptions)

        # Enhanced tag combination with frequency weighting
        tag_counts = {}
        tag_confidences = {}
        total_confidence = 0
        confidence_scores = []

        for analysis in frame_analyses:
            frame_confidence = analysis.get("confidence", 0)
            total_confidence += frame_confidence
            confidence_scores.append(frame_confidence)

            # Weight tags by frame confidence
            for tag in analysis.get("tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
                # Track average confidence for each tag
                if tag not in tag_confidences:
                    tag_confidences[tag] = []
                tag_confidences[tag].append(frame_confidence)

        # Calculate tag scores (frequency * average confidence)
        tag_scores = {}
        for tag, count in tag_counts.items():
            avg_tag_confidence = sum(tag_confidences[tag]) / len(tag_confidences[tag])
            frequency_score = count / len(frame_analyses)  # Normalize by total frames
            tag_scores[tag] = frequency_score * avg_tag_confidence

        # Select top tags based on combined score
        sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
        combined_tags = [tag for tag, score in sorted_tags[:20]]  # Increased to 20 tags

        # Enhanced confidence calculation
        if confidence_scores:
            # Use weighted average with higher weight for higher confidence frames
            weights = [max(0.1, score) for score in confidence_scores]  # Minimum weight 0.1
            weighted_confidence = sum(score * weight for score, weight in zip(confidence_scores, weights))
            total_weight = sum(weights)
            avg_confidence = weighted_confidence / total_weight if total_weight > 0 else 0
        else:
            avg_confidence = 0

        # Add frame analysis statistics
        frame_stats = {
            "total_frames": len(frame_analyses),
            "frames_with_descriptions": len(descriptions),
            "unique_tags": len(tag_counts),
            "confidence_range": {
                "min": min(confidence_scores) if confidence_scores else 0,
                "max": max(confidence_scores) if confidence_scores else 0,
                "std": self._calculate_std(confidence_scores) if len(confidence_scores) > 1 else 0
            }
        }

        return {
            "description": combined_description,
            "tags": combined_tags,
            "confidence": avg_confidence,
            "frame_statistics": frame_stats
        }

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation of a list of values."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _test_connection(self) -> None:
        """Test connection to Ollama service."""
        try:
            models = self.client.list()
            # Handle different response formats
            if hasattr(models, 'models'):
                model_list = models.models
            elif isinstance(models, dict) and 'models' in models:
                model_list = models['models']
            else:
                model_list = models if isinstance(models, list) else []

            model_names = []
            for model in model_list:
                if hasattr(model, 'model'):
                    model_names.append(model.model)
                elif hasattr(model, 'name'):
                    model_names.append(model.name)
                elif isinstance(model, dict) and 'model' in model:
                    model_names.append(model['model'])
                elif isinstance(model, dict) and 'name' in model:
                    model_names.append(model['name'])
                elif isinstance(model, str):
                    model_names.append(model)

            if self.model not in model_names:
                logger.warning(f"Model {self.model} not found. Available models: {model_names}")
                # Don't raise exception, as model might be pulled later

        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            # Don't raise exception to allow startup without Ollama
            logger.warning("Continuing without Ollama connection - AI features may not work")

    async def _extract_video_frames_with_timing(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Extract key frames from video with timing information for frame storage.

        Args:
            video_path: Path to the video file

        Returns:
            List of dictionaries with frame data: {'base64': str, 'timestamp': float, 'index': int}
        """
        try:
            frame_data = []
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            # Calculate frame extraction interval based on time
            interval_seconds = settings.VIDEO_FRAME_INTERVAL

            if duration > 0:
                # Extract frames at regular time intervals
                frame_interval = max(1, int(fps * interval_seconds))
                frame_indices = list(range(0, total_frames, frame_interval))

                # Log the calculated frame extraction details
                calculated_frames = len(frame_indices)
                logger.info(f"Video duration: {duration:.2f}s, FPS: {fps:.2f}, "
                           f"Interval: {interval_seconds}s, Calculated frames: {calculated_frames}")
            else:
                # Fallback: extract first frame only if duration cannot be determined
                frame_indices = [0] if total_frames > 0 else []
                logger.warning(f"Could not determine video duration, extracting first frame only")

            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    continue

                # Calculate timestamp
                timestamp = frame_idx / fps if fps > 0 else i * interval_seconds

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)

                # Resize if necessary
                max_dimension = settings.MAX_IMAGE_DIMENSION
                if pil_image.width > max_dimension or pil_image.height > max_dimension:
                    pil_image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

                # Convert to base64
                buffer = io.BytesIO()
                pil_image.save(buffer, format='JPEG', quality=settings.IMAGE_QUALITY)
                frame_base64 = base64.b64encode(buffer.getvalue()).decode()

                frame_data.append({
                    'base64': frame_base64,
                    'timestamp': timestamp,
                    'index': i
                })

            cap.release()

            logger.debug(f"Extracted {len(frame_data)} frames with timing data from video")
            return frame_data

        except Exception as e:
            logger.error(f"Failed to extract frames with timing from video {video_path}: {e}")
            return []

    async def _create_frame_document(self, video_file_id: str, frame_index: int,
                                   timestamp: float, analysis: Dict[str, Any]) -> Optional['VideoFrameDocument']:
        """
        Create a VideoFrameDocument from frame analysis results.

        Args:
            video_file_id: ID of the parent video
            frame_index: Index of the frame in the video
            timestamp: Timestamp of the frame in seconds
            analysis: Analysis results from LLM

        Returns:
            VideoFrameDocument or None if creation fails
        """
        try:
            from app.database.schemas import VideoFrameDocument
            from datetime import datetime, timezone

            # Generate frame ID
            frame_id = f"{video_file_id}_frame_{frame_index}"

            # Generate embedding for frame description
            embedding = None
            if analysis.get('description'):
                embedding = await self._generate_embedding(analysis['description'])

            # Create frame document
            frame_document = VideoFrameDocument(
                frame_id=frame_id,
                video_file_id=video_file_id,
                frame_index=frame_index,
                timestamp_seconds=timestamp,
                ai_description=analysis.get('description', ''),
                ai_tags=analysis.get('tags', []),
                ai_confidence=analysis.get('confidence'),
                frame_quality_score=analysis.get('confidence'),  # Use confidence as quality indicator
                extraction_method="time_interval",
                analyzed_date=datetime.now(timezone.utc),
                analysis_version="2.0",  # Frame-level storage version
                model_used=self.model,
                embedding=embedding
            )

            return frame_document

        except Exception as e:
            logger.error(f"Failed to create frame document for frame {frame_index}: {e}")
            return None

    async def _store_frame_documents(self, frame_documents: List['VideoFrameDocument']) -> int:
        """
        Store frame documents in the vector database.

        Args:
            frame_documents: List of VideoFrameDocument objects to store

        Returns:
            Number of successfully stored frame documents
        """
        try:
            from app.database.vector_db import VectorDatabase

            if not frame_documents:
                return 0

            # Initialize vector database
            vector_db = VectorDatabase()

            # Store frames in batch
            stored_frame_ids = vector_db.add_frames_batch(frame_documents)

            logger.info(f"Successfully stored {len(stored_frame_ids)} frame documents")
            return len(stored_frame_ids)

        except Exception as e:
            logger.error(f"Failed to store frame documents: {e}")
            return 0
