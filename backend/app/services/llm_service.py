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

logger = logging.getLogger(__name__)


class LLMService:
    """
    Service for AI-powered analysis of images and videos using Ollama.
    
    Provides methods for generating descriptions, tags, and embeddings
    for media files using local LLM models.
    """
    
    def __init__(self):
        """Initialize the LLM service with Ollama client."""
        try:
            self.client = ollama.Client(host=settings.OLLAMA_BASE_URL)
            self.model = settings.OLLAMA_MODEL
            self.timeout = settings.OLLAMA_TIMEOUT
            
            # Test connection
            self._test_connection()
            
            logger.info(f"LLM service initialized with model: {self.model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {e}")
            raise
    
    async def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze an image and return description, tags, and metadata.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            logger.debug(f"Analyzing image: {image_path}")
            start_time = time.time()
            
            # Prepare image for analysis
            image_base64 = await self._prepare_image(image_path)
            
            # Generate description and tags
            analysis_result = await self._analyze_with_llm(image_base64, "image")
            
            # Generate embedding
            embedding = await self._generate_embedding(analysis_result["description"])
            
            processing_time = time.time() - start_time
            logger.debug(f"Image analysis completed in {processing_time:.2f}s")
            
            return {
                "description": analysis_result["description"],
                "tags": analysis_result["tags"],
                "confidence": analysis_result["confidence"],
                "embedding": embedding,
                "processing_time": processing_time,
                "model_used": self.model
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze image {image_path}: {e}")
            return {
                "description": f"Error analyzing image: {str(e)}",
                "tags": [],
                "confidence": 0.0,
                "embedding": None,
                "processing_time": 0.0,
                "model_used": self.model
            }
    
    async def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze a video by extracting key frames and analyzing them.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            logger.debug(f"Analyzing video: {video_path}")
            start_time = time.time()
            
            # Extract key frames from video
            frames = await self._extract_video_frames(video_path)
            
            if not frames:
                raise ValueError("No frames could be extracted from video")
            
            # Analyze each frame
            frame_analyses = []
            for i, frame_base64 in enumerate(frames):
                try:
                    analysis = await self._analyze_with_llm(frame_base64, "video_frame")
                    frame_analyses.append(analysis)
                    logger.debug(f"Analyzed frame {i+1}/{len(frames)}")
                except Exception as e:
                    logger.warning(f"Failed to analyze frame {i}: {e}")
                    continue
            
            if not frame_analyses:
                raise ValueError("No frames could be analyzed")
            
            # Combine results from all frames
            combined_result = await self._combine_frame_analyses(frame_analyses)
            
            # Generate embedding from combined description
            embedding = await self._generate_embedding(combined_result["description"])
            
            processing_time = time.time() - start_time
            logger.debug(f"Video analysis completed in {processing_time:.2f}s")
            
            return {
                "description": combined_result["description"],
                "tags": combined_result["tags"],
                "confidence": combined_result["confidence"],
                "embedding": embedding,
                "processing_time": processing_time,
                "frames_analyzed": len(frame_analyses),
                "model_used": self.model
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze video {video_path}: {e}")
            return {
                "description": f"Error analyzing video: {str(e)}",
                "tags": [],
                "confidence": 0.0,
                "embedding": None,
                "processing_time": 0.0,
                "frames_analyzed": 0,
                "model_used": self.model
            }
    
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
            
            # Calculate frame extraction interval
            interval_seconds = settings.VIDEO_FRAME_INTERVAL
            max_frames = settings.MAX_VIDEO_FRAMES
            
            if duration > 0:
                # Extract frames at regular intervals
                frame_interval = max(1, int(fps * interval_seconds))
                frame_indices = list(range(0, total_frames, frame_interval))[:max_frames]
            else:
                # Fallback: extract first few frames
                frame_indices = list(range(min(max_frames, total_frames)))
            
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
    
    async def _analyze_with_llm(self, image_base64: str, media_type: str) -> Dict[str, Any]:
        """
        Analyze image with LLM to generate description and tags.
        
        Args:
            image_base64: Base64 encoded image
            media_type: Type of media being analyzed
            
        Returns:
            Dictionary with description, tags, and confidence
        """
        try:
            # Create analysis prompt
            if media_type == "video_frame":
                prompt = """Analyze this video frame and provide:
1. A detailed description (2-3 sentences) of what you see
2. Key objects, people, and activities present
3. Setting/location if identifiable
4. Mood/atmosphere

Respond in JSON format with keys: description, objects, setting, mood, tags"""
            else:
                prompt = """Analyze this image and provide:
1. A detailed description (2-3 sentences) of what you see
2. Key objects, people, and subjects present
3. Setting/location if identifiable
4. Mood/atmosphere
5. Colors and visual style

Respond in JSON format with keys: description, objects, setting, mood, colors, tags"""
            
            # Call Ollama API
            response = await asyncio.to_thread(
                self.client.generate,
                model=self.model,
                prompt=prompt,
                images=[image_base64],
                options={
                    "temperature": 0.3,  # Lower temperature for more consistent results
                    "top_p": 0.9,
                    "num_predict": 200
                }
            )
            
            # Parse response
            analysis_result = self._parse_llm_response(response['response'])
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {
                "description": "Analysis failed",
                "tags": [],
                "confidence": 0.0
            }
    
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
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response and extract structured information.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Structured analysis result
        """
        try:
            # Try to parse as JSON first
            if response.strip().startswith('{'):
                data = json.loads(response)
                
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
                
                return {
                    "description": description,
                    "tags": tags,
                    "confidence": 0.8  # Assume good confidence for structured response
                }
            
            # Fallback: parse unstructured response
            lines = response.strip().split('\n')
            description = ""
            tags = []
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith(('1.', '2.', '3.', '4.', '5.')):
                    if not description:
                        description = line
                    else:
                        # Extract potential tags from the line
                        words = line.replace(',', ' ').replace('.', ' ').split()
                        tags.extend([word.strip().lower() for word in words if len(word) > 2])
            
            # Clean tags
            tags = list(set(tags))[:10]  # Limit to 10 tags
            
            return {
                "description": description or "Image analysis completed",
                "tags": tags,
                "confidence": 0.6  # Lower confidence for unstructured response
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return {
                "description": response[:200] if response else "Analysis completed",
                "tags": [],
                "confidence": 0.3
            }
    
    async def _combine_frame_analyses(self, frame_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine analysis results from multiple video frames.
        
        Args:
            frame_analyses: List of frame analysis results
            
        Returns:
            Combined analysis result
        """
        if not frame_analyses:
            return {"description": "", "tags": [], "confidence": 0.0}
        
        # Combine descriptions
        descriptions = [analysis["description"] for analysis in frame_analyses if analysis["description"]]
        combined_description = " ".join(descriptions[:3])  # Use first 3 descriptions
        
        # Combine and count tags
        tag_counts = {}
        total_confidence = 0
        
        for analysis in frame_analyses:
            total_confidence += analysis.get("confidence", 0)
            for tag in analysis.get("tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Select most common tags
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        combined_tags = [tag for tag, count in sorted_tags[:15]]  # Top 15 tags
        
        # Calculate average confidence
        avg_confidence = total_confidence / len(frame_analyses) if frame_analyses else 0
        
        return {
            "description": combined_description,
            "tags": combined_tags,
            "confidence": avg_confidence
        }
    
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
