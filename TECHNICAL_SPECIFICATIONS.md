# Technical Specifications - Local Media Semantic Search

## Project Structure

```
VideoAutoCategorize/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application entry point
│   │   ├── config.py            # Configuration management
│   │   ├── models/              # Pydantic models
│   │   ├── services/            # Business logic
│   │   │   ├── indexing.py      # Media indexing service
│   │   │   ├── llm_service.py   # Ollama integration
│   │   │   ├── search.py        # Vector search service
│   │   │   └── file_service.py  # File management
│   │   ├── api/                 # API routes
│   │   │   ├── search.py
│   │   │   ├── indexing.py
│   │   │   └── media.py
│   │   └── database/            # Database operations
│   │       ├── vector_db.py     # ChromaDB operations
│   │       └── schemas.py       # Database schemas
│   ├── requirements.txt
│   ├── pyproject.toml          # Poetry configuration
│   └── tests/
├── frontend/
│   ├── src/
│   │   ├── components/          # React components
│   │   │   ├── SearchBar.tsx
│   │   │   ├── MediaGallery.tsx
│   │   │   ├── MediaViewer.tsx
│   │   │   └── SettingsPanel.tsx
│   │   ├── hooks/               # Custom React hooks
│   │   ├── services/            # API client services
│   │   ├── store/               # State management
│   │   ├── types/               # TypeScript definitions
│   │   ├── utils/               # Utility functions
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
├── docs/
│   ├── api/                     # API documentation
│   ├── user-guide/              # User documentation
│   └── development/             # Developer guides
├── scripts/
│   ├── setup.sh                # Environment setup
│   ├── install-ollama.sh       # Ollama installation
│   └── dev-start.sh            # Development startup
├── docker/                     # Docker configurations
├── .env.example
├── README.md
└── MVP_DEVELOPMENT_PLAN.md
```

## Backend Implementation Details

### 1. FastAPI Application Structure

```python
# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api import search, indexing, media
from app.config import settings

app = FastAPI(
    title="Media Semantic Search API",
    description="API for semantic search of local media files",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(indexing.router, prefix="/api/index", tags=["indexing"])
app.include_router(media.router, prefix="/api/media", tags=["media"])

# Serve media files
app.mount("/media", StaticFiles(directory=settings.MEDIA_ROOT), name="media")
```

### 2. Configuration Management

```python
# app/config.py
from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    # Database
    CHROMA_DB_PATH: str = "./data/chroma_db"
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma3:4b"
    
    # Media Processing
    SUPPORTED_IMAGE_FORMATS: List[str] = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]
    SUPPORTED_VIDEO_FORMATS: List[str] = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"]
    MAX_FILE_SIZE_MB: int = 500
    
    # API Configuration
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    API_V1_STR: str = "/api/v1"
    
    # Indexing
    BATCH_SIZE: int = 10
    MAX_CONCURRENT_PROCESSING: int = 4
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### 3. LLM Service Integration

```python
# app/services/llm_service.py
import ollama
from typing import List, Dict, Any
from app.config import settings
import base64
from PIL import Image
import io

class LLMService:
    def __init__(self):
        self.client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        self.model = settings.OLLAMA_MODEL
    
    async def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze image and return description and tags."""
        try:
            # Convert image to base64 for Ollama
            with Image.open(image_path) as img:
                # Resize if too large to save processing time
                if img.width > 1024 or img.height > 1024:
                    img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Prompt for comprehensive analysis
            prompt = """Analyze this image and provide:
            1. A detailed description (2-3 sentences)
            2. Key objects and subjects present
            3. Setting/location if identifiable
            4. Mood/atmosphere
            5. Colors and visual style
            
            Format as JSON with keys: description, objects, setting, mood, colors"""
            
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                images=[img_base64]
            )
            
            # Parse response and extract structured data
            return self._parse_analysis_response(response['response'])
            
        except Exception as e:
            return {
                "description": f"Error analyzing image: {str(e)}",
                "tags": [],
                "confidence": 0.0
            }
    
    async def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """Analyze video by extracting key frames."""
        # Implementation for video analysis
        # Extract frames at intervals, analyze each, combine results
        pass
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format."""
        # Implementation to parse and structure the response
        pass
```

### 4. Vector Database Operations

```python
# app/database/vector_db.py
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
from app.config import settings
import hashlib

class VectorDatabase:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_DB_PATH,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="media_embeddings",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_media(self, file_path: str, metadata: Dict[str, Any], 
                  description: str, embedding: List[float]) -> str:
        """Add media file to vector database."""
        file_id = self._generate_file_id(file_path)
        
        self.collection.add(
            ids=[file_id],
            embeddings=[embedding],
            metadatas=[{
                **metadata,
                "file_path": file_path,
                "description": description
            }],
            documents=[description]
        )
        
        return file_id
    
    def search_similar(self, query_embedding: List[float], 
                      limit: int = 20) -> List[Dict[str, Any]]:
        """Search for similar media files."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["metadatas", "documents", "distances"]
        )
        
        return self._format_search_results(results)
    
    def update_media(self, file_id: str, metadata: Dict[str, Any]) -> bool:
        """Update media metadata."""
        try:
            self.collection.update(
                ids=[file_id],
                metadatas=[metadata]
            )
            return True
        except Exception:
            return False
    
    def delete_media(self, file_id: str) -> bool:
        """Delete media from database."""
        try:
            self.collection.delete(ids=[file_id])
            return True
        except Exception:
            return False
    
    def _generate_file_id(self, file_path: str) -> str:
        """Generate unique ID for file."""
        return hashlib.md5(file_path.encode()).hexdigest()
    
    def _format_search_results(self, results: Dict) -> List[Dict[str, Any]]:
        """Format ChromaDB results for API response."""
        formatted = []
        for i, metadata in enumerate(results['metadatas'][0]):
            formatted.append({
                "file_id": results['ids'][0][i],
                "metadata": metadata,
                "description": results['documents'][0][i],
                "similarity_score": 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        return formatted
```

## Frontend Implementation Details

### 1. Main App Component

```typescript
// src/App.tsx
import React from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Container, Box } from '@mui/material';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import SearchBar from './components/SearchBar';
import MediaGallery from './components/MediaGallery';
import SettingsPanel from './components/SettingsPanel';
import { useSearchStore } from './store/searchStore';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

const queryClient = new QueryClient();

function App() {
  const { searchResults, isLoading } = useSearchStore();

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Container maxWidth="xl">
          <Box sx={{ py: 4 }}>
            <SearchBar />
            <MediaGallery 
              results={searchResults} 
              loading={isLoading} 
            />
            <SettingsPanel />
          </Box>
        </Container>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
```

### 2. Search Store (Zustand)

```typescript
// src/store/searchStore.ts
import { create } from 'zustand';
import { MediaItem, SearchFilters } from '../types/media';
import { searchAPI } from '../services/api';

interface SearchState {
  query: string;
  searchResults: MediaItem[];
  isLoading: boolean;
  filters: SearchFilters;
  setQuery: (query: string) => void;
  performSearch: (query: string) => Promise<void>;
  setFilters: (filters: SearchFilters) => void;
  clearResults: () => void;
}

export const useSearchStore = create<SearchState>((set, get) => ({
  query: '',
  searchResults: [],
  isLoading: false,
  filters: {
    mediaType: 'all',
    dateRange: null,
    minSimilarity: 0.5,
  },

  setQuery: (query: string) => set({ query }),

  performSearch: async (query: string) => {
    set({ isLoading: true, query });
    try {
      const results = await searchAPI.semanticSearch(query, get().filters);
      set({ searchResults: results, isLoading: false });
    } catch (error) {
      console.error('Search failed:', error);
      set({ searchResults: [], isLoading: false });
    }
  },

  setFilters: (filters: SearchFilters) => set({ filters }),

  clearResults: () => set({ searchResults: [], query: '' }),
}));
```

### 3. API Service

```typescript
// src/services/api.ts
import axios from 'axios';
import { MediaItem, SearchFilters, IndexingStatus } from '../types/media';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

export const searchAPI = {
  semanticSearch: async (query: string, filters: SearchFilters): Promise<MediaItem[]> => {
    const response = await apiClient.post('/api/search', {
      query,
      filters,
      limit: 50,
    });
    return response.data.results;
  },

  getMediaFile: async (fileId: string): Promise<string> => {
    return `${API_BASE_URL}/api/media/${fileId}`;
  },

  getMetadata: async (fileId: string): Promise<any> => {
    const response = await apiClient.get(`/api/metadata/${fileId}`);
    return response.data;
  },
};

export const indexingAPI = {
  startIndexing: async (paths: string[]): Promise<void> => {
    await apiClient.post('/api/index', { paths });
  },

  getIndexingStatus: async (): Promise<IndexingStatus> => {
    const response = await apiClient.get('/api/index/status');
    return response.data;
  },

  stopIndexing: async (): Promise<void> => {
    await apiClient.post('/api/index/stop');
  },
};
```

## Development Environment Setup

### 1. Prerequisites Installation Script

```bash
#!/bin/bash
# scripts/setup.sh

echo "Setting up Local Media Semantic Search development environment..."

# Check if Python 3.11+ is installed
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
if [[ $(echo "$python_version >= 3.11" | bc -l) -eq 0 ]]; then
    echo "Python 3.11+ required. Please install Python 3.11 or higher."
    exit 1
fi

# Check if Node.js 18+ is installed
node_version=$(node --version 2>&1 | grep -o '[0-9]\+' | head -1)
if [[ $node_version -lt 18 ]]; then
    echo "Node.js 18+ required. Please install Node.js 18 or higher."
    exit 1
fi

# Install Ollama if not present
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# Pull Gemma3 model
echo "Pulling Gemma3:4b model..."
ollama pull gemma3:4b

# Set up Python virtual environment
echo "Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install poetry

# Install Python dependencies
cd backend
poetry install
cd ..

# Install Node.js dependencies
echo "Setting up Node.js environment..."
cd frontend
npm install
cd ..

# Create environment file
cp .env.example .env

echo "Setup complete! Run 'scripts/dev-start.sh' to start development servers."
```

### 2. Development Startup Script

```bash
#!/bin/bash
# scripts/dev-start.sh

echo "Starting development environment..."

# Start Ollama service
ollama serve &
OLLAMA_PID=$!

# Start backend
cd backend
source ../venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Start frontend
cd ../frontend
npm run dev &
FRONTEND_PID=$!

echo "Services started:"
echo "- Ollama: Background service"
echo "- Backend: http://localhost:8000"
echo "- Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap "kill $OLLAMA_PID $BACKEND_PID $FRONTEND_PID" EXIT
wait
```

This technical specification provides the detailed implementation structure needed to build the MVP. The next step would be to start implementing these components following the development roadmap in the main plan.
