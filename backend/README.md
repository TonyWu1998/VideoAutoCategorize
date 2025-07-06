# Media Semantic Search Backend

FastAPI backend for the Local Media Semantic Search application.

## Features

- Semantic search using AI-powered vector similarity
- Local media file indexing and analysis
- RESTful API with automatic documentation
- ChromaDB vector database integration
- Ollama LLM integration for content analysis

## Development

```bash
# Install dependencies
poetry install

# Run development server
uvicorn app.main:app --reload

# Run tests
pytest
```

## API Documentation

When running, visit http://localhost:8000/docs for interactive API documentation.
