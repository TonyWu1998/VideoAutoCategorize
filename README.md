# Local Media Semantic Search Application

A full-stack application that enables semantic search of locally stored images and videos using AI-powered content analysis. Search your media files using natural language queries like "beach sunset" or "family gathering" and get relevant results ranked by semantic similarity.

## 🚀 Features

- **Semantic Search**: Find media files using natural language descriptions
- **Local AI Processing**: Uses Gemma3:4b model running locally via Ollama
- **Multiple Format Support**: Images (JPG, PNG, GIF, etc.) and videos (MP4, AVI, MOV, etc.)
- **Real-time Indexing**: Automatic detection and processing of new media files
- **Clean Web Interface**: Modern React frontend with Material-UI components
- **Privacy-First**: All processing happens locally, no cloud dependencies
- **Cross-Platform**: Works on Windows, Mac, and Linux

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │  Python Backend │    │ Vector Database │
│   (Material-UI)  │◄──►│   (FastAPI)     │◄──►│   (ChromaDB)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Ollama + Gemma3 │
                       │  (Local LLM)    │
                       └─────────────────┘
```

## 🛠️ Technology Stack

### Frontend
- **React 18** with TypeScript
- **Material-UI (MUI) v5** for components
- **Vite** for fast development and building
- **React Query** for API state management
- **Zustand** for client-side state

### Backend
- **FastAPI** (Python 3.11+) for REST API
- **ChromaDB** for vector storage and semantic search
- **Ollama** for local LLM integration
- **Pillow & OpenCV** for image processing
- **FFmpeg** for video processing

### AI/ML
- **Gemma3:4b** vision model via Ollama
- **Local processing** - no external API calls
- **Vector embeddings** for semantic similarity

## 📋 Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **Git**
- **4GB+ RAM** (recommended 8GB for optimal performance)
- **2GB+ free disk space** for models and data

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/VideoAutoCategorize.git
cd VideoAutoCategorize
```

### 2. Run Setup Script
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

This script will:
- Install Ollama and pull the Gemma3:4b model
- Set up Python virtual environment
- Install all dependencies
- Create configuration files

### 3. Start Development Environment
```bash
./scripts/dev-start.sh
```

This will start:
- **Backend API**: http://localhost:8000
- **Frontend**: http://localhost:5173
- **API Documentation**: http://localhost:8000/docs

### 4. Configure Media Directories

1. Open the application at http://localhost:5173
2. Go to Settings panel
3. Add your media directories for indexing
4. Start the indexing process

## 📖 Documentation

- **[MVP Development Plan](MVP_DEVELOPMENT_PLAN.md)** - Comprehensive development roadmap
- **[Technical Specifications](TECHNICAL_SPECIFICATIONS.md)** - Detailed implementation guide
- **[API Documentation](http://localhost:8000/docs)** - Interactive API docs (when running)

## 🔧 Development

### Project Structure
```
VideoAutoCategorize/
├── backend/           # Python FastAPI backend
├── frontend/          # React TypeScript frontend
├── scripts/           # Setup and utility scripts
├── docs/             # Documentation
└── docker/           # Docker configurations
```

### Running Tests
```bash
# Backend tests
cd backend
poetry run pytest

# Frontend tests
cd frontend
npm test
```

### Code Quality
```bash
# Backend linting
cd backend
poetry run black .
poetry run mypy .

# Frontend linting
cd frontend
npm run lint
npm run type-check
```

## 🎯 Usage

### Basic Search
1. Type natural language queries in the search bar:
   - "sunset over water"
   - "people laughing"
   - "red car in parking lot"
   - "birthday party with cake"

2. Use filters to refine results:
   - Media type (images/videos/all)
   - Date range
   - Similarity threshold

### Advanced Features
- **Batch Operations**: Select multiple files for actions
- **Metadata Viewing**: See AI-generated descriptions and tags
- **File Management**: Open files in default applications
- **Export Results**: Save search results for later use

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root:

```env
# Database
CHROMA_DB_PATH=./data/chroma_db

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b

# API Configuration
ALLOWED_ORIGINS=["http://localhost:3000", "http://localhost:5173"]

# Processing
BATCH_SIZE=10
MAX_CONCURRENT_PROCESSING=4
MAX_FILE_SIZE_MB=500
```

### Supported File Formats

**Images**: JPG, JPEG, PNG, GIF, BMP, WebP, TIFF
**Videos**: MP4, AVI, MOV, MKV, WMV, FLV, M4V

## 🚀 Deployment

### Production Build
```bash
# Build frontend
cd frontend
npm run build

# Start production backend
cd backend
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker Deployment
```bash
docker-compose up -d
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run quality checks: `npm run lint` and `poetry run black .`
5. Commit your changes: `git commit -m 'Add feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

**Ollama not starting**
```bash
# Check if Ollama is running
ollama list

# Restart Ollama service
ollama serve
```

**Model not found**
```bash
# Pull the required model
ollama pull gemma3:4b
```

**Port conflicts**
```bash
# Check what's using the ports
lsof -i :8000  # Backend
lsof -i :5173  # Frontend
```

**Memory issues**
- Reduce `BATCH_SIZE` in configuration
- Close other applications
- Consider using a smaller model

### Getting Help

- Check the [Issues](https://github.com/yourusername/VideoAutoCategorize/issues) page
- Review the [Documentation](docs/)
- Join our [Discord Community](https://discord.gg/your-invite)

## 🎯 Roadmap

### MVP (Current)
- [x] Basic semantic search functionality
- [x] Image and video indexing
- [x] Web interface
- [x] Local AI processing

### Future Enhancements
- [ ] Mobile app support
- [ ] Advanced filtering options
- [ ] Batch editing capabilities
- [ ] Integration with cloud storage
- [ ] Multi-language support
- [ ] Custom model training

## 📊 Performance

### Benchmarks
- **Indexing Speed**: 10-50 files/minute (varies by file size)
- **Search Latency**: <500ms for typical queries
- **Memory Usage**: 2-4GB during normal operation
- **Storage Overhead**: <10% of original media size

### Optimization Tips
- Use SSD storage for better performance
- Allocate sufficient RAM for the AI model
- Index during off-peak hours for large collections
- Regular database maintenance for optimal search speed

---

**Built with ❤️ for privacy-conscious media enthusiasts**
