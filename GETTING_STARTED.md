# Getting Started - Local Media Semantic Search

This guide will help you set up and run the Local Media Semantic Search application on your local machine.

## Quick Start

### 1. Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11+** - [Download Python](https://www.python.org/downloads/)
- **Node.js 18+** - [Download Node.js](https://nodejs.org/)
- **Git** - [Download Git](https://git-scm.com/)

### 2. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd VideoAutoCategorize

# Run the automated setup script
./scripts/setup.sh
```

The setup script will:
- Install Ollama and download the Gemma3:4b model
- Create a Python virtual environment
- Install all Python dependencies
- Install all Node.js dependencies
- Create required directories and configuration files

### 3. Start Development Servers

```bash
# Start all services (Ollama, Backend, Frontend)
./scripts/dev-start.sh
```

This will start:
- **Ollama service** (AI model server)
- **Backend API** at http://localhost:8000
- **Frontend** at http://localhost:5173

### 4. Access the Application

Open your browser and navigate to:
- **Main Application**: http://localhost:5173
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

## First Steps

### 1. Configure Media Directories

1. Click the **Settings** icon in the top-right corner
2. Go to the **Indexing** tab
3. Add paths to your media directories (e.g., `/Users/yourname/Pictures`)
4. Click **Start Indexing**

### 2. Wait for Indexing

The system will:
- Scan your directories for supported media files
- Analyze each file using AI to generate descriptions and tags
- Store the results in a local vector database

### 3. Start Searching

Once indexing is complete, you can search using natural language:
- "sunset over water"
- "family gathering"
- "red car in parking lot"
- "birthday party with cake"

## Supported File Formats

### Images
- JPG, JPEG, PNG, GIF, BMP, WebP, TIFF, SVG

### Videos
- MP4, AVI, MOV, MKV, WMV, FLV, M4V, WebM

## Troubleshooting

### Common Issues

**1. Ollama Model Not Found**
```bash
# Manually pull the model
ollama pull gemma3:4b
```

**2. Port Already in Use**
```bash
# Check what's using the ports
lsof -i :8000  # Backend
lsof -i :5173  # Frontend
lsof -i :11434 # Ollama
```

**3. Python Dependencies Issues**
```bash
# Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

**4. Node.js Dependencies Issues**
```bash
# Clear and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Memory Issues

If you encounter memory issues:
1. Reduce batch size in settings
2. Close other applications
3. Consider using a smaller AI model (if available)

### Performance Tips

- Use SSD storage for better performance
- Allocate at least 4GB RAM for optimal operation
- Index during off-peak hours for large collections

## Development

### Running Tests

```bash
# Backend tests
cd backend
source ../venv/bin/activate
pytest

# Frontend tests
cd frontend
npm test

# Integration tests
./scripts/test-integration.sh
```

### Code Quality

```bash
# Backend linting
cd backend
black .
mypy .

# Frontend linting
cd frontend
npm run lint
npm run type-check
```

### Project Structure

```
VideoAutoCategorize/
├── backend/           # Python FastAPI backend
│   ├── app/          # Application code
│   ├── tests/        # Test files
│   └── requirements.txt
├── frontend/         # React TypeScript frontend
│   ├── src/          # Source code
│   ├── public/       # Static assets
│   └── package.json
├── scripts/          # Setup and utility scripts
├── data/            # Database and media storage
├── logs/            # Application logs
└── docs/            # Documentation
```

## Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Database
CHROMA_DB_PATH=./data/chroma_db

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b

# Processing
BATCH_SIZE=10
MAX_CONCURRENT_PROCESSING=4
MAX_FILE_SIZE_MB=500

# Search
DEFAULT_SEARCH_LIMIT=20
MIN_SIMILARITY_THRESHOLD=0.3
```

### Performance Tuning

For large media collections:
- Increase `MAX_MEMORY_USAGE_MB`
- Adjust `BATCH_SIZE` based on available RAM
- Enable `ENABLE_DB_COMPRESSION` for storage efficiency

## Next Steps

1. **Index Your Media**: Start with a small directory to test
2. **Explore Search**: Try different types of queries
3. **Customize Settings**: Adjust similarity thresholds and filters
4. **Monitor Performance**: Check system status in settings

## Getting Help

- Check the [Technical Specifications](TECHNICAL_SPECIFICATIONS.md)
- Review the [MVP Development Plan](MVP_DEVELOPMENT_PLAN.md)
- Run integration tests: `./scripts/test-integration.sh`
- Check application logs in the `logs/` directory

## What's Next?

This is a foundational implementation. Future enhancements could include:
- Mobile app support
- Cloud storage integration
- Advanced filtering options
- Custom model training
- Multi-language support

Happy searching! 🔍✨
