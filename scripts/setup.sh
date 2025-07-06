#!/bin/bash

# Local Media Semantic Search - Development Environment Setup Script
# This script sets up the complete development environment

set -e  # Exit on any error

echo "🚀 Setting up Local Media Semantic Search development environment..."
echo "=================================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on macOS or Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    print_status "Detected macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    print_status "Detected Linux"
else
    print_error "Unsupported operating system: $OSTYPE"
    exit 1
fi

# Check Python version
print_status "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 11 ]]; then
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.11+ required. Found Python $PYTHON_VERSION"
        print_status "Please install Python 3.11 or higher"
        exit 1
    fi
else
    print_error "Python 3 not found. Please install Python 3.11+"
    exit 1
fi

# Check Node.js version
print_status "Checking Node.js installation..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version 2>&1 | grep -o '[0-9]\+' | head -1)
    if [[ $NODE_VERSION -ge 18 ]]; then
        print_success "Node.js v$NODE_VERSION found"
    else
        print_error "Node.js 18+ required. Found Node.js v$NODE_VERSION"
        print_status "Please install Node.js 18 or higher"
        exit 1
    fi
else
    print_error "Node.js not found. Please install Node.js 18+"
    exit 1
fi

# Install Ollama if not present
print_status "Checking Ollama installation..."
if ! command -v ollama &> /dev/null; then
    print_status "Installing Ollama..."
    if [[ "$OS" == "macos" ]]; then
        curl -fsSL https://ollama.ai/install.sh | sh
    elif [[ "$OS" == "linux" ]]; then
        curl -fsSL https://ollama.ai/install.sh | sh
    fi
    print_success "Ollama installed"
else
    print_success "Ollama already installed"
fi

# Start Ollama service
print_status "Starting Ollama service..."
if [[ "$OS" == "macos" ]]; then
    # On macOS, Ollama runs as a service
    ollama serve &
    OLLAMA_PID=$!
    sleep 3
elif [[ "$OS" == "linux" ]]; then
    # On Linux, start Ollama in background
    ollama serve &
    OLLAMA_PID=$!
    sleep 3
fi

# Pull Gemma3 model
print_status "Pulling Gemma3:4b model (this may take a while)..."
if ollama pull gemma3:4b; then
    print_success "Gemma3:4b model downloaded successfully"
else
    print_warning "Failed to download Gemma3:4b model. You can try again later with: ollama pull gemma3:4b"
fi

# Set up Python virtual environment
print_status "Setting up Python virtual environment..."
if [[ ! -d "venv" ]]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_success "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install Poetry if not present
if ! command -v poetry &> /dev/null; then
    print_status "Installing Poetry..."
    pip install poetry
    print_success "Poetry installed"
else
    print_success "Poetry already installed"
fi

# Install Python dependencies
print_status "Installing Python dependencies..."
cd backend
if [[ -f "pyproject.toml" ]]; then
    poetry install
    print_success "Python dependencies installed via Poetry"
elif [[ -f "requirements.txt" ]]; then
    pip install -r requirements.txt
    print_success "Python dependencies installed via pip"
else
    print_error "No Python dependency file found (pyproject.toml or requirements.txt)"
    exit 1
fi
cd ..

# Install Node.js dependencies
print_status "Installing Node.js dependencies..."
cd frontend
if [[ -f "package.json" ]]; then
    if command -v npm &> /dev/null; then
        npm install
        print_success "Node.js dependencies installed via npm"
    else
        print_error "npm not found"
        exit 1
    fi
else
    print_error "No package.json found in frontend directory"
    exit 1
fi
cd ..

# Create environment file
print_status "Creating environment configuration..."
if [[ ! -f ".env" ]]; then
    if [[ -f ".env.example" ]]; then
        cp .env.example .env
        print_success "Environment file created from .env.example"
        print_warning "Please review and update .env file with your specific configuration"
    else
        print_warning "No .env.example file found. You may need to create .env manually"
    fi
else
    print_success "Environment file already exists"
fi

# Create required directories
print_status "Creating required directories..."
mkdir -p data/chroma_db
mkdir -p data/media
mkdir -p data/thumbnails
mkdir -p logs
mkdir -p backups
print_success "Required directories created"

# Set permissions
print_status "Setting permissions..."
chmod +x scripts/*.sh
print_success "Script permissions set"

# Final setup verification
print_status "Verifying setup..."

# Test Python imports
if python3 -c "import fastapi, chromadb, ollama" 2>/dev/null; then
    print_success "Python dependencies verified"
else
    print_error "Some Python dependencies are missing"
fi

# Test Node.js setup
cd frontend
if npm run type-check &>/dev/null; then
    print_success "Frontend TypeScript compilation verified"
else
    print_warning "Frontend TypeScript compilation issues detected"
fi
cd ..

echo ""
echo "=================================================================="
print_success "Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Review and update the .env file if needed"
echo "2. Run './scripts/dev-start.sh' to start the development servers"
echo "3. Open http://localhost:5173 in your browser"
echo ""
echo "Services will be available at:"
echo "- Frontend: http://localhost:5173"
echo "- Backend API: http://localhost:8000"
echo "- API Documentation: http://localhost:8000/docs"
echo ""
print_status "Happy coding! 🎉"
