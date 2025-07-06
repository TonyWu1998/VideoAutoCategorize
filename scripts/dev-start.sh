#!/bin/bash

# Local Media Semantic Search - Development Startup Script
# This script starts all development services

set -e  # Exit on any error

echo "🚀 Starting Local Media Semantic Search development environment..."
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

# Function to cleanup background processes
cleanup() {
    print_status "Shutting down services..."
    
    if [[ ! -z "$OLLAMA_PID" ]]; then
        kill $OLLAMA_PID 2>/dev/null || true
        print_status "Ollama service stopped"
    fi
    
    if [[ ! -z "$BACKEND_PID" ]]; then
        kill $BACKEND_PID 2>/dev/null || true
        print_status "Backend service stopped"
    fi
    
    if [[ ! -z "$FRONTEND_PID" ]]; then
        kill $FRONTEND_PID 2>/dev/null || true
        print_status "Frontend service stopped"
    fi
    
    print_success "All services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check if we're in the project root
if [[ ! -f "MVP_DEVELOPMENT_PLAN.md" ]]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Check if setup has been run
if [[ ! -d "venv" ]]; then
    print_error "Virtual environment not found. Please run './scripts/setup.sh' first"
    exit 1
fi

if [[ ! -d "frontend/node_modules" ]]; then
    print_error "Frontend dependencies not found. Please run './scripts/setup.sh' first"
    exit 1
fi

# Load environment variables
if [[ -f ".env" ]]; then
    source .env
    print_success "Environment variables loaded"
else
    print_warning "No .env file found. Using default configuration"
fi

# Start Ollama service
print_status "Starting Ollama service..."
if ! pgrep -f "ollama serve" > /dev/null; then
    ollama serve &
    OLLAMA_PID=$!
    print_success "Ollama service started (PID: $OLLAMA_PID)"
    
    # Wait for Ollama to be ready
    print_status "Waiting for Ollama to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            print_success "Ollama is ready"
            break
        fi
        if [[ $i -eq 30 ]]; then
            print_error "Ollama failed to start within 30 seconds"
            exit 1
        fi
        sleep 1
    done
else
    print_success "Ollama service already running"
fi

# Check if Gemma3 model is available
print_status "Checking Gemma3:4b model availability..."
if ollama list | grep -q "gemma3:4b"; then
    print_success "Gemma3:4b model is available"
else
    print_warning "Gemma3:4b model not found. Attempting to pull..."
    if ollama pull gemma3:4b; then
        print_success "Gemma3:4b model downloaded"
    else
        print_error "Failed to download Gemma3:4b model"
        print_status "You can continue without the model, but AI features won't work"
    fi
fi

# Start backend service
print_status "Starting backend service..."
cd backend

# Activate virtual environment
source ../venv/bin/activate

# Start FastAPI server
if command -v uvicorn &> /dev/null; then
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    print_success "Backend service started (PID: $BACKEND_PID)"
else
    print_error "uvicorn not found. Please check your Python environment"
    exit 1
fi

cd ..

# Wait for backend to be ready
print_status "Waiting for backend to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/api/health/simple >/dev/null 2>&1; then
        print_success "Backend is ready"
        break
    fi
    if [[ $i -eq 30 ]]; then
        print_error "Backend failed to start within 30 seconds"
        exit 1
    fi
    sleep 1
done

# Start frontend service
print_status "Starting frontend service..."
cd frontend

if command -v npm &> /dev/null; then
    npm run dev &
    FRONTEND_PID=$!
    print_success "Frontend service started (PID: $FRONTEND_PID)"
else
    print_error "npm not found. Please check your Node.js installation"
    exit 1
fi

cd ..

# Wait for frontend to be ready
print_status "Waiting for frontend to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:5173 >/dev/null 2>&1; then
        print_success "Frontend is ready"
        break
    fi
    if [[ $i -eq 30 ]]; then
        print_warning "Frontend may still be starting up"
        break
    fi
    sleep 1
done

echo ""
echo "=================================================================="
print_success "All services started successfully!"
echo ""
echo "🌐 Services are now available:"
echo "   Frontend:        http://localhost:5173"
echo "   Backend API:     http://localhost:8000"
echo "   API Docs:        http://localhost:8000/docs"
echo "   Health Check:    http://localhost:8000/api/health"
echo ""
echo "📝 Service PIDs:"
echo "   Ollama:          ${OLLAMA_PID:-'Already running'}"
echo "   Backend:         ${BACKEND_PID:-'N/A'}"
echo "   Frontend:        ${FRONTEND_PID:-'N/A'}"
echo ""
echo "🔧 Development Tips:"
echo "   - Backend auto-reloads on file changes"
echo "   - Frontend has hot module replacement"
echo "   - Check logs in terminal for any issues"
echo "   - Use Ctrl+C to stop all services"
echo ""
print_status "Ready for development! 🎉"
echo ""

# Keep script running and wait for interrupt
wait
