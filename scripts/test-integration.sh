#!/bin/bash

# Integration test script for Local Media Semantic Search
# Tests basic functionality of the complete system

set -e

echo "🧪 Running integration tests for Local Media Semantic Search..."
echo "=============================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

print_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

# Test counter
TESTS_RUN=0
TESTS_PASSED=0

run_test() {
    local test_name="$1"
    local test_command="$2"
    
    TESTS_RUN=$((TESTS_RUN + 1))
    print_status "Running: $test_name"
    
    if eval "$test_command" >/dev/null 2>&1; then
        print_success "$test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        print_error "$test_name"
        return 1
    fi
}

# Check if services are running
print_status "Checking if services are running..."

# Test backend health
run_test "Backend health check" "curl -s http://localhost:8000/api/health/simple"

# Test backend API endpoints
run_test "Backend root endpoint" "curl -s http://localhost:8000/"
run_test "Backend API info" "curl -s http://localhost:8000/api"

# Test frontend
run_test "Frontend accessibility" "curl -s http://localhost:5173"

# Test API endpoints
print_status "Testing API endpoints..."

# Test search endpoint (should handle empty query gracefully)
run_test "Search endpoint availability" "curl -s -X GET 'http://localhost:8000/api/search?q=test&limit=1'"

# Test indexing status
run_test "Indexing status endpoint" "curl -s http://localhost:8000/api/index/status"

# Test media stats
run_test "Media stats endpoint" "curl -s http://localhost:8000/api/media/stats"

# Test supported formats
run_test "Supported formats endpoint" "curl -s http://localhost:8000/api/media/formats/supported"

# Test database connectivity
print_status "Testing database connectivity..."
run_test "Database health check" "curl -s http://localhost:8000/api/health/database"

# Test Ollama connectivity (may fail if model not available)
print_status "Testing Ollama connectivity..."
if run_test "Ollama health check" "curl -s http://localhost:8000/api/health/ollama"; then
    echo "  ✓ Ollama is properly configured"
else
    echo "  ⚠ Ollama may not be configured (this is expected if model isn't downloaded)"
fi

# Run backend unit tests if pytest is available
print_status "Running backend unit tests..."
cd backend
if command -v pytest &> /dev/null; then
    if source ../venv/bin/activate && pytest tests/ -v; then
        print_success "Backend unit tests passed"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        print_error "Backend unit tests failed"
    fi
    TESTS_RUN=$((TESTS_RUN + 1))
else
    echo "  ⚠ pytest not available, skipping unit tests"
fi
cd ..

# Test frontend build
print_status "Testing frontend build..."
cd frontend
if npm run type-check >/dev/null 2>&1; then
    print_success "Frontend TypeScript compilation"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    print_error "Frontend TypeScript compilation"
fi
TESTS_RUN=$((TESTS_RUN + 1))
cd ..

# Summary
echo ""
echo "=============================================================="
echo "Integration Test Results:"
echo "  Tests Run: $TESTS_RUN"
echo "  Tests Passed: $TESTS_PASSED"
echo "  Tests Failed: $((TESTS_RUN - TESTS_PASSED))"

if [[ $TESTS_PASSED -eq $TESTS_RUN ]]; then
    print_success "All tests passed! 🎉"
    echo ""
    echo "✅ System is ready for development"
    echo "✅ All core services are functional"
    echo "✅ API endpoints are responding"
    echo "✅ Frontend is accessible"
    exit 0
else
    print_error "Some tests failed"
    echo ""
    echo "❌ $((TESTS_RUN - TESTS_PASSED)) test(s) failed"
    echo "🔧 Please check the logs and fix any issues"
    exit 1
fi
