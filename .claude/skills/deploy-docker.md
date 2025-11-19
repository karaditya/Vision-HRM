# Deploy with Docker

Build and run Docker container for the RAG API.

## Purpose
- Containerize the application
- Test deployment
- Prepare for production
- Ensure reproducibility

## Usage

```bash
/deploy-docker
```

Or with custom model:
```bash
export MODEL_PATH=outputs/my_model
/deploy-docker
```

## Commands

```bash
MODEL_PATH=${MODEL_PATH:-outputs/quick_test}

echo "=================================="
echo "Docker Deployment"
echo "=================================="
echo "Model: $MODEL_PATH"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH/best_model.pt" ]; then
    echo "❌ Model not found"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not installed"
    echo "   Install: https://docs.docker.com/get-docker/"
    exit 1
fi

# Create Dockerfile if it doesn't exist
if [ ! -f "Dockerfile" ]; then
    echo "Creating Dockerfile..."

    cat > Dockerfile << 'EOF'
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt requirements-mvp.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-mvp.txt || true

# Copy application code
COPY models/ ./models/
COPY security/ ./security/
COPY *.py ./
COPY dataset_loader.py train_language_model.py rag_system.py rag_inference_api.py ./

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/outputs /app/rag_data /app/logs && \
    chown -R appuser:appuser /app

USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-u", "-m", "flask", "--app", "rag_inference_api:create_rest_api", "run", "--host", "0.0.0.0", "--port", "8000"]
EOF

    echo "✓ Dockerfile created"
fi

# Create docker-compose.yml if it doesn't exist
if [ ! -f "docker-compose.yml" ]; then
    echo "Creating docker-compose.yml..."

    cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  hrm-rag:
    build: .
    container_name: hrm-rag-api
    ports:
      - "8000:8000"
    volumes:
      - ./outputs:/app/outputs:ro
      - ./rag_data:/app/rag_data
      - ./logs:/app/logs
    environment:
      - MODEL_PATH=/app/outputs/quick_test
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - ENCRYPTION_KEY=${ENCRYPTION_KEY}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
EOF

    echo "✓ docker-compose.yml created"
fi

# Build image
echo ""
echo "Building Docker image..."
echo "This may take a few minutes..."

docker build -t hrm-rag:latest . || {
    echo "❌ Docker build failed"
    exit 1
}

echo "✓ Image built: hrm-rag:latest"

# Check image size
IMAGE_SIZE=$(docker images hrm-rag:latest --format "{{.Size}}")
echo "  Image size: $IMAGE_SIZE"

# Stop existing container if running
if docker ps -a | grep -q hrm-rag-api; then
    echo ""
    echo "Stopping existing container..."
    docker stop hrm-rag-api 2>/dev/null || true
    docker rm hrm-rag-api 2>/dev/null || true
fi

# Run container
echo ""
echo "Starting container..."

docker run -d \
    --name hrm-rag-api \
    -p 8000:8000 \
    -v "$(pwd)/$MODEL_PATH:/app/outputs/quick_test:ro" \
    -v "$(pwd)/rag_data:/app/rag_data" \
    -v "$(pwd)/logs:/app/logs" \
    -e JWT_SECRET_KEY="${JWT_SECRET_KEY:-test-secret-key}" \
    -e ENCRYPTION_KEY="${ENCRYPTION_KEY:-test-encryption-key}" \
    hrm-rag:latest

# Wait for startup
echo "Waiting for service to start..."
sleep 5

# Check if container is running
if docker ps | grep -q hrm-rag-api; then
    echo "✓ Container running"

    # Test health endpoint
    echo ""
    echo "Testing health endpoint..."

    if curl -f http://localhost:8000/health 2>/dev/null; then
        echo "✓ Health check passed"
    else
        echo "⚠ Health check failed (may need more time to start)"
    fi

    echo ""
    echo "=================================="
    echo "✓ Deployment Complete!"
    echo "=================================="
    echo ""
    echo "Service URL:     http://localhost:8000"
    echo "Container name:  hrm-rag-api"
    echo ""
    echo "Useful commands:"
    echo "  View logs:     docker logs hrm-rag-api"
    echo "  Follow logs:   docker logs -f hrm-rag-api"
    echo "  Stop:          docker stop hrm-rag-api"
    echo "  Start:         docker start hrm-rag-api"
    echo "  Remove:        docker rm -f hrm-rag-api"
    echo ""
    echo "Test the API:"
    echo "  curl http://localhost:8000/health"
    echo "  curl http://localhost:8000/stats"
    echo ""

else
    echo "❌ Container failed to start"
    echo "Check logs: docker logs hrm-rag-api"
    exit 1
fi
```

## Using Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

## Testing the Deployment

```bash
# Health check
curl http://localhost:8000/health

# Stats
curl http://localhost:8000/stats

# Query (if API implemented)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"question": "What is diabetes?"}'
```

## Production Deployment

For production, modify:

1. **Environment Variables**
```yaml
environment:
  - JWT_SECRET_KEY=${JWT_SECRET_KEY}  # From secrets
  - ENCRYPTION_KEY=${ENCRYPTION_KEY}  # From secrets
  - DATABASE_URL=${DATABASE_URL}
```

2. **Resource Limits**
```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
    reservations:
      cpus: '1'
      memory: 2G
```

3. **Logging**
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

## Kubernetes Deployment

See: `kubernetes/deployment.yaml` (to be created in Week 13-14)

## Success Criteria

- [x] Dockerfile created
- [x] Image builds successfully
- [x] Container starts
- [x] Health check passes
- [x] API accessible
- [x] Logs available
