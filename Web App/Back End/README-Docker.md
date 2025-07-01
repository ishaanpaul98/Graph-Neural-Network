# Docker Setup for Graph Neural Network Backend

This guide explains how to dockerize and run the Flask backend for the Graph Neural Network movie recommendation system.

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, for multi-service orchestration)

## Quick Start

### Option 1: Using Docker Compose (Recommended)

From the root directory of your project:

```bash
# Build and start the backend service
docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop the service
docker-compose down
```

### Option 2: Using Docker directly

Navigate to the Back End directory and run:

**On Linux/Mac:**
```bash
chmod +x docker-run.sh
./docker-run.sh
```

**On Windows:**
```cmd
docker-run.bat
```

**Manual Docker commands:**
```bash
# Build the image
docker build -t gnn-backend .

# Run the container
docker run -d --name gnn-backend -p 5000:5000 --restart unless-stopped gnn-backend

# View logs
docker logs gnn-backend

# Stop the container
docker stop gnn-backend

# Remove the container
docker rm gnn-backend
```

## API Endpoints

Once the container is running, the following endpoints will be available:

- **Health Check**: `GET http://localhost:5000/api/health`
- **Get Available Movies**: `GET http://localhost:5000/api/movies`
- **Get Recommendations**: `POST http://localhost:5000/api/recommend`

## Configuration

### Environment Variables

The following environment variables can be customized:

- `FLASK_ENV`: Set to `production` for production deployment
- `PYTHONUNBUFFERED`: Set to `1` for immediate log output
- `FLASK_APP`: Set to `app.py` (default)

### Port Configuration

The default port is 5000. To change it, modify the `EXPOSE` directive in the Dockerfile and update the port mapping in docker-compose.yml or docker run command.

## Model Files

The Docker image includes the trained model file (`mpgnn_model.pth`) in the `/app/models/` directory. If you need to update the model:

1. Replace the model file in `Back End/models/`
2. Rebuild the Docker image

## Troubleshooting

### Common Issues

1. **Port already in use**: Change the port mapping in docker-compose.yml or use a different port
2. **Model not found**: Ensure the model file exists in the `Back End/models/` directory
3. **Memory issues**: The model requires significant memory. Consider increasing Docker memory limits

### Logs and Debugging

```bash
# View container logs
docker logs gnn-backend

# View real-time logs
docker logs -f gnn-backend

# Access container shell
docker exec -it gnn-backend /bin/bash
```

### Health Check

The container includes a health check that verifies the API is responding:

```bash
curl http://localhost:5000/api/health
```

Expected response: `{"status": "healthy"}`

## Production Deployment

For production deployment, consider:

1. **Using a reverse proxy** (nginx) in front of the Flask app
2. **Setting up SSL/TLS** certificates
3. **Configuring proper logging** and monitoring
4. **Using environment-specific configuration files**
5. **Setting up container orchestration** (Kubernetes, Docker Swarm)

## Security Considerations

- The container runs as a non-root user for security
- System dependencies are minimized
- Health checks are implemented
- CORS is configured for frontend communication

## Performance Optimization

- The Dockerfile uses multi-stage builds for smaller images
- Dependencies are cached efficiently
- Gunicorn is used as the WSGI server for better performance
- The `.dockerignore` file excludes unnecessary files from the build context 