@echo off
echo Building Docker image for Graph Neural Network Backend...
docker build -t gnn-backend .

echo Running Docker container...
docker run -d --name gnn-backend -p 5000:5000 --restart unless-stopped gnn-backend

echo Container started! Backend is running on http://localhost:5000
echo Health check: http://localhost:5000/api/health
echo.
echo To stop the container: docker stop gnn-backend
echo To remove the container: docker rm gnn-backend
echo To view logs: docker logs gnn-backend
pause 