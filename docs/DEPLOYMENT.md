# FoodVisionAI Deployment Guide

This guide covers deploying FoodVisionAI in various environments.

## 🚀 Quick Start (Docker)

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- 4GB RAM minimum
- 10GB disk space

### 1. Clone and Configure

```bash
# Clone repository
git clone <repository-url>
cd foodvision_ai

# Copy environment file
cp .env.example .env

# Edit .env with your configuration
nano .env  # or use your preferred editor
```

### 2. Set Required Environment Variables

```bash
# In .env file, set at minimum:
GEMINI_API_KEY=your_actual_gemini_api_key
MONGODB_ROOT_PASSWORD=secure_password_here
```

### 3. Start Services

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Check status
docker-compose ps
```

### 4. Access Application

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 5. Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v
```

---

## 🔧 Development Setup (Local)

### Prerequisites
- Python 3.11+
- MongoDB 7.0+
- 8GB RAM recommended

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup MongoDB

See [MONGODB_SETUP.md](MONGODB_SETUP.md) for detailed instructions.

```bash
# Quick start with Docker
docker run -d -p 27017:27017 --name mongodb mongo:7.0
```

### 3. Configure Environment

```bash
# Copy environment file
cp .env.example .env

# Edit configuration
# Set GEMINI_API_KEY and other required variables
```

### 4. Run Application

```bash
# Start the application
python main.py

# Or with uvicorn directly
uvicorn foodvision_ai.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🌐 Production Deployment

### Option 1: Docker Compose (Recommended)

**Best for**: Small to medium deployments, single server

```bash
# Production configuration
cp .env.example .env.production

# Edit production settings
nano .env.production

# Deploy
docker-compose --env-file .env.production up -d

# Enable auto-restart
docker-compose --env-file .env.production up -d --restart unless-stopped
```

### Option 2: Kubernetes

**Best for**: Large scale, multi-server deployments

```bash
# Create namespace
kubectl create namespace foodvision

# Create secrets
kubectl create secret generic foodvision-secrets \
  --from-literal=mongodb-password=<password> \
  --from-literal=gemini-api-key=<key> \
  -n foodvision

# Deploy (requires k8s manifests - see k8s/ directory)
kubectl apply -f k8s/ -n foodvision
```

### Option 3: Cloud Platforms

#### AWS (Elastic Beanstalk)
```bash
# Install EB CLI
pip install awsebcli

# Initialize
eb init -p docker foodvision-ai

# Create environment
eb create foodvision-prod

# Deploy
eb deploy
```

#### Google Cloud (Cloud Run)
```bash
# Build and push image
gcloud builds submit --tag gcr.io/PROJECT_ID/foodvision-ai

# Deploy
gcloud run deploy foodvision-ai \
  --image gcr.io/PROJECT_ID/foodvision-ai \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Azure (Container Instances)
```bash
# Create resource group
az group create --name foodvision-rg --location eastus

# Deploy container
az container create \
  --resource-group foodvision-rg \
  --name foodvision-ai \
  --image <your-registry>/foodvision-ai:latest \
  --dns-name-label foodvision-ai \
  --ports 8000
```

---

## 🔒 Security Checklist

### Before Production Deployment

- [ ] Change default MongoDB password
- [ ] Set strong SECRET_KEY in .env
- [ ] Configure CORS_ORIGINS for your domain
- [ ] Enable HTTPS/SSL
- [ ] Set DEBUG=false
- [ ] Configure firewall rules
- [ ] Set up rate limiting
- [ ] Enable authentication if needed
- [ ] Regular security updates
- [ ] Backup strategy in place

### Recommended Security Headers

Add to nginx/reverse proxy:
```nginx
add_header X-Frame-Options "SAMEORIGIN";
add_header X-Content-Type-Options "nosniff";
add_header X-XSS-Protection "1; mode=block";
add_header Strict-Transport-Security "max-age=31536000";
```

---

## 📊 Monitoring & Logging

### Health Checks

```bash
# Application health
curl http://localhost:8000/health

# MongoDB health
docker exec foodvision_mongodb mongosh --eval "db.adminCommand('ping')"
```

### View Logs

```bash
# Application logs
docker-compose logs -f app

# MongoDB logs
docker-compose logs -f mongodb

# All logs
docker-compose logs -f
```

### Log Files

- Application: `logs/foodvision_ai.log`
- Access logs: Handled by uvicorn
- Error logs: stderr

---

## 🔄 Backup & Recovery

### Database Backup

```bash
# Backup MongoDB
docker exec foodvision_mongodb mongodump \
  --out /data/backup \
  --db foodvision_ai

# Copy backup to host
docker cp foodvision_mongodb:/data/backup ./backup
```

### Database Restore

```bash
# Restore MongoDB
docker exec foodvision_mongodb mongorestore \
  --db foodvision_ai \
  /data/backup/foodvision_ai
```

### Automated Backups

Add to crontab:
```bash
# Daily backup at 2 AM
0 2 * * * /path/to/backup-script.sh
```

---

## 🚨 Troubleshooting

### Application Won't Start

```bash
# Check logs
docker-compose logs app

# Check environment variables
docker-compose config

# Restart services
docker-compose restart
```

### Database Connection Issues

```bash
# Check MongoDB is running
docker-compose ps mongodb

# Test connection
docker exec foodvision_mongodb mongosh --eval "db.adminCommand('ping')"

# Check network
docker network inspect foodvision_foodvision_network
```

### Performance Issues

```bash
# Check resource usage
docker stats

# Scale workers (if using gunicorn)
# Update docker-compose.yml:
# command: gunicorn -w 4 -k uvicorn.workers.UvicornWorker foodvision_ai.api.main:app
```

---

## 📈 Scaling

### Horizontal Scaling

```yaml
# docker-compose.yml
services:
  app:
    deploy:
      replicas: 3
    # ... rest of config
```

### Load Balancing

Use nginx as reverse proxy:
```nginx
upstream foodvision {
    server app1:8000;
    server app2:8000;
    server app3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://foodvision;
    }
}
```

---

## 🔧 Maintenance

### Update Application

```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose build app
docker-compose up -d app
```

### Update Dependencies

```bash
# Update requirements.txt
pip install --upgrade -r requirements.txt
pip freeze > requirements.txt

# Rebuild image
docker-compose build app
```

---

## 📞 Support

For issues and questions:
- Check logs first
- Review documentation
- Open GitHub issue
- Contact support team

---

**Last Updated**: 2025-12-15  
**Version**: 1.0.0

