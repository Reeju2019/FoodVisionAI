# MongoDB Setup for FoodVisionAI

## Quick Start Options

### Option 1: Docker (Recommended for Development)

**Prerequisites**: Docker installed on your system

```bash
# Pull and run MongoDB container
docker run -d \
  --name foodvision_mongodb \
  -p 27017:27017 \
  -v mongodb_data:/data/db \
  mongo:latest

# Verify it's running
docker ps

# View logs
docker logs foodvision_mongodb
```

**Connection String**: `mongodb://localhost:27017`

---

### Option 2: Docker Compose (Recommended for Production)

Create `docker-compose.yml` in project root:

```yaml
version: '3.8'

services:
  mongodb:
    image: mongo:7.0
    container_name: foodvision_mongodb
    restart: unless-stopped
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
      - mongodb_config:/data/configdb
    environment:
      MONGO_INITDB_DATABASE: foodvision_ai
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: ${MONGODB_ROOT_PASSWORD}
    networks:
      - foodvision_network

volumes:
  mongodb_data:
    driver: local
  mongodb_config:
    driver: local

networks:
  foodvision_network:
    driver: bridge
```

**Start MongoDB**:
```bash
docker-compose up -d mongodb
```

**Connection String**: `mongodb://<admin>:<password>@localhost:27017/foodvision_ai?authSource=admin`

---

### Option 3: MongoDB Community Edition (Local Installation)

#### Windows Installation:
1. Download from: https://www.mongodb.com/try/download/community
2. Run the installer (choose "Complete" installation)
3. Install as Windows Service (recommended)
4. Start MongoDB service:
   ```cmd
   net start MongoDB
   ```
5. Verify installation:
   ```cmd
   mongod --version
   ```

#### Linux Installation (Ubuntu/Debian):
```bash
# Import MongoDB public GPG key
wget -qO - https://www.mongodb.org/static/pgp/server-7.0.asc | sudo apt-key add -

# Create list file
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list

# Update package database
sudo apt-get update

# Install MongoDB
sudo apt-get install -y mongodb-org

# Start MongoDB
sudo systemctl start mongod
sudo systemctl enable mongod

# Verify
sudo systemctl status mongod
```

#### macOS Installation:
```bash
# Using Homebrew
brew tap mongodb/brew
brew install mongodb-community@7.0

# Start MongoDB
brew services start mongodb-community@7.0

# Verify
brew services list
```

**Connection String**: `mongodb://localhost:27017`

---

### Option 4: MongoDB Atlas (Cloud - Free Tier)

**Best for**: Production deployment, no local setup required

1. **Create Account**
   - Go to https://www.mongodb.com/atlas
   - Sign up for free account

2. **Create Cluster**
   - Choose "Shared" (Free tier)
   - Select cloud provider and region
   - Click "Create Cluster"

3. **Configure Access**
   - Database Access: Create database user
   - Network Access: Add IP address (0.0.0.0/0 for development)

4. **Get Connection String**
   - Click "Connect" on your cluster
   - Choose "Connect your application"
   - Copy connection string
   - Replace `<password>` with your database user password

**Connection String**: 
```
mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/foodvision_ai?retryWrites=true&w=majority
```

---

## Configuration

### Update .env File

```bash
# Local MongoDB
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=foodvision_ai

# MongoDB Atlas
MONGODB_URL=mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/foodvision_ai?retryWrites=true&w=majority
DATABASE_NAME=foodvision_ai

# Docker with authentication
MONGODB_URL=mongodb://admin:password@localhost:27017/foodvision_ai?authSource=admin
DATABASE_NAME=foodvision_ai
```

---

## Verification

### Test Connection

```python
# Run this Python script to test connection
from pymongo import MongoClient
from foodvision_ai.config import settings

try:
    client = MongoClient(settings.mongodb_url)
    db = client[settings.database_name]
    
    # Test connection
    client.server_info()
    print("✅ MongoDB connection successful!")
    print(f"📊 Database: {settings.database_name}")
    print(f"📁 Collections: {db.list_collection_names()}")
    
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
```

### Using MongoDB Compass (GUI)

1. Download: https://www.mongodb.com/try/download/compass
2. Install and open
3. Connect using your connection string
4. Browse databases and collections

---

## Troubleshooting

### Connection Refused
```bash
# Check if MongoDB is running
docker ps  # For Docker
sudo systemctl status mongod  # For Linux
net start MongoDB  # For Windows

# Check port availability
netstat -an | grep 27017
```

### Authentication Failed
- Verify username and password in connection string
- Check `authSource` parameter
- Ensure user has proper permissions

### Network Access (Atlas)
- Add your IP address to Atlas whitelist
- Use 0.0.0.0/0 for development (not recommended for production)

---

## Production Recommendations

1. **Enable Authentication**: Always use username/password
2. **Use SSL/TLS**: Enable encrypted connections
3. **Regular Backups**: Set up automated backups
4. **Monitoring**: Use MongoDB Atlas monitoring or Prometheus
5. **Connection Pooling**: Configure appropriate pool size
6. **Indexes**: Create indexes for frequently queried fields

---

## Next Steps

After MongoDB is running:
1. Update `.env` file with connection string
2. Run `python main.py` to start FoodVisionAI
3. Application will automatically create collections
4. Access web interface at http://localhost:8000

