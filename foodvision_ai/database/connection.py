"""
MongoDB connection utilities with retry logic and error handling.
"""
import asyncio
import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    mongodb_url: str = "mongodb://localhost:27017"
    database_name: str = "foodvision_ai"
    max_connections: int = 100
    min_connections: int = 10
    connection_timeout: int = 10000  # milliseconds
    server_selection_timeout: int = 5000  # milliseconds
    
    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }


class DatabaseConnection:
    """MongoDB connection manager with retry logic."""
    
    def __init__(self, settings: Optional[DatabaseSettings] = None):
        self.settings = settings or DatabaseSettings()
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self.logger = logging.getLogger(__name__)
    
    async def connect(self, max_retries: int = 3, retry_delay: float = 1.0) -> bool:
        """
        Connect to MongoDB with retry logic.
        
        Args:
            max_retries: Maximum number of connection attempts
            retry_delay: Delay between retry attempts in seconds
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                self.client = AsyncIOMotorClient(
                    self.settings.mongodb_url,
                    maxPoolSize=self.settings.max_connections,
                    minPoolSize=self.settings.min_connections,
                    connectTimeoutMS=self.settings.connection_timeout,
                    serverSelectionTimeoutMS=self.settings.server_selection_timeout,
                )
                
                # Test the connection
                await self.client.admin.command('ping')
                self.database = self.client[self.settings.database_name]
                
                self.logger.info(f"Successfully connected to MongoDB: {self.settings.database_name}")
                return True
                
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                self.logger.warning(
                    f"Connection attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    self.logger.error(f"Failed to connect to MongoDB after {max_retries} attempts")
                    return False
        
        return False
    
    async def disconnect(self):
        """Close the database connection."""
        if self.client:
            self.client.close()
            self.logger.info("Disconnected from MongoDB")
    
    def get_database(self) -> AsyncIOMotorDatabase:
        """Get the database instance."""
        if self.database is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self.database
    
    async def health_check(self) -> bool:
        """Check if the database connection is healthy."""
        try:
            if self.client is None:
                return False
            await self.client.admin.command('ping')
            return True
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return False


# Global database connection instance
db_connection = DatabaseConnection()


async def get_database() -> Optional[AsyncIOMotorDatabase]:
    """Get the database instance for dependency injection."""
    try:
        return db_connection.get_database()
    except RuntimeError:
        # Return None if database is not connected (for development/testing)
        return None