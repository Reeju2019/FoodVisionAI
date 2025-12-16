"""
Google Drive integration for image storage and public link generation.
"""
import io
import logging
import mimetypes
from typing import Optional, Dict, Any, BinaryIO
from pathlib import Path
import tempfile
import os

from google.oauth2.service_account import Credentials
from google.oauth2.credentials import Credentials as UserCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaFileUpload
from googleapiclient.errors import HttpError
from pydantic_settings import BaseSettings


class DriveConfig(BaseSettings):
    """Google Drive configuration settings."""
    service_account_file: Optional[str] = None
    credentials_json: Optional[str] = None
    folder_id: Optional[str] = None  # Folder to upload images to
    folder_name: str = "FoodVisionAI_Images"
    
    class Config:
        env_file = ".env"
        env_prefix = "GOOGLE_DRIVE_"


class GoogleDriveService:
    """
    Google Drive service for image upload and public link generation.
    
    Handles authentication, file uploads, and public link creation
    with comprehensive error handling and retry logic.
    """
    
    def __init__(self, config: Optional[DriveConfig] = None):
        """
        Initialize Google Drive service.
        
        Args:
            config: Drive configuration settings
        """
        self.config = config or DriveConfig()
        self.service = None
        self.folder_id = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize service
        self._initialize_service()
    
    def _initialize_service(self):
        """Initialize Google Drive API service with authentication."""
        try:
            credentials = self._get_credentials()
            if credentials:
                self.service = build('drive', 'v3', credentials=credentials)
                self._ensure_folder_exists()
                self.logger.info("Google Drive service initialized successfully")
            else:
                self.logger.debug("No Google Drive credentials found, service will use fallback mode")
        except Exception as e:
            self.logger.error(f"Failed to initialize Google Drive service: {e}")
            self.service = None
    
    def _get_credentials(self) -> Optional[Credentials]:
        """
        Get Google Drive API credentials.
        
        Returns:
            Credentials object or None if not available
        """
        try:
            # Try service account file first
            if self.config.service_account_file and os.path.exists(self.config.service_account_file):
                credentials = Credentials.from_service_account_file(
                    self.config.service_account_file,
                    scopes=['https://www.googleapis.com/auth/drive.file']
                )
                self.logger.info("Using service account credentials")
                return credentials
            
            # Try credentials JSON string
            if self.config.credentials_json:
                import json
                credentials_info = json.loads(self.config.credentials_json)
                credentials = Credentials.from_service_account_info(
                    credentials_info,
                    scopes=['https://www.googleapis.com/auth/drive.file']
                )
                self.logger.info("Using credentials from JSON string")
                return credentials
            
            self.logger.debug("No Google Drive credentials configured")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load Google Drive credentials: {e}")
            return None
    
    def _ensure_folder_exists(self):
        """Ensure the upload folder exists and get its ID."""
        if not self.service:
            return
        
        try:
            # Use configured folder ID if available
            if self.config.folder_id:
                self.folder_id = self.config.folder_id
                self.logger.info(f"Using configured folder ID: {self.folder_id}")
                return
            
            # Search for existing folder
            query = f"name='{self.config.folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = self.service.files().list(q=query, fields="files(id, name)").execute()
            folders = results.get('files', [])
            
            if folders:
                self.folder_id = folders[0]['id']
                self.logger.info(f"Found existing folder: {self.config.folder_name} (ID: {self.folder_id})")
            else:
                # Create new folder
                folder_metadata = {
                    'name': self.config.folder_name,
                    'mimeType': 'application/vnd.google-apps.folder'
                }
                
                folder = self.service.files().create(
                    body=folder_metadata,
                    fields='id'
                ).execute()
                
                self.folder_id = folder.get('id')
                self.logger.info(f"Created new folder: {self.config.folder_name} (ID: {self.folder_id})")
                
        except HttpError as e:
            self.logger.error(f"Failed to ensure folder exists: {e}")
            self.folder_id = None
    
    async def upload_image(
        self, 
        file_content: BinaryIO, 
        filename: str, 
        content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload an image to Google Drive and generate public link.
        
        Args:
            file_content: Binary file content
            filename: Name for the uploaded file
            content_type: MIME type of the file
            
        Returns:
            Dictionary with upload results including public URL
        """
        if not self.service:
            return self._fallback_upload(file_content, filename, content_type)
        
        try:
            # Validate file type
            if not self._is_valid_image_type(filename, content_type):
                raise ValueError(f"Invalid image type for file: {filename}")
            
            # Prepare file metadata
            file_metadata = {
                'name': filename,
                'parents': [self.folder_id] if self.folder_id else []
            }
            
            # Determine content type
            if not content_type:
                content_type, _ = mimetypes.guess_type(filename)
                if not content_type:
                    content_type = 'application/octet-stream'
            
            # Create media upload
            media = MediaIoBaseUpload(
                file_content,
                mimetype=content_type,
                resumable=True
            )
            
            # Upload file
            file_result = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id,name,webViewLink,webContentLink'
            ).execute()
            
            file_id = file_result.get('id')
            
            # Make file publicly accessible
            public_url = await self._make_file_public(file_id)
            
            result = {
                'success': True,
                'file_id': file_id,
                'filename': filename,
                'public_url': public_url,
                'web_view_link': file_result.get('webViewLink'),
                'web_content_link': file_result.get('webContentLink'),
                'upload_method': 'google_drive'
            }
            
            self.logger.info(f"Successfully uploaded {filename} to Google Drive (ID: {file_id})")
            return result
            
        except HttpError as e:
            self.logger.error(f"Google Drive upload failed: {e}")
            return {
                'success': False,
                'error': f"Google Drive upload failed: {str(e)}",
                'filename': filename,
                'upload_method': 'google_drive'
            }
        except Exception as e:
            self.logger.error(f"Upload error: {e}")
            return {
                'success': False,
                'error': f"Upload error: {str(e)}",
                'filename': filename,
                'upload_method': 'google_drive'
            }
    
    async def _make_file_public(self, file_id: str) -> str:
        """
        Make a file publicly accessible and return the public URL.
        
        Args:
            file_id: Google Drive file ID
            
        Returns:
            Public URL for the file
        """
        try:
            # Set file permissions to public
            permission = {
                'role': 'reader',
                'type': 'anyone'
            }
            
            self.service.permissions().create(
                fileId=file_id,
                body=permission
            ).execute()
            
            # Generate public URL
            public_url = f"https://drive.google.com/uc?id={file_id}&export=download"
            
            self.logger.info(f"Made file {file_id} publicly accessible")
            return public_url
            
        except HttpError as e:
            self.logger.error(f"Failed to make file public: {e}")
            # Return a fallback URL that might still work
            return f"https://drive.google.com/file/d/{file_id}/view"
    
    def _fallback_upload(
        self, 
        file_content: BinaryIO, 
        filename: str, 
        content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fallback upload method when Google Drive is not available.
        
        This saves the file locally and returns a local file path.
        In production, this could be replaced with another cloud storage service.
        
        Args:
            file_content: Binary file content
            filename: Name for the uploaded file
            content_type: MIME type of the file
            
        Returns:
            Dictionary with fallback upload results
        """
        try:
            # Create uploads directory
            uploads_dir = Path("uploads")
            uploads_dir.mkdir(exist_ok=True)
            
            # Generate unique filename
            import uuid
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = uploads_dir / unique_filename
            
            # Save file locally
            with open(file_path, 'wb') as f:
                file_content.seek(0)
                f.write(file_content.read())
            
            # Generate local URL (this would need to be served by the web server)
            local_url = f"/uploads/{unique_filename}"
            
            result = {
                'success': True,
                'file_id': unique_filename,
                'filename': filename,
                'public_url': local_url,
                'local_path': str(file_path),
                'upload_method': 'local_fallback'
            }
            
            self.logger.info(f"Fallback upload: saved {filename} locally as {unique_filename}")
            return result
            
        except Exception as e:
            self.logger.error(f"Fallback upload failed: {e}")
            return {
                'success': False,
                'error': f"Fallback upload failed: {str(e)}",
                'filename': filename,
                'upload_method': 'local_fallback'
            }
    
    def _is_valid_image_type(self, filename: str, content_type: Optional[str] = None) -> bool:
        """
        Validate if the file is a supported image type.
        
        Args:
            filename: Name of the file
            content_type: MIME type of the file
            
        Returns:
            True if valid image type, False otherwise
        """
        # Check by content type
        if content_type:
            valid_types = [
                'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 
                'image/bmp', 'image/webp', 'image/tiff'
            ]
            if content_type.lower() in valid_types:
                return True
        
        # Check by file extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
        file_ext = Path(filename).suffix.lower()
        return file_ext in valid_extensions
    
    async def delete_file(self, file_id: str) -> bool:
        """
        Delete a file from Google Drive.
        
        Args:
            file_id: Google Drive file ID
            
        Returns:
            True if deletion succeeded, False otherwise
        """
        if not self.service:
            self.logger.warning("Cannot delete file: Google Drive service not available")
            return False
        
        try:
            self.service.files().delete(fileId=file_id).execute()
            self.logger.info(f"Successfully deleted file {file_id}")
            return True
            
        except HttpError as e:
            self.logger.error(f"Failed to delete file {file_id}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Delete error for file {file_id}: {e}")
            return False
    
    async def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a file in Google Drive.
        
        Args:
            file_id: Google Drive file ID
            
        Returns:
            File information dictionary or None if not found
        """
        if not self.service:
            return None
        
        try:
            file_info = self.service.files().get(
                fileId=file_id,
                fields='id,name,size,mimeType,createdTime,modifiedTime,webViewLink'
            ).execute()
            
            return {
                'id': file_info.get('id'),
                'name': file_info.get('name'),
                'size': file_info.get('size'),
                'mime_type': file_info.get('mimeType'),
                'created_time': file_info.get('createdTime'),
                'modified_time': file_info.get('modifiedTime'),
                'web_view_link': file_info.get('webViewLink')
            }
            
        except HttpError as e:
            self.logger.error(f"Failed to get file info for {file_id}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error getting file info for {file_id}: {e}")
            return None
    
    def is_available(self) -> bool:
        """
        Check if Google Drive service is available.
        
        Returns:
            True if service is available, False otherwise
        """
        return self.service is not None


# Convenience functions
async def upload_image_to_drive(
    file_content: BinaryIO, 
    filename: str, 
    content_type: Optional[str] = None,
    config: Optional[DriveConfig] = None
) -> Dict[str, Any]:
    """
    Convenience function to upload an image to Google Drive.
    
    Args:
        file_content: Binary file content
        filename: Name for the uploaded file
        content_type: MIME type of the file
        config: Optional drive configuration
        
    Returns:
        Upload result dictionary
    """
    drive_service = GoogleDriveService(config)
    return await drive_service.upload_image(file_content, filename, content_type)


def validate_image_file(filename: str, content_type: Optional[str] = None) -> bool:
    """
    Validate if a file is a supported image type.
    
    Args:
        filename: Name of the file
        content_type: MIME type of the file
        
    Returns:
        True if valid image type, False otherwise
    """
    drive_service = GoogleDriveService()
    return drive_service._is_valid_image_type(filename, content_type)