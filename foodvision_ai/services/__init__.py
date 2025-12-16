"""
External services integration for FoodVisionAI

Contains integrations with Google Drive and other external services.
"""

from .google_drive import GoogleDriveService, DriveConfig

__all__ = [
    "GoogleDriveService",
    "DriveConfig"
]