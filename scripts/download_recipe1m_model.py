#!/usr/bin/env python3
"""
Script to download Recipe1M+ trained model from Google Drive

Run this after training in Colab to download the model to your local project.
"""

import os
import sys
import gdown
from pathlib import Path

def download_recipe1m_model():
    """
    Download Recipe1M+ trained model from Google Drive
    """
    print("🔽 Downloading Recipe1M+ trained model...")
    
    # Create models directory if it doesn't exist
    models_dir = Path("foodvision_ai/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Google Drive file ID (replace with your actual file ID after Colab training)
    # You'll get this ID from the Google Drive share link
    file_id = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"
    
    # Download URL
    url = f"https://drive.google.com/uc?id={file_id}"
    
    # Output path
    output_path = models_dir / "recipe1m_trained_model_final.pth"
    
    try:
        # Download file
        gdown.download(url, str(output_path), quiet=False)
        
        # Verify download
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✅ Model downloaded successfully!")
            print(f"📁 Location: {output_path}")
            print(f"📊 Size: {file_size:.1f} MB")
            return True
        else:
            print("❌ Download failed - file not found")
            return False
            
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\n💡 Manual download instructions:")
        print("1. Go to your Google Drive")
        print("2. Find 'recipe1m_trained_model_final.pth'")
        print("3. Download it manually")
        print(f"4. Place it in: {output_path}")
        return False

def setup_dependencies():
    """
    Install required dependencies for model loading
    """
    print("📦 Installing dependencies...")
    
    try:
        import gdown
        print("✅ gdown already installed")
    except ImportError:
        print("Installing gdown...")
        os.system("pip install gdown")
    
    try:
        import timm
        print("✅ timm already installed")
    except ImportError:
        print("Installing timm...")
        os.system("pip install timm")

def verify_integration():
    """
    Verify that the model can be loaded
    """
    print("🔍 Verifying model integration...")
    
    try:
        # Try to import the architecture
        sys.path.append('.')
        from foodvision_ai.models.recipe1m_architecture import Recipe1MModelLoader
        
        model_path = "foodvision_ai/models/recipe1m_trained_model_final.pth"
        
        if os.path.exists(model_path):
            # Try to load the model
            loader = Recipe1MModelLoader(model_path, device='cpu')
            print("✅ Model loaded successfully!")
            print("🎉 Integration complete - your app can now use Recipe1M+ model!")
            return True
        else:
            print(f"❌ Model file not found: {model_path}")
            return False
            
    except Exception as e:
        print(f"❌ Integration verification failed: {e}")
        return False

def main():
    """
    Main function to download and setup Recipe1M+ model
    """
    print("🚀 Recipe1M+ Model Setup")
    print("=" * 50)
    
    # Step 1: Setup dependencies
    setup_dependencies()
    
    # Step 2: Download model
    print("\n" + "=" * 50)
    download_success = download_recipe1m_model()
    
    if not download_success:
        print("\n⚠️  Manual setup required:")
        print("1. Complete training in Colab")
        print("2. Download the .pth file from Google Drive")
        print("3. Place it in foodvision_ai/models/recipe1m_trained_model_final.pth")
        print("4. Run this script again")
        return
    
    # Step 3: Verify integration
    print("\n" + "=" * 50)
    verify_integration()
    
    print("\n🎉 Setup complete!")
    print("Your FoodVisionAI app now has Recipe1M+ integration!")

if __name__ == "__main__":
    main()