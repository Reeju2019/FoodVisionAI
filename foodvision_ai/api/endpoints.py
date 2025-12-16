"""
API endpoints for FoodVisionAI application.
"""
import io
import logging
import uuid
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from motor.motor_asyncio import AsyncIOMotorDatabase

from ..database import get_database, DatabaseOperations
from ..database.models import (
    CreateAnalysisRequest,
    AnalysisStatusResponse,
    ModelRemarkEntry
)
from ..models.academic_pipeline import AcademicFoodPipeline


# Create routers
upload_router = APIRouter()
status_router = APIRouter()
analytics_router = APIRouter()

logger = logging.getLogger(__name__)


async def get_db_operations(db: AsyncIOMotorDatabase = Depends(get_database)) -> DatabaseOperations:
    """Dependency to get database operations instance."""
    return DatabaseOperations(db)


@upload_router.post("/upload", response_model=dict)
async def upload_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db_ops: DatabaseOperations = Depends(get_db_operations)
):
    """
    Upload endpoint for triggering analysis pipeline.
    
    This endpoint:
    1. Validates the uploaded file
    2. Generates a unique image ID
    3. Creates a database record
    4. Triggers the analysis pipeline in the background
    5. Returns the image ID for status tracking
    """
    try:
        # Validate file type using Google Drive service
        from ..services.google_drive import validate_image_file
        if not validate_image_file(file.filename, file.content_type):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only image files (JPEG, PNG, GIF, BMP, WebP, TIFF) are allowed."
            )
        
        # Validate file size (10MB limit)
        if file.size and file.size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum size is 10MB."
            )
        
        # Generate unique image ID
        image_id = str(uuid.uuid4())
        
        # Upload to Google Drive
        from ..services.google_drive import GoogleDriveService
        drive_service = GoogleDriveService()
        
        # Read file content
        file_content = await file.read()
        file_stream = io.BytesIO(file_content)
        
        # Upload to Google Drive
        upload_result = await drive_service.upload_image(
            file_content=file_stream,
            filename=f"{image_id}_{file.filename}",
            content_type=file.content_type
        )
        
        if not upload_result.get('success'):
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload image: {upload_result.get('error', 'Unknown error')}"
            )
        
        image_url = upload_result['public_url']
        
        # Create database record
        create_request = CreateAnalysisRequest(
            image_id=image_id,
            image_url=image_url
        )
        
        record = await db_ops.create_analysis_record(create_request)
        
        # Add initial remark
        initial_remark = ModelRemarkEntry(
            component="upload_handler",
            status="success",
            message=f"Image uploaded successfully: {file.filename}"
        )
        await db_ops.add_model_remark(image_id, initial_remark)
        
        # Trigger academic analysis pipeline in background
        from ..operator.academic_integration import trigger_academic_analysis_pipeline
        background_tasks.add_task(trigger_academic_analysis_pipeline, image_id, image_url)
        
        logger.info(f"Image uploaded successfully with ID: {image_id}")
        
        return {
            "image_id": image_id,
            "status": "uploaded",
            "message": "Image uploaded successfully. Analysis pipeline will begin shortly.",
            "analytics_url": f"/analytics/{image_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process upload"
        )


@status_router.get("/status/{image_id}", response_model=AnalysisStatusResponse)
async def get_analysis_status(
    image_id: str,
    db_ops: DatabaseOperations = Depends(get_db_operations)
):
    """
    Get real-time analysis status for polling.
    
    This endpoint provides current processing status and results
    for the frontend to display real-time updates.
    """
    try:
        status = await db_ops.get_analysis_status(image_id)
        
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis record not found for image_id: {image_id}"
            )
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status retrieval error for {image_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve analysis status"
        )


@analytics_router.get("/analytics/{image_id}", response_class=HTMLResponse)
async def get_analytics_page(
    image_id: str,
    db_ops: DatabaseOperations = Depends(get_db_operations)
):
    """
    Serve the analytics page for displaying results.
    
    This endpoint returns an HTML page that will poll the status endpoint
    for real-time updates and display the analysis results.
    """
    try:
        # Verify the image_id exists
        record = await db_ops.get_analysis_record(image_id)
        if not record:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis record not found for image_id: {image_id}"
            )
        
        from pathlib import Path
        
        try:
            # Try to serve the new frontend file
            frontend_path = Path("frontend/analytics.html")
            if frontend_path.exists():
                with open(frontend_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    # Replace placeholder with actual image_id
                    html_content = html_content.replace('{{IMAGE_ID}}', image_id)
                    return HTMLResponse(content=html_content)
        except Exception as e:
            logger.warning(f"Could not load frontend analytics file: {e}")
        
        # Fallback to embedded HTML
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>FoodVisionAI - Analysis Results</title>
            <script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-gray-100 min-h-screen">
            <div class="container mx-auto px-4 py-8">
                <h1 class="text-3xl font-bold text-center mb-8">FoodVisionAI Analysis</h1>
                <div class="max-w-4xl mx-auto bg-white rounded-lg shadow-lg p-6">
                    <div class="mb-4">
                        <h2 class="text-xl font-semibold mb-2">Image ID: {image_id}</h2>
                        <div id="status-container">
                            <p class="text-gray-600">Loading analysis results...</p>
                        </div>
                    </div>
                    
                    <div id="results-container" class="hidden">
                        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                            <div class="bg-blue-50 p-4 rounded-lg">
                                <h3 class="font-semibold text-blue-800 mb-2">Vision Analysis</h3>
                                <div id="vision-results"></div>
                            </div>
                            <div class="bg-green-50 p-4 rounded-lg">
                                <h3 class="font-semibold text-green-800 mb-2">Nutrition Facts</h3>
                                <div id="nutrition-results"></div>
                            </div>
                            <div class="bg-purple-50 p-4 rounded-lg">
                                <h3 class="font-semibold text-purple-800 mb-2">Cuisine Classification</h3>
                                <div id="cuisine-results"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                // Simple polling for status updates
                async function pollStatus() {{
                    try {{
                        const response = await fetch('/api/v1/status/{image_id}');
                        const data = await response.json();
                        
                        // Update status display
                        document.getElementById('status-container').innerHTML = 
                            `<p class="text-sm text-gray-600">Status: ${{data.status}}</p>`;
                        
                        // Show results if available
                        if (Object.keys(data.results).length > 0) {{
                            document.getElementById('results-container').classList.remove('hidden');
                            
                            // Update vision results
                            if (data.results.vision) {{
                                document.getElementById('vision-results').innerHTML = 
                                    `<p><strong>Ingredients:</strong> ${{data.results.vision.ingredients?.join(', ') || 'Processing...'}}</p>
                                     <p><strong>Description:</strong> ${{data.results.vision.description || 'Processing...'}}</p>`;
                            }}
                            
                            // Update nutrition results
                            if (data.results.nutrition) {{
                                document.getElementById('nutrition-results').innerHTML = 
                                    `<p><strong>Calories:</strong> ${{data.results.nutrition.calories || 0}}</p>
                                     <p><strong>Fat:</strong> ${{data.results.nutrition.fat || 0}}g</p>
                                     <p><strong>Carbs:</strong> ${{data.results.nutrition.carbohydrates || 0}}g</p>
                                     <p><strong>Protein:</strong> ${{data.results.nutrition.protein || 0}}g</p>`;
                            }}
                            
                            // Update cuisine results
                            if (data.results.cuisine) {{
                                const cuisines = data.results.cuisine.cuisines || [];
                                document.getElementById('cuisine-results').innerHTML = 
                                    cuisines.map(c => `<p><strong>${{c.name}}:</strong> ${{(c.confidence * 100).toFixed(1)}}%</p>`).join('');
                            }}
                        }}
                        
                        // Continue polling if still processing
                        if (data.status === 'processing') {{
                            setTimeout(pollStatus, 2000);
                        }}
                        
                    }} catch (error) {{
                        console.error('Error polling status:', error);
                        document.getElementById('status-container').innerHTML = 
                            '<p class="text-red-600">Error loading status</p>';
                    }}
                }}
                
                // Start polling
                pollStatus();
            </script>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analytics page error for {image_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to load analytics page"
        )


@upload_router.get("/uploads/{filename}")
async def serve_uploaded_file(filename: str):
    """
    Serve uploaded files for local fallback mode.
    
    This endpoint serves files that were uploaded using the local fallback
    when Google Drive is not available.
    """
    from pathlib import Path
    from fastapi.responses import FileResponse
    
    try:
        file_path = Path("uploads") / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Validate that the file is within the uploads directory (security)
        if not str(file_path.resolve()).startswith(str(Path("uploads").resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
        
        return FileResponse(
            path=file_path,
            media_type="image/*",
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving file {filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve file")


@upload_router.get("/upload", response_class=HTMLResponse)
async def get_upload_page():
    """
    Serve the upload page for file uploads.
    
    This endpoint returns an HTML page with drag-and-drop functionality
    for uploading food images.
    """
    from pathlib import Path
    
    try:
        # Try to serve the new frontend file
        frontend_path = Path("frontend/upload.html")
        if frontend_path.exists():
            with open(frontend_path, 'r', encoding='utf-8') as f:
                return HTMLResponse(content=f.read())
    except Exception as e:
        logger.warning(f"Could not load frontend file: {e}")
    
    # Fallback to embedded HTML
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FoodVisionAI - Upload</title>
        <script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-100 min-h-screen">
        <div class="container mx-auto px-4 py-8">
            <h1 class="text-3xl font-bold text-center mb-8">FoodVisionAI</h1>
            <div class="max-w-2xl mx-auto bg-white rounded-lg shadow-lg p-6">
                <h2 class="text-xl font-semibold mb-4">Upload Food Image</h2>
                
                <div class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
                    <input type="file" id="fileInput" accept="image/*" class="hidden">
                    <div id="dropZone" class="cursor-pointer">
                        <svg class="mx-auto h-12 w-12 text-gray-400 mb-4" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                            <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
                        </svg>
                        <p class="text-lg text-gray-600 mb-2">Drop your food image here</p>
                        <p class="text-sm text-gray-500 mb-4">or click to browse</p>
                        <button type="button" class="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg">
                            Choose File
                        </button>
                    </div>
                </div>
                
                <div id="uploadStatus" class="mt-4 hidden">
                    <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
                        <p class="text-blue-800">Uploading...</p>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            const fileInput = document.getElementById('fileInput');
            const dropZone = document.getElementById('dropZone');
            const uploadStatus = document.getElementById('uploadStatus');
            
            // Click to browse
            dropZone.addEventListener('click', () => fileInput.click());
            
            // Drag and drop
            dropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                dropZone.classList.add('bg-blue-50', 'border-blue-300');
            });
            
            dropZone.addEventListener('dragleave', () => {
                dropZone.classList.remove('bg-blue-50', 'border-blue-300');
            });
            
            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.classList.remove('bg-blue-50', 'border-blue-300');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    handleFile(files[0]);
                }
            });
            
            // File input change
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    handleFile(e.target.files[0]);
                }
            });
            
            async function handleFile(file) {
                // Validate file type
                if (!file.type.startsWith('image/')) {
                    alert('Please select an image file');
                    return;
                }
                
                // Show upload status
                uploadStatus.classList.remove('hidden');
                
                // Create form data
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/api/v1/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        // Redirect to analytics page
                        window.location.href = `/api/v1/analytics/${result.image_id}`;
                    } else {
                        throw new Error(result.detail || 'Upload failed');
                    }
                    
                } catch (error) {
                    console.error('Upload error:', error);
                    uploadStatus.innerHTML = `
                        <div class="bg-red-50 border border-red-200 rounded-lg p-4">
                            <p class="text-red-800">Upload failed: ${error.message}</p>
                        </div>
                    `;
                }
            }
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)