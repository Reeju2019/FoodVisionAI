"""
Database operations for storing and retrieving analysis results.
"""
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo.errors import DuplicateKeyError, PyMongoError
from pymongo import ReturnDocument

from .models import (
    FoodAnalysisRecord, 
    CreateAnalysisRequest, 
    UpdateAnalysisRequest,
    AnalysisStatusResponse,
    ModelRemarkEntry,
    VisionModelResult,
    NutritionModelResult,
    CuisineModelResult,
    ProcessingStatus,
    ProgressInfo
)


class DatabaseOperations:
    """Database operations for food analysis records."""
    
    def __init__(self, database: AsyncIOMotorDatabase):
        self.db = database
        self.collection: AsyncIOMotorCollection = database.food_analysis
        self.logger = logging.getLogger(__name__)
    
    async def create_analysis_record(self, request: CreateAnalysisRequest) -> FoodAnalysisRecord:
        """
        Create a new analysis record in the database.
        
        Args:
            request: Request containing image_id and image_url
            
        Returns:
            FoodAnalysisRecord: Created record
            
        Raises:
            DuplicateKeyError: If image_id already exists
            PyMongoError: For other database errors
        """
        try:
            # Create initial record with proper structure
            record = FoodAnalysisRecord(
                id=request.image_id,
                description_json={
                    "status": ProcessingStatus.PROCESSING,
                    "image_url": request.image_url,
                    "vision_model": {},
                    "nutrition_model": {},
                    "cuisine_model": {},
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                },
                in_progress=True,
                is_error=False,
                model_remark=[]
            )
            
            # Convert to dict for MongoDB insertion
            record_dict = record.dict(by_alias=True)
            
            # Insert into database
            await self.collection.insert_one(record_dict)
            
            self.logger.info(f"Created analysis record for image_id: {request.image_id}")
            return record
            
        except DuplicateKeyError:
            self.logger.error(f"Analysis record already exists for image_id: {request.image_id}")
            raise
        except PyMongoError as e:
            self.logger.error(f"Database error creating record: {e}")
            raise
    
    async def get_analysis_record(self, image_id: str) -> Optional[FoodAnalysisRecord]:
        """
        Retrieve an analysis record by image_id.
        
        Args:
            image_id: Unique identifier for the image
            
        Returns:
            FoodAnalysisRecord or None if not found
        """
        try:
            record_dict = await self.collection.find_one({"_id": image_id})
            if record_dict:
                return FoodAnalysisRecord(**record_dict)
            return None
        except PyMongoError as e:
            self.logger.error(f"Database error retrieving record {image_id}: {e}")
            raise
    
    async def update_analysis_record(
        self, 
        image_id: str, 
        update_request: UpdateAnalysisRequest
    ) -> Optional[FoodAnalysisRecord]:
        """
        Update an existing analysis record.
        
        Args:
            image_id: Unique identifier for the image
            update_request: Update data
            
        Returns:
            Updated FoodAnalysisRecord or None if not found
        """
        try:
            update_data = {"description_json.updated_at": datetime.utcnow()}
            
            # Update individual model results
            if update_request.vision_result:
                update_request.vision_result.timestamp = datetime.utcnow()
                for key, value in update_request.vision_result.dict().items():
                    update_data[f"description_json.vision_model.{key}"] = value
            
            if update_request.nutrition_result:
                update_request.nutrition_result.timestamp = datetime.utcnow()
                for key, value in update_request.nutrition_result.dict().items():
                    update_data[f"description_json.nutrition_model.{key}"] = value
            
            if update_request.cuisine_result:
                update_request.cuisine_result.timestamp = datetime.utcnow()
                for key, value in update_request.cuisine_result.dict().items():
                    update_data[f"description_json.cuisine_model.{key}"] = value
            
            # Update status fields
            if update_request.status:
                update_data["description_json.status"] = update_request.status
            
            if update_request.in_progress is not None:
                update_data["in_progress"] = update_request.in_progress
            
            if update_request.is_error is not None:
                update_data["is_error"] = update_request.is_error
            
            # Handle model remark addition
            push_data = {}
            if update_request.add_remark:
                push_data["model_remark"] = update_request.add_remark.dict()
            
            # Build update query
            update_query = {}
            if update_data:
                update_query["$set"] = update_data
            if push_data:
                update_query["$push"] = push_data
            
            if not update_query:
                return await self.get_analysis_record(image_id)
            
            # Perform update
            result = await self.collection.find_one_and_update(
                {"_id": image_id},
                update_query,
                return_document=ReturnDocument.AFTER
            )
            
            if result:
                self.logger.info(f"Updated analysis record for image_id: {image_id}")
                return FoodAnalysisRecord(**result)
            
            return None
            
        except PyMongoError as e:
            self.logger.error(f"Database error updating record {image_id}: {e}")
            raise
    
    async def add_model_remark(self, image_id: str, remark: ModelRemarkEntry) -> bool:
        """
        Add a remark to the model_remark list.
        
        Args:
            image_id: Unique identifier for the image
            remark: Remark entry to add
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            result = await self.collection.update_one(
                {"_id": image_id},
                {
                    "$push": {"model_remark": remark.dict()},
                    "$set": {"description_json.updated_at": datetime.utcnow()}
                }
            )
            
            success = result.modified_count > 0
            if success:
                self.logger.info(f"Added remark to record {image_id}: {remark.component}")
            
            return success
            
        except PyMongoError as e:
            self.logger.error(f"Database error adding remark to {image_id}: {e}")
            return False
    
    async def get_analysis_status(self, image_id: str) -> Optional[AnalysisStatusResponse]:
        """
        Get current analysis status for real-time updates.
        
        Args:
            image_id: Unique identifier for the image
            
        Returns:
            AnalysisStatusResponse or None if not found
        """
        try:
            record = await self.get_analysis_record(image_id)
            if not record:
                return None
            
            # Build progress information
            progress = {
                "vision": {
                    "completed": record.description_json.vision_model.completed,
                    "progress": 1.0 if record.description_json.vision_model.completed else 0.0
                },
                "nutrition": {
                    "completed": record.description_json.nutrition_model.completed,
                    "progress": 1.0 if record.description_json.nutrition_model.completed else 0.0
                },
                "cuisine": {
                    "completed": record.description_json.cuisine_model.completed,
                    "progress": 1.0 if record.description_json.cuisine_model.completed else 0.0
                }
            }
            
            # Build results
            results = {}
            if record.description_json.vision_model.completed:
                results["vision"] = record.description_json.vision_model.dict()
            if record.description_json.nutrition_model.completed:
                results["nutrition"] = record.description_json.nutrition_model.dict()
            if record.description_json.cuisine_model.completed:
                results["cuisine"] = record.description_json.cuisine_model.dict()
            
            # Extract errors from model remarks
            errors = [
                remark.message for remark in record.model_remark 
                if remark.status == "error"
            ]
            
            return AnalysisStatusResponse(
                image_id=image_id,
                status=record.description_json.status,
                progress=progress,
                results=results,
                errors=errors,
                last_updated=record.description_json.updated_at
            )
            
        except PyMongoError as e:
            self.logger.error(f"Database error getting status for {image_id}: {e}")
            return None
    
    async def list_analysis_records(
        self, 
        limit: int = 50, 
        skip: int = 0,
        status_filter: Optional[ProcessingStatus] = None
    ) -> List[FoodAnalysisRecord]:
        """
        List analysis records with optional filtering.
        
        Args:
            limit: Maximum number of records to return
            skip: Number of records to skip
            status_filter: Optional status filter
            
        Returns:
            List of FoodAnalysisRecord
        """
        try:
            query = {}
            if status_filter:
                query["description_json.status"] = status_filter
            
            cursor = self.collection.find(query).skip(skip).limit(limit)
            records = []
            
            async for record_dict in cursor:
                records.append(FoodAnalysisRecord(**record_dict))
            
            return records
            
        except PyMongoError as e:
            self.logger.error(f"Database error listing records: {e}")
            return []
    
    async def delete_analysis_record(self, image_id: str) -> bool:
        """
        Delete an analysis record.

        Args:
            image_id: Unique identifier for the image

        Returns:
            bool: True if deleted, False if not found
        """
        try:
            result = await self.collection.delete_one({"_id": image_id})
            success = result.deleted_count > 0

            if success:
                self.logger.info(f"Deleted analysis record for image_id: {image_id}")

            return success

        except PyMongoError as e:
            self.logger.error(f"Database error deleting record {image_id}: {e}")
            return False

    # Convenience methods for academic pipeline integration

    async def update_analysis_status(self, image_id: str, status: str) -> bool:
        """
        Update just the status of an analysis record.

        Args:
            image_id: Unique identifier for the image
            status: New status (e.g., "processing", "completed", "failed")

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            result = await self.collection.update_one(
                {"_id": image_id},
                {
                    "$set": {
                        "description_json.status": status,
                        "description_json.updated_at": datetime.utcnow(),
                        "in_progress": status == "processing",
                        "is_error": status == "failed"
                    }
                }
            )

            success = result.modified_count > 0
            if success:
                self.logger.info(f"Updated status for {image_id} to: {status}")

            return success

        except PyMongoError as e:
            self.logger.error(f"Database error updating status for {image_id}: {e}")
            return False

    async def update_analysis_results(self, image_id: str, results: Dict[str, Any]) -> bool:
        """
        Update analysis results from academic pipeline.

        Args:
            image_id: Unique identifier for the image
            results: Dictionary containing vision, nutrition, and other results

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            update_data = {
                "description_json.updated_at": datetime.utcnow()
            }

            # Update vision results
            if "vision" in results:
                for key, value in results["vision"].items():
                    update_data[f"description_json.vision_model.{key}"] = value
                update_data["description_json.vision_model.completed"] = True

            # Update nutrition results
            if "nutrition" in results:
                for key, value in results["nutrition"].items():
                    update_data[f"description_json.nutrition_model.{key}"] = value
                update_data["description_json.nutrition_model.completed"] = True

            # Update cuisine results
            if "cuisine" in results:
                for key, value in results["cuisine"].items():
                    update_data[f"description_json.cuisine_model.{key}"] = value
                update_data["description_json.cuisine_model.completed"] = True

            result = await self.collection.update_one(
                {"_id": image_id},
                {"$set": update_data}
            )

            success = result.modified_count > 0
            if success:
                self.logger.info(f"Updated results for {image_id}")

            return success

        except PyMongoError as e:
            self.logger.error(f"Database error updating results for {image_id}: {e}")
            return False