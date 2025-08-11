"""
File Upload API for Retail Intelligence Platform
Handles CSV upload, validation, and initial processing
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import pandas as pd
import os
import uuid
from datetime import datetime
from pathlib import Path
import logging
import aiofiles
from typing import Dict, Any

from config import settings
from core.data_processor import RetailDataProcessor

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize data processor
data_processor = RetailDataProcessor()

# Store processing results temporarily (in production, use Redis or database)
processing_results = {}

@router.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
) -> JSONResponse:
    """
    Upload and process retail CSV file
    
    Args:
        file: CSV file upload
        
    Returns:
        Processing results with data analysis and insights
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file extension
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"File type not allowed. Supported: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
        
        # Check file size
        file_content = await file.read()
        if len(file_content) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file_id}_{file.filename}"
        
        # Save file to upload directory
        file_path = settings.UPLOAD_DIR / safe_filename
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)
        
        logger.info(f"File uploaded: {safe_filename} ({len(file_content)} bytes)")
        
        # Process file in background
        background_tasks.add_task(process_file_background, file_id, str(file_path), file.filename)
        
        return JSONResponse(
            status_code=202,
            content={
                "success": True,
                "data": {
                    "file_id": file_id,
                    "filename": file.filename,
                    "status": "processing"
                },
                "message": "File uploaded successfully. Processing started."
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/upload/status/{file_id}")
async def get_upload_status(file_id: str) -> JSONResponse:
    """
    Get processing status for uploaded file
    
    Args:
        file_id: Unique file identifier
        
    Returns:
        Processing status and results
    """
    try:
        if file_id not in processing_results:
            raise HTTPException(status_code=404, detail="File not found")
        
        result = processing_results[file_id]
        
        return JSONResponse(content={
            "success": True,
            "file_id": file_id,
            "status": result.get("status", "unknown"),
            "progress": result.get("progress", 0),
            "results": result.get("results"),
            "error": result.get("error")
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/upload/results/{file_id}")
async def get_processing_results(file_id: str) -> JSONResponse:
    """
    Get detailed processing results
    
    Args:
        file_id: Unique file identifier
        
    Returns:
        Complete processing results including data analysis
    """
    try:
        if file_id not in processing_results:
            raise HTTPException(status_code=404, detail="File not found")
        
        result = processing_results[file_id]
        
        if result.get("status") != "completed":
            return JSONResponse(content={
                "success": False,
                "message": f"Processing not completed. Status: {result.get('status')}",
                "status": result.get("status")
            })
        
        return JSONResponse(content={
            "success": True,
            "file_id": file_id,
            "results": result.get("results"),
            "processed_at": result.get("completed_at")
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Results retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/upload/{file_id}")
async def delete_uploaded_file(file_id: str) -> JSONResponse:
    """
    Delete uploaded file and processing results
    
    Args:
        file_id: Unique file identifier
        
    Returns:
        Deletion confirmation
    """
    try:
        # Remove from processing results
        if file_id in processing_results:
            result = processing_results[file_id]
            
            # Delete physical file if exists
            if "file_path" in result:
                file_path = Path(result["file_path"])
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted file: {file_path}")
            
            # Remove from memory
            del processing_results[file_id]
        
        return JSONResponse(content={
            "success": True,
            "message": "File deleted successfully",
            "file_id": file_id
        })
        
    except Exception as e:
        logger.error(f"Deletion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/upload/list")
async def list_uploaded_files() -> JSONResponse:
    """
    List all uploaded files and their status
    
    Returns:
        List of uploaded files with status
    """
    try:
        files_list = []
        
        for file_id, result in processing_results.items():
            files_list.append({
                "file_id": file_id,
                "filename": result.get("filename", "Unknown"),
                "status": result.get("status", "unknown"),
                "uploaded_at": result.get("uploaded_at"),
                "completed_at": result.get("completed_at"),
                "file_size": result.get("file_size", 0)
            })
        
        # Sort by upload time (newest first)
        files_list.sort(key=lambda x: x.get("uploaded_at", ""), reverse=True)
        
        return JSONResponse(content={
            "success": True,
            "files": files_list,
            "total_files": len(files_list)
        })
        
    except Exception as e:
        logger.error(f"File listing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_file_background(file_id: str, file_path: str, original_filename: str):
    """
    Background task to process uploaded file
    
    Args:
        file_id: Unique file identifier
        file_path: Path to uploaded file
        original_filename: Original filename
    """
    try:
        # Initialize processing status
        processing_results[file_id] = {
            "status": "processing",
            "progress": 0,
            "file_path": file_path,
            "filename": original_filename,
            "uploaded_at": datetime.now().isoformat(),
            "file_size": os.path.getsize(file_path)
        }
        
        logger.info(f"Starting processing for file: {file_id}")
        
        # Update progress
        processing_results[file_id]["progress"] = 25
        processing_results[file_id]["status"] = "analyzing"
        
        # Process the file
        results = data_processor.process_csv(file_path)
        
        # Update progress
        processing_results[file_id]["progress"] = 75
        
        if results["success"]:
            # Save processed data
            processed_filename = f"processed_{file_id}.csv"
            processed_path = settings.PROCESSED_DIR / processed_filename
            
            # Save processed DataFrame
            results["data"].to_csv(processed_path, index=False)
            
            # Update results (remove DataFrame for JSON serialization)
            results_copy = results.copy()
            results_copy["data"] = f"Saved to {processed_filename}"
            results_copy["processed_file_path"] = str(processed_path)
            
            # Mark as completed
            processing_results[file_id].update({
                "status": "completed",
                "progress": 100,
                "results": results_copy,
                "completed_at": datetime.now().isoformat()
            })
            
            logger.info(f"Processing completed successfully for file: {file_id}")
            
        else:
            # Mark as failed
            processing_results[file_id].update({
                "status": "failed",
                "progress": 100,
                "error": results.get("error", "Unknown error"),
                "completed_at": datetime.now().isoformat()
            })
            
            logger.error(f"Processing failed for file: {file_id} - {results.get('error')}")
        
    except Exception as e:
        # Mark as failed
        processing_results[file_id].update({
            "status": "failed",
            "progress": 100,
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })
        
        logger.error(f"Background processing error for file {file_id}: {e}")

@router.post("/upload/validate")
async def validate_csv_structure(file: UploadFile = File(...)) -> JSONResponse:
    """
    Validate CSV structure without full processing
    
    Args:
        file: CSV file upload
        
    Returns:
        Validation results and column analysis
    """
    try:
        if not file.filename or not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="CSV file required")
        
        # Read file content
        file_content = await file.read()
        
        # Create temporary file for validation
        temp_file_id = str(uuid.uuid4())
        temp_path = settings.UPLOAD_DIR / f"temp_{temp_file_id}.csv"
        
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(file_content)
        
        try:
            # Quick validation
            df = pd.read_csv(temp_path, nrows=5)  # Read only first 5 rows
            
            # Analyze columns
            column_analysis = data_processor._analyze_columns(df)
            
            # Basic validation
            validation = {
                "valid": True,
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "detected_columns": {
                    "date_column": column_analysis.get('date_column'),
                    "sales_column": column_analysis.get('sales_column'),
                    "product_column": column_analysis.get('product_column'),
                    "quantity_column": column_analysis.get('quantity_column')
                },
                "issues": [],
                "recommendations": []
            }
            
            # Check for required columns
            if not column_analysis.get('date_column'):
                validation["issues"].append("No date column detected")
                validation["recommendations"].append("Ensure date column is named 'date', 'order_date', or similar")
            
            if not column_analysis.get('sales_column'):
                validation["issues"].append("No sales/revenue column detected")
                validation["recommendations"].append("Ensure sales column is named 'sales', 'revenue', 'amount', or similar")
            
            if validation["issues"]:
                validation["valid"] = False
            
            return JSONResponse(content={
                "success": True,
                "validation": validation
            })
            
        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.get("/upload/sample")
async def get_sample_data_format() -> JSONResponse:
    """
    Get sample CSV format for user reference
    
    Returns:
        Sample data structure and requirements
    """
    try:
        sample_data = {
            "required_columns": {
                "date": {
                    "description": "Transaction date",
                    "format": "YYYY-MM-DD or DD/MM/YYYY",
                    "examples": ["2024-01-15", "15/01/2024"]
                },
                "sales": {
                    "description": "Sales amount in Indian Rupees",
                    "format": "Numeric value",
                    "examples": ["1500.50", "2000", "₹1500"]
                }
            },
            "optional_columns": {
                "product": {
                    "description": "Product name or SKU",
                    "examples": ["iPhone 15", "SKU-001", "Laptop"]
                },
                "quantity": {
                    "description": "Quantity sold",
                    "examples": ["1", "5", "10"]
                },
                "customer": {
                    "description": "Customer ID or name",
                    "examples": ["CUST001", "John Doe"]
                }
            },
            "sample_csv": [
                {
                    "date": "2024-01-01",
                    "sales": "1500.00",
                    "product": "Product A",
                    "quantity": "2",
                    "customer": "CUST001"
                },
                {
                    "date": "2024-01-02",
                    "sales": "2000.00",
                    "product": "Product B",
                    "quantity": "1",
                    "customer": "CUST002"
                }
            ],
            "tips": [
                "Ensure date format is consistent throughout the file",
                "Remove currency symbols (₹) from sales amounts - use numbers only",
                "Use clear column names like 'date', 'sales', 'product'",
                "Include at least 30 rows for meaningful analysis",
                "Avoid empty cells in date and sales columns"
            ]
        }
        
        return JSONResponse(content={
            "success": True,
            "sample_format": sample_data
        })
        
    except Exception as e:
        logger.error(f"Sample data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))