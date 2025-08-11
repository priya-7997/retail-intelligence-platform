"""
Retail Intelligence Platform - Main FastAPI Application
Industry-grade retail forecasting platform for Indian market
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import logging
from pathlib import Path

from config import settings
from api import upload, forecast, insights, dashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("retail_intelligence.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered retail intelligence platform for Indian businesses",
    version=settings.APP_VERSION,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files - Updated for correct project structure
frontend_path = Path(__file__).parent.parent / "frontend"

# Primary static mount
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Additional mounts for direct access to scripts and styles
if (frontend_path / "scripts").exists():
    app.mount("/scripts", StaticFiles(directory=frontend_path / "scripts"), name="scripts")
if (frontend_path / "styles").exists():
    app.mount("/styles", StaticFiles(directory=frontend_path / "styles"), name="styles")
if (frontend_path / "assets").exists():
    app.mount("/assets", StaticFiles(directory=frontend_path / "assets"), name="assets")

# Include API routers
app.include_router(upload.router, prefix=f"{settings.API_V1_PREFIX}/upload", tags=["upload"])
app.include_router(forecast.router, prefix=f"{settings.API_V1_PREFIX}/forecast", tags=["forecast"])
app.include_router(insights.router, prefix=f"{settings.API_V1_PREFIX}/insights", tags=["insights"])
app.include_router(dashboard.router, prefix=f"{settings.API_V1_PREFIX}/dashboard", tags=["dashboard"])

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard"""
    try:
        html_file = frontend_path / "index.html"
        if html_file.exists():
            return HTMLResponse(content=html_file.read_text(), status_code=200)
        else:
            return HTMLResponse(
                content="<h1>Retail Intelligence Platform</h1><p>Frontend not found. Please check installation.</p>",
                status_code=200
            )
    except Exception as e:
        logger.error(f"Error serving root page: {e}")
        return HTMLResponse(
            content="<h1>Server Error</h1><p>Please check server logs.</p>",
            status_code=500
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": "development" if settings.DEBUG else "production"
    }

@app.get("/api/info")
async def api_info():
    """API information endpoint"""
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "AI-powered retail intelligence platform for Indian businesses",
        "currency": settings.CURRENCY,
        "locale": settings.LOCALE,
        "features": [
            "Automatic CSV processing",
            "3-model intelligent forecasting",
            "Business insights generation",
            "Indian market optimization",
            "Real-time alerts"
        ]
    }

# Fixed Exception Handlers - Return JSONResponse objects instead of dicts
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "status_code": 404}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle any unhandled exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred", "status_code": 500}
    )

@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info("="*60)
    logger.info(f"STARTING {settings.APP_NAME}")
    logger.info("="*60)
    logger.info(f"Version: {settings.APP_VERSION}")
    logger.info(f"Environment: {'Development' if settings.DEBUG else 'Production'}")
    logger.info(f"Host: {settings.HOST}:{settings.PORT}")
    logger.info(f"Currency: {settings.CURRENCY}")
    logger.info(f"Locale: {settings.LOCALE}")
    logger.info(f"Timezone: {settings.TIMEZONE}")
    
    # Create necessary data directories based on project structure
    data_dir = Path(__file__).parent.parent / "data"
    try:
        # Main data directories
        (data_dir / "uploads").mkdir(parents=True, exist_ok=True)
        (data_dir / "processed").mkdir(parents=True, exist_ok=True)
        (data_dir / "samples").mkdir(parents=True, exist_ok=True)
        (data_dir / "exports").mkdir(parents=True, exist_ok=True)
        
        # Model directories
        model_dir = Path(__file__).parent / "models" / "saved_models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Settings directories (if they exist in config)
        if hasattr(settings, 'UPLOAD_DIR'):
            settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        if hasattr(settings, 'PROCESSED_DIR'):
            settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        if hasattr(settings, 'MODEL_SAVE_DIR'):
            settings.MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
            
        logger.info("Required directories created successfully")
    except Exception as e:
        logger.error(f"Error creating directories: {e}")
    
    # Log static file paths for debugging
    logger.info(f"Frontend path: {frontend_path}")
    logger.info(f"Frontend exists: {frontend_path.exists()}")
    if frontend_path.exists():
        logger.info(f"Frontend contents: {list(frontend_path.iterdir())}")
    
    logger.info("="*60)
    logger.info("[SUCCESS] RETAIL INTELLIGENCE PLATFORM READY!")
    logger.info("="*60)
    logger.info(f"Dashboard: http://localhost:{settings.PORT}")
    logger.info(f"API Documentation: http://localhost:{settings.PORT}/docs")
    logger.info(f"Health Check: http://localhost:{settings.PORT}/health")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("Shutting down Retail Intelligence Platform...")
    logger.info("Cleanup completed successfully")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=1 if settings.DEBUG else settings.MAX_WORKERS,
        log_level="info"
    )