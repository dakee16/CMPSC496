"""A private student read endpoint, separate from the grading pipeline."""
import logging
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from main.student_progress import progress_snapshot


def student_progress_router(get_client, require_student):
    router = APIRouter()

    @router.get("/student/progress")
    def read_progress(request: Request, timezone: str = Query(default="UTC", max_length=100)):
        claims = require_student(request)
        try:
            ZoneInfo(timezone)
        except (ZoneInfoNotFoundError, ValueError):
            raise HTTPException(status_code=400, detail={
                "message": "Choose a valid time zone."}) from None
        try:
            data = progress_snapshot(get_client(), claims["sub"], timezone)
        except Exception:
            logging.getLogger(__name__).exception("Student progress could not be loaded")
            raise HTTPException(status_code=503, detail={
                "message": "Your progress is temporarily unavailable. Please try again."}) from None
        return JSONResponse(data, headers={"Cache-Control": "private, no-store"})

    return router
