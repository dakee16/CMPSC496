"""runner_service.py - student code execution as its own service.

Run this where it can be scaled apart from the web tier and where a student's
program can reach nothing worth reaching: no application secrets in the
environment and no route to Supabase or OpenAI (docker-compose.scale.yml).

    uvicorn main.runner_service:app --host 0.0.0.0 --port 8100

The web tier calls it when MICROTUTOR_RUNNER_URL is set (main/execution.py).
Bodies are pyvalue literals, the same lossless format used to reach the child.
"""
import os
import secrets

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse

from .execution import run_student_code
from .pyvalue import dumps, loads

MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_TIMEOUT = 30.0

app = FastAPI(title="MicroTutor runner", docs_url=None, redoc_url=None,
              openapi_url=None)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run")
async def run(request: Request):
    expected = os.environ.get("MICROTUTOR_RUNNER_TOKEN", "")
    if not expected:
        raise HTTPException(503, "runner token is not configured")
    given = request.headers.get("x-runner-token", "")
    if not secrets.compare_digest(given.encode(), expected.encode()):
        raise HTTPException(401, "bad runner token")
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        raise HTTPException(413, "request too large")
    try:
        job = loads(body.decode())
        args = (job["code"], job["inputs"], job.get("entry_name"),
                min(float(job.get("timeout", 6.0)), MAX_TIMEOUT), job.get("tests"))
    except Exception:
        raise HTTPException(400, "malformed job") from None
    status, results, error = await run_in_threadpool(run_student_code, *args)
    return PlainTextResponse(dumps([status, results, error]))
