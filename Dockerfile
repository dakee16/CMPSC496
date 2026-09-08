# MicroTutor - one uvicorn process serving both the API and frontend/.
#
# Portable across Render / Railway / Fly: all three build a Dockerfile and
# inject $PORT. Nothing here is host-specific.
#
# ONE WORKER, deliberately - see start.sh.
FROM python:3.13-slim

# Student code runs as a subprocess of this process (main/execution.py). The
# AST policy and rlimits are the first fence; not being root is the second.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# Dependencies first so an edit to the app does not reinstall them.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x start.sh && chown -R appuser:appuser /app

# NO `USER appuser` line here, deliberately. Hosts mount their volume owned by
# root, so a container that has already dropped privileges cannot write to it
# and never boots. start.sh keeps root just long enough to chown the volume,
# then steps down to appuser before uvicorn starts.

# Only for local `docker run -p 8000:8000`; hosts override it with their own.
ENV PORT=8000
EXPOSE 8000

# The host's health check should hit /health (frontend/api_server.py:244).
CMD ["./start.sh"]
