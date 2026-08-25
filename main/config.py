"""
config.py — typed, startup-validated configuration.

Two rules shape this file:
  * Production NEVER silently degrades. Anonymous identity and the local
    in-process execution runner are development conveniences; in production
    they must be refused loudly at startup, not discovered later by a student.
  * Validation happens ONCE at import/startup, not at first use. A missing
    secret should stop the process, not surface as a 500 mid-lesson.

No provider call, database connection, or container launch happens here.
"""
import os
from dataclasses import dataclass, field
from typing import Literal

Environment = Literal["development", "staging", "production"]
ExecutionBackend = Literal["local_process", "container"]


class ConfigError(RuntimeError):
    """Startup configuration is invalid. The process must not continue."""


def _bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got {raw!r}")


def _csv(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return list(default)
    return [p.strip() for p in raw.split(",") if p.strip()]


@dataclass(frozen=True)
class DatabaseConfig:
    url: str = ""                     # postgres DSN; empty => sqlite dev adapter
    sqlite_path: str = "data/grading_sessions.sqlite3"
    pool_min: int = 1
    pool_max: int = 10
    statement_timeout_ms: int = 15_000

    @property
    def backend(self) -> str:
        return "postgres" if self.url else "sqlite"


@dataclass(frozen=True)
class AuthConfig:
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""   # server-only; never sent to a browser
    jwt_audience: str = "authenticated"
    jwt_issuer: str = ""
    # Role source: a claim on the Supabase JWT (app_metadata.role by default),
    # matching how this project already stores per-user metadata.
    role_claim_path: str = "app_metadata.role"
    instructor_roles: tuple[str, ...] = ("instructor", "admin")
    allow_anonymous: bool = False          # DEV ONLY — refused in production


@dataclass(frozen=True)
class LLMConfig:
    adapter_model: str = "gpt-4o-mini"
    primary_judge_model: str = "gpt-4o-mini"
    verifier_judge_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    request_timeout_s: float = 30.0
    max_retries: int = 2
    circuit_break_after: int = 5           # consecutive failures before opening
    circuit_reset_s: float = 60.0
    api_key: str = ""


@dataclass(frozen=True)
class ExecutionConfig:
    backend: ExecutionBackend = "local_process"
    container_image: str = "microtutor-runner:latest"
    container_runtime: str = "docker"
    wall_timeout_s: float = 6.0
    cpu_seconds: int = 5
    memory_bytes: int = 512 * 1024 * 1024
    max_processes: int = 0
    max_file_bytes: int = 1024 * 1024
    control_message_bytes: int = 4 * 1024 * 1024
    # Escape hatch so a deliberate operator choice is auditable, not accidental.
    allow_unsafe_local_in_production: bool = False


@dataclass(frozen=True)
class LimitsConfig:
    max_request_bytes: int = 256 * 1024
    max_student_code_bytes: int = 32 * 1024
    session_ttl_hours: int = 12
    max_active_sessions_per_user: int = 5
    submissions_per_minute: int = 12
    submissions_per_session_per_minute: int = 6
    max_attempts_per_chunk: int = 2        # second failure reveals


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    json: bool = True
    redact_student_code: bool = True


@dataclass(frozen=True)
class Config:
    environment: Environment = "development"
    allowed_origins: list[str] = field(default_factory=lambda: ["*"])
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


def load(env: dict | None = None) -> Config:
    """Build the config from the environment. Pure — no I/O beyond os.environ."""
    e = env if env is not None else os.environ
    prev = None
    if env is not None:                    # allow injection for callers/tests
        prev, os.environ = os.environ, {**os.environ, **env}
    try:
        environment = (os.environ.get("MICROTUTOR_ENV") or "development").strip().lower()
        if environment not in ("development", "staging", "production"):
            raise ConfigError(
                f"MICROTUTOR_ENV must be development|staging|production, "
                f"got {environment!r}")
        cfg = Config(
            environment=environment,                       # type: ignore[arg-type]
            allowed_origins=_csv("MICROTUTOR_ALLOWED_ORIGINS", ["*"]),
            database=DatabaseConfig(
                url=os.environ.get("DATABASE_URL", ""),
                sqlite_path=os.environ.get("MICROTUTOR_SESSION_DB",
                                           "data/grading_sessions.sqlite3"),
                pool_min=_int("DB_POOL_MIN", 1), pool_max=_int("DB_POOL_MAX", 10),
                statement_timeout_ms=_int("DB_STATEMENT_TIMEOUT_MS", 15_000)),
            auth=AuthConfig(
                supabase_url=os.environ.get("SUPABASE_URL", ""),
                supabase_anon_key=os.environ.get("SUPABASE_KEY", ""),
                supabase_service_role_key=os.environ.get(
                    "SUPABASE_SERVICE_ROLE_KEY", ""),
                jwt_audience=os.environ.get("SUPABASE_JWT_AUDIENCE", "authenticated"),
                jwt_issuer=os.environ.get("SUPABASE_JWT_ISSUER", ""),
                role_claim_path=os.environ.get("MICROTUTOR_ROLE_CLAIM",
                                               "app_metadata.role"),
                allow_anonymous=_bool("MICROTUTOR_ALLOW_ANONYMOUS", False)),
            llm=LLMConfig(
                adapter_model=os.environ.get("MICROTUTOR_ADAPTER_MODEL", "gpt-4o-mini"),
                primary_judge_model=os.environ.get("MICROTUTOR_JUDGE_MODEL",
                                                   "gpt-4o-mini"),
                verifier_judge_model=os.environ.get("MICROTUTOR_VERIFIER_MODEL",
                                                    "gpt-4o-mini"),
                request_timeout_s=float(os.environ.get("MICROTUTOR_LLM_TIMEOUT", 30)),
                max_retries=_int("MICROTUTOR_LLM_RETRIES", 2),
                circuit_break_after=_int("MICROTUTOR_LLM_CIRCUIT_BREAK", 5),
                api_key=os.environ.get("OPENAI_API_KEY", "")),
            execution=ExecutionConfig(
                backend=(os.environ.get("MICROTUTOR_EXECUTION_BACKEND")
                         or "local_process"),                # type: ignore[arg-type]
                container_image=os.environ.get("MICROTUTOR_RUNNER_IMAGE",
                                               "microtutor-runner:latest"),
                container_runtime=os.environ.get("MICROTUTOR_CONTAINER_RUNTIME",
                                                 "docker"),
                allow_unsafe_local_in_production=_bool(
                    "MICROTUTOR_ALLOW_UNSAFE_LOCAL_EXECUTION", False)),
            limits=LimitsConfig(
                max_request_bytes=_int("MICROTUTOR_MAX_REQUEST_BYTES", 256 * 1024),
                max_student_code_bytes=_int("MICROTUTOR_MAX_CODE_BYTES", 32 * 1024),
                session_ttl_hours=_int("MICROTUTOR_SESSION_TTL_HOURS", 12),
                max_active_sessions_per_user=_int("MICROTUTOR_MAX_SESSIONS", 5),
                submissions_per_minute=_int("MICROTUTOR_SUBMIT_RPM", 12)),
            logging=LoggingConfig(
                level=os.environ.get("MICROTUTOR_LOG_LEVEL", "INFO").upper(),
                json=_bool("MICROTUTOR_LOG_JSON", True),
                redact_student_code=_bool("MICROTUTOR_REDACT_CODE", True)))
    finally:
        if prev is not None:
            os.environ = prev
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    """Fail loudly and specifically. Production must never degrade silently."""
    problems: list[str] = []

    if cfg.execution.backend not in ("local_process", "container"):
        problems.append(
            f"MICROTUTOR_EXECUTION_BACKEND must be local_process|container, "
            f"got {cfg.execution.backend!r}")

    if cfg.is_production:
        if cfg.auth.allow_anonymous:
            problems.append(
                "MICROTUTOR_ALLOW_ANONYMOUS is forbidden in production: student "
                "identity must be an authenticated JWT subject")
        if not cfg.auth.supabase_url or not cfg.auth.supabase_service_role_key:
            problems.append(
                "production requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY "
                "for server-side token validation")
        if not cfg.database.url:
            problems.append(
                "production requires DATABASE_URL; the SQLite adapter is a "
                "local-development store only")
        if cfg.execution.backend == "local_process" and \
                not cfg.execution.allow_unsafe_local_in_production:
            problems.append(
                "production refuses the local in-process runner: set "
                "MICROTUTOR_EXECUTION_BACKEND=container (or, deliberately and "
                "auditably, MICROTUTOR_ALLOW_UNSAFE_LOCAL_EXECUTION=true)")
        if "*" in cfg.allowed_origins:
            problems.append(
                "production forbids a wildcard CORS origin; set "
                "MICROTUTOR_ALLOWED_ORIGINS to an explicit allowlist")
        if not cfg.llm.api_key:
            problems.append("production requires OPENAI_API_KEY for Tier 3/4")

    if cfg.llm.temperature != 0.0:
        problems.append("LLM temperature must be 0.0 for reproducible judging")
    if cfg.limits.max_student_code_bytes > cfg.limits.max_request_bytes:
        problems.append("max_student_code_bytes cannot exceed max_request_bytes")

    if problems:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(problems))


_CACHED: Config | None = None


def get() -> Config:
    """Process-wide config, validated once."""
    global _CACHED
    if _CACHED is None:
        _CACHED = load()
    return _CACHED
