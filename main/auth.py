"""
auth.py - username/password sign-in, backed by Supabase.

WHY THIS AND NOT MICROSOFT (for now)

Azure/Entra credentials have to come from PSU IT and that is not on our
schedule, so main/psu_auth.py stays on hold. Until it lands, identity is a
row in `students` (username + bcrypt hash) and the network boundary is the
college VPN: only people on the campus network can reach the site at all.

That split matters, and each half does a different job:

  * the VPN answers "may this machine reach us" - it keeps the public
    internet out, and it is what makes "students and faculty only" true;
  * this module answers "who is this person, and are they faculty" - the VPN
    cannot tell one student from another, and every student is on it.

So the VPN is not a substitute for the cookie below, and the cookie is not a
substitute for the VPN.

WHAT IS TRUSTED

The signed cookie, and nothing else. The browser may still send a name or a
role in a request body - it is decoration for the UI and the server ignores
it. `role` in particular is read from the database row, never from the
client, so a student cannot become an instructor by editing sessionStorage.

MIGRATING TO SSO LATER

Usernames are PSU email addresses, which is the same string Entra hands back
as `preferred_username`. When Azure credentials arrive, matching an SSO login
to an existing account is a lookup on this column - not a data migration.
"""
import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any, Dict, Optional

import bcrypt
from dotenv import load_dotenv

load_dotenv()

# Subdomains are real at Penn State (campus and college subdomains), so
# membership is tested by suffix, not equality.
ALLOWED_DOMAINS = tuple(
    d.strip().lower()
    for d in os.environ.get("MICROTUTOR_ALLOWED_DOMAINS", "psu.edu").split(",")
    if d.strip())

SESSION_SECRET = os.environ.get("MICROTUTOR_SESSION_SECRET", "").strip()
SESSION_COOKIE = "microtutor_session"
SESSION_HOURS = 12          # a class day; long enough to not re-login mid-lab

# bcrypt silently ignores everything past 72 BYTES. A password rejected for
# being too long is an annoyance; one where the last N characters turn out not
# to matter is a security bug nobody can see.
MIN_PASSWORD = 8
MAX_PASSWORD_BYTES = 72


class AuthError(Exception):
    """Sign-in or registration failed. The message is safe to show a student;
    anything diagnostic goes in `detail` and stays server-side.

    Deliberately not an HTTPException, so this module has no FastAPI
    dependency and can be exercised without an app."""

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.detail = detail


def is_configured() -> bool:
    """True once a session signing key exists. Without it every cookie would
    verify against an empty secret, so the routes refuse to serve at all
    rather than issue sessions anyone can forge."""
    return bool(SESSION_SECRET)


# ── usernames ────────────────────────────────────────────────────────────

def normalize_username(username: str) -> str:
    """Lowercase and strip. Case is not identity: `A1@psu.edu` and
    `a1@psu.edu` are the same person and must not become two accounts. This
    runs on BOTH registration and login, which is what lets the lookup stay a
    plain equality test instead of a case-insensitive pattern match."""
    return (username or "").strip().lower()


def valid_username(username: str) -> bool:
    """A PSU address, checked at registration only.

    Not a security control - the VPN and the password are. It is here so the
    username column holds the one string that will match an Entra
    `preferred_username` claim when SSO arrives."""
    username = normalize_username(username)
    if username.count("@") != 1:
        return False
    local, domain = username.split("@")
    if not local:
        return False
    return any(domain == d or domain.endswith("." + d) for d in ALLOWED_DOMAINS)


# ── passwords ────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    if len(password) < MIN_PASSWORD:
        raise AuthError(f"Password must be at least {MIN_PASSWORD} characters.")
    if len(password.encode()) > MAX_PASSWORD_BYTES:
        raise AuthError("Password is too long (72 bytes maximum).")
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, stored_hash: str) -> bool:
    """False for anything that is not a match, including a row whose hash is
    the `!demo-no-login` placeholder the old role-picker wrote - bcrypt raises
    on a malformed hash, and a raise here would be a 500 on a wrong password."""
    try:
        return bcrypt.checkpw(password.encode(), (stored_hash or "").encode())
    except (ValueError, TypeError):
        return False


# ── session cookie ───────────────────────────────────────────────────────
# Signed with SESSION_SECRET, not encrypted: the payload is the student's own
# id, name and role, which they already know. Signing is what matters - it
# stops them editing the cookie to become someone else, or an instructor.
# HMAC-SHA256 over stdlib, so no new dependency for the one thing we can least
# afford to get wrong.

def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def sign_session(payload: Dict[str, Any]) -> str:
    """Sign an arbitrary claims dict. `exp` is set here rather than by the
    caller so no caller can issue a session that outlives the policy."""
    body = _b64url(json.dumps({**payload,
                               "exp": int(time.time()) + SESSION_HOURS * 3600},
                              separators=(",", ":")).encode())
    sig = _b64url(hmac.new(SESSION_SECRET.encode(), body.encode(),
                           hashlib.sha256).digest())
    return f"{body}.{sig}"


def issue_session(student: Dict[str, Any]) -> str:
    """Cookie value for a `students` row.

    `sub` is the row id, not the username: an address can be reassigned when a
    student changes their name, and re-pointing years of saved work at the
    wrong person is not a recoverable mistake."""
    username = student["username"]
    return sign_session({"sub": student["id"],
                         "username": username,
                         "name": student.get("name") or username.split("@")[0],
                         "role": student.get("role") or "student"})


def read_session(token: str) -> Optional[Dict[str, Any]]:
    """Claims from a valid unexpired cookie, else None.

    Returns None for every failure rather than raising or distinguishing them:
    a tampered cookie and an expired one both mean "sign in again", and telling
    them apart only helps someone probing the signature."""
    if not token or not SESSION_SECRET or token.count(".") != 1:
        return None
    body, sig = token.split(".", 1)
    expected = _b64url(hmac.new(SESSION_SECRET.encode(), body.encode(),
                                hashlib.sha256).digest())
    if not hmac.compare_digest(sig, expected):    # constant-time, not ==
        return None
    try:
        payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("exp", 0) < time.time():
        return None
    return payload


def cookie_kwargs(secure: bool = True) -> Dict[str, Any]:
    """Flags for set_cookie. `secure=False` is ONLY for local http
    development; leaving it false in production would let the session ride
    over plain http - and "it is behind the VPN" does not fix that, because
    the VPN carries other students too."""
    return {"httponly": True, "secure": secure, "samesite": "lax",
            "max_age": SESSION_HOURS * 3600, "path": "/"}


# ── the two things the routes actually call ──────────────────────────────

def register_student(sb, username: str, password: str) -> Dict[str, Any]:
    """Create an account and return the row. Raises AuthError on any refusal.

    Role is NOT a parameter. Every account is created as a student and an
    instructor is promoted by hand in SQL; a self-service role field would let
    anyone on the VPN grant themselves the assignment-upload screen."""
    username = normalize_username(username)
    if not valid_username(username):
        raise AuthError("Use your Penn State email address "
                        f"({', '.join('@' + d for d in ALLOWED_DOMAINS)}).",
                        detail=f"rejected username: {username!r}")
    pw_hash = hash_password(password)             # raises on a weak password

    try:
        rows = sb.table("students").insert(
            {"username": username, "password_hash": pw_hash,
             "role": "student"}).execute().data
    except Exception as e:
        # Only a UNIQUE violation is "already exists". Reporting every failed
        # insert that way is how a missing `role` column - the state this is in
        # until migrations/003 runs - told a brand new student that they
        # already had an account, with the real cause only in the server log.
        text = str(e).lower()
        if "duplicate" in text or "23505" in text or "unique" in text:
            raise AuthError("That account already exists. Sign in instead.",
                            detail=f"duplicate username: {username}") from e
        raise AuthError("Could not create the account. Try again, or ask "
                        "your instructor if it keeps failing.",
                        detail=f"insert failed: {e}") from e
    if not rows:
        raise AuthError("Could not create the account. Try again.",
                        detail="insert returned no row")
    return rows[0]


def authenticate(sb, username: str, password: str) -> Dict[str, Any]:
    """The row for a correct username+password, else AuthError.

    One message for both "no such account" and "wrong password", on purpose:
    distinguishing them tells an outsider which addresses are registered."""
    username = normalize_username(username)
    wrong = AuthError("Incorrect username or password.")

    rows = sb.table("students").select("*").eq(
        "username", username).limit(1).execute().data
    if not rows:
        # Spend the time anyway. Returning instantly for an unknown username
        # and slowly for a known one leaks the account list by stopwatch.
        bcrypt.checkpw(b"x", bcrypt.hashpw(b"x", bcrypt.gensalt()))
        raise wrong
    row = rows[0]
    if not verify_password(password, row.get("password_hash", "")):
        raise wrong
    return row


if __name__ == "__main__":
    # The parts that must hold with no database and no network: the username
    # rule, password limits, and the cookie signature.
    import main.auth as m

    m.SESSION_SECRET = "test-secret-not-a-real-key"
    m.ALLOWED_DOMAINS = ("psu.edu",)

    assert m.normalize_username("  ABC123@PSU.edu ") == "abc123@psu.edu"
    assert m.valid_username("abc123@psu.edu")
    assert m.valid_username("s@engr.psu.edu"), "subdomains are real"
    assert not m.valid_username("someone@gmail.com")
    assert not m.valid_username("@psu.edu"), "empty local part"
    assert not m.valid_username("a@b@psu.edu")
    # The checks that actually matter: lookalike domains must not pass.
    assert not m.valid_username("attacker@notpsu.edu"), "suffix confusion"
    assert not m.valid_username("attacker@psu.edu.evil.com")

    h = m.hash_password("correct horse battery")
    assert m.verify_password("correct horse battery", h)
    assert not m.verify_password("wrong", h)
    assert not m.verify_password("x", "!demo-no-login"), "malformed hash raised"
    for bad in ("short", "x" * 73):
        try:
            m.hash_password(bad)
            raise AssertionError(f"accepted a {len(bad)}-character password")
        except m.AuthError:
            pass

    tok = m.issue_session({"id": "u-1", "username": "abc123@psu.edu",
                           "role": "teacher"})
    claims = m.read_session(tok)
    assert claims["sub"] == "u-1" and claims["role"] == "teacher"
    assert claims["name"] == "abc123", "name defaults to the local part"
    body, sig = tok.split(".")
    assert m.read_session(f"{body}x.{sig}") is None, "tampered body accepted"
    assert m.read_session(f"{body}.{sig[:-1]}A") is None, "bad signature accepted"
    assert m.read_session("") is None and m.read_session("junk") is None

    m.SESSION_HOURS = -1
    assert m.read_session(m.issue_session(
        {"id": "u-1", "username": "a@psu.edu"})) is None, "expired cookie accepted"
    m.SESSION_HOURS = 12

    # An empty secret must not verify anything, or an unconfigured deployment
    # would accept cookies signed with "".
    forged = m.sign_session({"sub": "u-2"})
    m.SESSION_SECRET = ""
    assert not m.is_configured()
    assert m.read_session(forged) is None, "unconfigured server accepted a cookie"

    print("auth self-check ok")
