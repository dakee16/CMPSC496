"""
psu_auth.py - PSU-only sign-in via Microsoft Entra ID (Azure AD).

STATUS: complete but DORMANT. Nothing imports this yet. It goes live by doing
three things, in this order:

  1. Register the app in the PSU Azure portal and put four values in .env:
         AZURE_TENANT_ID=<the PSU tenant GUID>
         AZURE_CLIENT_ID=<application (client) id>
         AZURE_CLIENT_SECRET=<client secret value>
         AZURE_REDIRECT_URI=https://<host>/auth/callback
     plus a session signing key, which is ours and not Microsoft's:
         MICROTUTOR_SESSION_SECRET=<64 random hex chars>
     The redirect URI must match the portal registration EXACTLY, including
     scheme, port and trailing slash - a mismatch is the single most common
     first-day failure and Microsoft reports it as AADSTS50011.
  2. pip install "PyJWT[crypto]" (already listed, commented, in requirements).
  3. Uncomment the auth block in frontend/api_server.py and the button in
     frontend/login.html.

WHY THIS SHAPE

Authorization Code flow with PKCE, confidential client. The browser never sees
a token: it gets a signed session cookie, and the id_token is exchanged and
verified server-side. That means a stolen cookie is scoped to this app and
expires on our schedule, rather than being a Microsoft credential.

FOUR THINGS ENFORCE "PSU ONLY", and all four are needed:

  * the authorize URL is built against the PSU TENANT, not /common, so the
    Microsoft account picker will not offer a personal account at all;
  * the `tid` claim must equal AZURE_TENANT_ID - a token minted by any other
    tenant is rejected even if it is perfectly valid;
  * the verified email claim must end in an allowed PSU domain;
  * `state` and `nonce` are checked, so a token from another login attempt
    cannot be replayed into this one.

The domain check alone would not be enough: `email` is not guaranteed unique or
verified in every tenant, which is why the tenant check carries the real weight
and the domain check narrows within it.
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

TENANT_ID = os.environ.get("AZURE_TENANT_ID", "").strip()
CLIENT_ID = os.environ.get("AZURE_CLIENT_ID", "").strip()
CLIENT_SECRET = os.environ.get("AZURE_CLIENT_SECRET", "").strip()
REDIRECT_URI = os.environ.get("AZURE_REDIRECT_URI",
                              "http://localhost:8000/auth/callback").strip()

# Students, not the directory: openid/profile/email is all we need, and asking
# for more would make the consent screen scarier than the app deserves.
SCOPES = "openid profile email"

# Subdomains are real at Penn State (psu.edu proper plus campus/college
# subdomains), so membership is tested by suffix, not equality.
ALLOWED_DOMAINS = tuple(
    d.strip().lower()
    for d in os.environ.get("MICROTUTOR_ALLOWED_DOMAINS", "psu.edu").split(",")
    if d.strip())

SESSION_SECRET = os.environ.get("MICROTUTOR_SESSION_SECRET", "").strip()
SESSION_COOKIE = "microtutor_session"
SESSION_HOURS = 12          # a class day; long enough to not re-login mid-lab
CLOCK_SKEW = 120            # seconds of tolerance on exp/nbf

_AUTHORITY = "https://login.microsoftonline.com"


class AuthError(Exception):
    """Sign-in failed. The message is safe to show a student; anything
    diagnostic goes in `detail` and stays server-side.

    Kept separate from HTTPException so this module has no FastAPI dependency
    and can be unit-tested without an app."""

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.detail = detail


def is_configured() -> bool:
    """True once the Azure credentials exist. The routes check this so a
    half-configured deployment fails with a clear message at the door instead
    of a confusing redirect loop three steps later."""
    return bool(TENANT_ID and CLIENT_ID and CLIENT_SECRET and SESSION_SECRET)


# ── PKCE ─────────────────────────────────────────────────────────────────

def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def new_pkce() -> tuple[str, str]:
    """(verifier, challenge). The verifier stays in the user's temporary cookie
    and never crosses to Microsoft; the challenge does. An attacker who
    intercepts the authorization code therefore cannot redeem it."""
    verifier = _b64url(secrets.token_bytes(64))
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
    return verifier, challenge


# ── step 1: send them to Microsoft ───────────────────────────────────────

def authorize_url(state: str, nonce: str, challenge: str,
                  login_hint: str | None = None) -> str:
    """The URL to redirect the browser to.

    Built against the PSU tenant, so the account picker never offers a personal
    Microsoft account. `prompt=select_account` is deliberate: on a shared lab
    machine the previous student's session must not be silently reused."""
    from urllib.parse import urlencode
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "response_mode": "query",
        "scope": SCOPES,
        "state": state,
        "nonce": nonce,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "prompt": "select_account",
    }
    if login_hint:
        params["login_hint"] = login_hint
    return f"{_AUTHORITY}/{TENANT_ID}/oauth2/v2.0/authorize?{urlencode(params)}"


# ── step 2: trade the code for tokens ────────────────────────────────────

def exchange_code(code: str, verifier: str) -> Dict[str, Any]:
    """Redeem the authorization code at the token endpoint.

    Server-to-server over TLS with the client secret, so this response is
    already authenticated as coming from Microsoft. The id_token is still
    verified afterwards - defence in depth is cheap here and the failure mode
    of skipping it is a forged identity."""
    r = requests.post(
        f"{_AUTHORITY}/{TENANT_ID}/oauth2/v2.0/token",
        data={"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET,
              "grant_type": "authorization_code", "code": code,
              "redirect_uri": REDIRECT_URI, "code_verifier": verifier,
              "scope": SCOPES},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30)
    if r.status_code != 200:
        # Microsoft's error body names the real cause (AADSTS…); it is useful
        # in a log and meaningless-to-alarming in a browser.
        raise AuthError("Sign-in failed. Please try again.",
                        detail=f"token endpoint {r.status_code}: {r.text[:300]}")
    return r.json()


# ── step 3: verify the identity token ────────────────────────────────────

def validate_id_token(id_token: str, nonce: str) -> Dict[str, Any]:
    """Verify signature, issuer, audience, nonce and tenant; return the claims.

    PyJWT is imported here rather than at module scope so this file stays
    importable before the dependency is installed - which is exactly the state
    the repository is in until auth goes live. PyJWKClient does its own key
    fetching and caching, so there is no JWKS cache to get wrong here."""
    try:
        import jwt
        from jwt import PyJWKClient
    except ImportError as e:                       # pragma: no cover
        raise AuthError("Sign-in is not configured on this server.",
                        detail=f'install PyJWT[crypto]: {e}') from e

    try:
        signing_key = PyJWKClient(
            f"{_AUTHORITY}/{TENANT_ID}/discovery/v2.0/keys"
        ).get_signing_key_from_jwt(id_token)
        claims = jwt.decode(
            id_token, signing_key.key, algorithms=["RS256"],
            audience=CLIENT_ID,
            issuer=f"{_AUTHORITY}/{TENANT_ID}/v2.0",
            leeway=CLOCK_SKEW,
            options={"require": ["exp", "iat", "aud", "iss"]})
    except Exception as e:
        raise AuthError("Sign-in failed. Please try again.",
                        detail=f"id_token rejected: {type(e).__name__}: {e}"
                        ) from e

    # jwt.decode does not check nonce - it is an OIDC concern, not a JWT one.
    # Without this a token captured from another login can be replayed here.
    if not nonce or claims.get("nonce") != nonce:
        raise AuthError("Sign-in expired. Please start again.",
                        detail="nonce mismatch")

    if claims.get("tid") != TENANT_ID:
        raise AuthError("Please sign in with your Penn State account.",
                        detail=f"wrong tenant: {claims.get('tid')}")
    return claims


def psu_email(claims: Dict[str, Any]) -> str:
    """The verified PSU address, or AuthError.

    Claim order matters: `email` is the OIDC standard one but is not always
    present; `preferred_username` is the UPN in a work/school tenant and is what
    PSU actually populates. `upn` is the v1 fallback."""
    raw = (claims.get("email") or claims.get("preferred_username")
           or claims.get("upn") or "").strip().lower()
    if "@" not in raw:
        raise AuthError("Your account did not provide an email address.",
                        detail=f"no usable email claim: {sorted(claims)}")
    domain = raw.rsplit("@", 1)[1]
    if not any(domain == d or domain.endswith("." + d) for d in ALLOWED_DOMAINS):
        raise AuthError("MicroTutor is only open to Penn State accounts "
                        f"({', '.join('@' + d for d in ALLOWED_DOMAINS)}).",
                        detail=f"rejected domain: {domain}")
    return raw


def looks_like_psu_email(address: str) -> bool:
    """Cheap check for the email box on the login page.

    This is a CONVENIENCE, never a control: it stops someone typing a gmail
    address and waiting through a redirect to be told no. The real decision is
    psu_email() on a verified claim, and this function must never be used to
    admit anyone."""
    address = (address or "").strip().lower()
    if address.count("@") != 1:
        return False
    domain = address.rsplit("@", 1)[1]
    return any(domain == d or domain.endswith("." + d) for d in ALLOWED_DOMAINS)


# ── our own session cookie ───────────────────────────────────────────────
# Signed with SESSION_SECRET, not encrypted: the payload is the student's own
# email and name, which they already know. Signing is what matters - it stops
# them editing the cookie to become someone else. HMAC-SHA256 over stdlib, so
# no new dependency for the one thing we can least afford to get wrong.

def issue_session(claims: Dict[str, Any]) -> str:
    email = psu_email(claims)
    payload = {"sub": claims.get("oid") or claims.get("sub") or email,
               "email": email,
               "name": claims.get("name") or email.split("@")[0],
               "exp": int(time.time()) + SESSION_HOURS * 3600}
    body = _b64url(json.dumps(payload, separators=(",", ":")).encode())
    sig = _b64url(hmac.new(SESSION_SECRET.encode(), body.encode(),
                           hashlib.sha256).digest())
    return f"{body}.{sig}"


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
    """Flags for set_cookie. `secure=False` is ONLY for local http development;
    leaving it false in production would let the session ride over plain http.

    SameSite=lax rather than strict because the student ARRIVES on a redirect
    back from Microsoft, and strict would drop the cookie on that first hop."""
    return {"httponly": True, "secure": secure, "samesite": "lax",
            "max_age": SESSION_HOURS * 3600, "path": "/"}


if __name__ == "__main__":
    # The parts that must hold with no Azure tenant and no network: the domain
    # rule, the cookie signature, and PKCE. Anything touching Microsoft is not
    # exercised here and cannot be until credentials exist.
    import main.psu_auth as m

    m.SESSION_SECRET = "test-secret-not-a-real-key"
    m.ALLOWED_DOMAINS = ("psu.edu",)

    assert m.looks_like_psu_email("abc123@psu.edu")
    assert m.looks_like_psu_email("s@engr.psu.edu"), "subdomains are real"
    assert not m.looks_like_psu_email("someone@gmail.com")
    assert not m.looks_like_psu_email("psu.edu")
    assert not m.looks_like_psu_email("a@b@psu.edu")
    # The checks that actually matter: lookalike domains must not pass.
    assert not m.looks_like_psu_email("attacker@notpsu.edu"), "suffix confusion"
    assert not m.looks_like_psu_email("attacker@psu.edu.evil.com")

    tok = m.issue_session({"oid": "abc", "name": "A Student",
                           "email": "abc123@psu.edu"})
    assert m.read_session(tok)["email"] == "abc123@psu.edu"
    body, sig = tok.split(".")
    assert m.read_session(f"{body}x.{sig}") is None, "tampered body accepted"
    assert m.read_session(f"{body}.{sig[:-1]}A") is None, "bad signature accepted"
    assert m.read_session("") is None and m.read_session("junk") is None

    m.SESSION_HOURS = -1
    assert m.read_session(
        m.issue_session({"oid": "a", "email": "a@psu.edu"})) is None, \
        "expired cookie accepted"
    m.SESSION_HOURS = 12

    try:
        m.psu_email({"email": "outsider@gmail.com"})
        raise AssertionError("non-PSU email was admitted")
    except m.AuthError:
        pass

    v, c = m.new_pkce()
    assert c == m._b64url(hashlib.sha256(v.encode()).digest())
    assert v != c and len(v) > 40
    print("psu_auth self-check ok (Azure paths untested - no credentials yet)")
