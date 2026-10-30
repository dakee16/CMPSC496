"""
db_retry.py - one retry for a database READ that a network blip cut off.

WHY (2 Oct, live). At 17:13 two pages - /solved and /assignments - answered
500 at the same instant: "httpcore.ReadError: [Errno 11] Resource temporarily
unavailable", a dropped connection to Supabase mid-read. The other 9 loads of
those pages that day were fine. A page should not fail on a blip the next
attempt would not see.

WHAT IS RETRIED: a GET - PostgREST's select - that failed with a connection-
level error, once. A read changes nothing, so a second try is always safe.
WHAT IS NOT: anything else. An insert, update or delete that broke mid-flight
may already have happened; repeating it could write twice, so it fails as it
always did. A slow query (timeout) is not retried either - that would double
the wait instead of curing it.

Wraps only the database client's transport: timeouts, headers and everything
else about the connection stay exactly as Supabase set them.
"""
import httpx

_BLIPS = (httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError,
          httpx.ConnectError)


class RetryReads(httpx.BaseTransport):
    def __init__(self, inner: httpx.BaseTransport):
        self.inner = inner

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        try:
            return self.inner.handle_request(request)
        except _BLIPS:
            if request.method != "GET":
                raise
            return self.inner.handle_request(request)

    def close(self) -> None:
        self.inner.close()


def _http1(transport: httpx.BaseTransport) -> httpx.BaseTransport:
    """The same connection settings over HTTP/1.1.

    HTTP/2 PUTS EVERY REQUEST ON ONE CONNECTION, and this client is shared by
    every thread serving every student. Under the 15-student load test (30 Oct)
    the frames tangled: saves were lost ("ReadError: [Errno 11]" - the 2 Oct
    blip above - and "deque mutated during iteration"), the approval check
    failed closed on three approved plans, and two tutor turns answered 500.
    On the live box, 20 threads mixing reads and saves: HTTP/2 lost 129 of
    7,200 requests over four rounds, HTTP/1.1 none, at the same speed. HTTP/1.1
    gives each request in flight its own connection from a thread-safe pool.
    The certificate check (the SSL context) is carried over unchanged."""
    pool = getattr(transport, "_pool", None)
    if not isinstance(transport, httpx.HTTPTransport) or pool is None or not pool._http2:
        return transport
    return httpx.HTTPTransport(verify=pool._ssl_context, http2=False)


def retry_reads(client):
    """`client` (a Supabase client) over HTTP/1.1 (_http1), with one retry on
    its database reads. Never raises: a client that cannot be changed is
    returned as it was."""
    try:
        session = client.postgrest.session
        if not isinstance(session._transport, RetryReads):
            session._transport = RetryReads(_http1(session._transport))
    except Exception as e:
        print(f"  ⚠️  database read retry not installed: {e!r}"[:160])
    return client
