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


def retry_reads(client):
    """`client` (a Supabase client) with one retry on its database reads.
    Never raises: a client that cannot be wrapped is returned as it was."""
    try:
        session = client.postgrest.session
        if not isinstance(session._transport, RetryReads):
            session._transport = RetryReads(session._transport)
    except Exception as e:
        print(f"  ⚠️  database read retry not installed: {e!r}"[:160])
    return client
