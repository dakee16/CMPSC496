"""
prepare_bus.py - watching the preparation run that is ALREADY happening.

A teacher uploads a file and waits minutes while every problem is prepared. The
upload page shows one row per problem and a spinner, which answers "is it still
going" and nothing else. What they actually want, when a problem is taking too
long or comes back "needs work", is to open that one problem and watch the
pipeline reason about it - the live playground, but pointed at the run in flight.

WHY A BUS AND NOT A SECOND RUN. The obvious implementation is to have the new
tab call /playground/live, which runs the pipeline and narrates it. For a
problem currently being prepared that is wrong twice over: it pays for a second
set of model calls, and the two runs race to write the same content-hash entry
in the oracle cache, so whichever finishes last silently overwrites the other's
verdict. The upload IS the run. This module only lets a second reader watch it.

WHY A BACKLOG. The teacher clicks a problem some seconds after it started - that
is the whole point, they click BECAUSE it is taking a while. A subscriber that
only received future events would open on a blank page with the run already
three stages in. Every event is kept and replayed to a late subscriber, so the
tab always shows the run from its beginning.

Process memory, deliberately. A channel lives as long as the server does and no
longer: if the process restarts, the run it was narrating died with it, so there
is nothing left to watch and nothing worth persisting. Both the backlog and the
number of channels are bounded - an upload of two hundred problems must not be
able to grow this without limit.
"""
import queue
import threading
from collections import OrderedDict, deque

# One problem's narration is a few hundred events; mutation testing on a
# solution with many mutants is the outlier. Far above a normal run, low enough
# that a pathological one cannot exhaust the process.
MAX_BACKLOG = 5000
# Roughly ten uploads of twenty problems still being held. Oldest FINISHED
# channel is evicted first - a live one is never dropped out from under a run.
MAX_CHANNELS = 200

_SENTINEL = object()
_lock = threading.Lock()
_channels: "OrderedDict[str, _Channel]" = OrderedDict()


class _Channel:
    __slots__ = ("backlog", "subscribers", "done")

    def __init__(self):
        self.backlog = deque(maxlen=MAX_BACKLOG)
        self.subscribers: list[queue.Queue] = []
        self.done = False


def key(assignment_id: str, slug: str) -> str:
    """The address of one problem's preparation within one upload.

    Scoped by assignment, not by slug alone: the same slug is re-prepared every
    time a file is re-uploaded, and two of those runs can overlap."""
    return f"{assignment_id}/{slug}"


def _evict_locked() -> None:
    """Drop finished channels, oldest first, until under the cap. Called with
    the lock held. A channel still running is never evicted."""
    while len(_channels) > MAX_CHANNELS:
        for k, ch in _channels.items():
            if ch.done and not ch.subscribers:
                del _channels[k]
                break
        else:
            return                      # nothing evictable; let it run over


def open_channel(k: str) -> None:
    """Begin a run. Replaces any previous channel at this address - a re-upload
    is a new run, and replaying the old one's transcript under it would be a
    lie about what is currently happening."""
    with _lock:
        _channels[k] = _Channel()
        _channels.move_to_end(k)
        _evict_locked()


def publish(k: str, event: dict) -> None:
    """Record one event and hand it to every current subscriber.

    Never raises and never blocks: this is called from inside the preparation
    pipeline, and a broken or slow watcher must not be able to affect the run it
    is watching. A subscriber whose queue cannot take the event loses it rather
    than stalling preparation."""
    with _lock:
        ch = _channels.get(k)
        if ch is None or ch.done:
            return
        ch.backlog.append(event)
        subscribers = list(ch.subscribers)
    for q in subscribers:
        try:
            q.put_nowait(event)
        except Exception:
            pass


def finish(k: str) -> None:
    """Mark the run over and release every subscriber's stream."""
    with _lock:
        ch = _channels.get(k)
        if ch is None:
            return
        ch.done = True
        subscribers = list(ch.subscribers)
    for q in subscribers:
        try:
            q.put_nowait(_SENTINEL)
        except Exception:
            pass


def is_open(k: str) -> bool:
    with _lock:
        return k in _channels


def subscribe(k: str, heartbeat: float = 15.0):
    """Yield this run's events: the backlog first, then live ones until it ends.

    Yields None as a heartbeat. The caller writes that as a blank line, which
    the NDJSON readers on the other end already skip - an idle connection has to
    put SOMETHING on the wire periodically or a proxy will close it, and a real
    event type would show up in the transcript as noise.

    Ends when the run finishes. An address with no channel ends immediately,
    which is the honest answer to "watch a run that is not happening"."""
    q: queue.Queue = queue.Queue(maxsize=MAX_BACKLOG)
    with _lock:
        ch = _channels.get(k)
        if ch is None:
            return
        # Snapshot and register under ONE lock: registering first would let an
        # event land in the queue that is also still in the backlog we are about
        # to replay, and taking the snapshot first would lose an event arriving
        # in between.
        backlog, done = list(ch.backlog), ch.done
        if not done:
            ch.subscribers.append(q)

    try:
        for event in backlog:
            yield event
        if done:
            return
        while True:
            try:
                event = q.get(timeout=heartbeat)
            except queue.Empty:
                yield None
                continue
            if event is _SENTINEL:
                return
            yield event
    finally:
        with _lock:
            ch = _channels.get(k)
            if ch is not None and q in ch.subscribers:
                ch.subscribers.remove(q)


if __name__ == "__main__":
    import time

    k = key("asg-1", "stack-push")
    open_channel(k)
    publish(k, {"type": "stage", "name": "start"})
    publish(k, {"type": "ground_truth", "code": "..."})

    # A subscriber joining LATE still sees the run from its beginning.
    seen = []
    sub = subscribe(k)
    seen.append(next(sub))
    seen.append(next(sub))
    assert [e["type"] for e in seen] == ["stage", "ground_truth"], seen

    # ...and then follows it live.
    def later():
        time.sleep(0.05)
        publish(k, {"type": "verdict", "strong": True})
        finish(k)
    threading.Thread(target=later, daemon=True).start()
    rest = [e for e in sub if e is not None]
    assert [e["type"] for e in rest] == ["verdict"], rest

    # The run is over: a new subscriber gets the transcript and stops.
    assert [e["type"] for e in subscribe(k) if e] == \
        ["stage", "ground_truth", "verdict"]
    # Watching something that never ran ends rather than hanging.
    assert list(subscribe(key("nope", "nope"))) == []
    # A finished channel accepts no more events - a late write cannot rewrite
    # the transcript of a run that already ended.
    publish(k, {"type": "stage", "name": "sneaky"})
    assert len([e for e in subscribe(k) if e]) == 3

    # Re-opening the same address is a NEW run, not a continuation.
    open_channel(k)
    publish(k, {"type": "stage", "name": "start"})
    finish(k)
    assert [e["type"] for e in subscribe(k) if e] == ["stage"]

    # Eviction keeps only finished, unwatched channels out of the way.
    for i in range(MAX_CHANNELS + 20):
        kk = key("bulk", str(i))
        open_channel(kk)
        finish(kk)
    assert len(_channels) <= MAX_CHANNELS, len(_channels)
    print("prepare_bus.py self-check OK")
