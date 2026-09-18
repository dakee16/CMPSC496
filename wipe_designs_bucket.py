"""Empty the `designs` storage bucket - the uploaded plan images.

SQL deletes the mt_designs ROWS, which are only paths; the files themselves
live in Supabase Storage and outlive their rows. Run this after the SQL so
nothing is left pointing at a student's uploaded work.

    .venv/bin/python wipe_designs_bucket.py          # list what is there
    .venv/bin/python wipe_designs_bucket.py --delete # actually remove it

Reads SUPABASE_URL / SUPABASE_KEY from .env, same as the app.
"""
import os
import sys

from dotenv import load_dotenv
from supabase import create_client

from main.archive import DESIGN_BUCKET

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
store = sb.storage.from_(DESIGN_BUCKET)


def walk(prefix: str = "") -> list[str]:
    """Every file under `prefix`. The bucket is nested student/slug/file, and
    list() returns one level at a time, so this recurses rather than assuming
    a flat layout."""
    found = []
    try:
        entries = store.list(prefix) or []
    except Exception as e:
        print(f"  ! could not list {prefix or '/'}: {str(e)[:120]}")
        return found
    for item in entries:
        name = item.get("name")
        if not name:
            continue
        path = f"{prefix}/{name}" if prefix else name
        # A folder comes back with no id; a file carries one.
        if item.get("id") is None:
            found += walk(path)
        else:
            found.append(path)
    return found


files = walk()
print(f"{len(files)} file(s) in bucket '{DESIGN_BUCKET}'")
for f in files:
    print("   ", f)

if "--delete" not in sys.argv:
    print("\nNothing deleted. Re-run with --delete to remove these.")
    raise SystemExit(0)

if not files:
    print("\nNothing to delete.")
    raise SystemExit(0)

# In batches: one enormous remove() call is the kind of request that half
# succeeds and leaves you unsure what is still there.
for i in range(0, len(files), 50):
    batch = files[i:i + 50]
    store.remove(batch)
    print(f"  removed {len(batch)}")

left = walk()
print(f"\nDone. {len(left)} file(s) remain.")
