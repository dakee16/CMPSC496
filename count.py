import json

# Paste the exact slugs that were manually reviewed here. This list was
# previously corrupted -- a full chunk_pool.json dump had been pasted into
# the middle of it (turning one list element into a dict), which made
# pool.get(slug) raise TypeError: unhashable type the moment it ran.
#
# That dump has been removed. The real list of 30 reviewed slugs was never
# recorded when the manual review happened (see: SIGCSE red-block 10/N,
# still open) -- these three are what survived. Replace this list with the
# actual reviewed set once it's reconstructed or the review is redone.
REVIEWED_SLUGS = [
    "palindrome-number", "two-sum", "roman-to-integer",
]

# chunk_pool.json is keyed by a content hash of each problem's description +
# solution (main/identity.py:content_hash), not by slug. A slug alone cannot
# reproduce that hash, so the lookup pulls each problem's full row from
# Supabase and hashes that -- otherwise every entry reports as "missing".
import os

from dotenv import load_dotenv
from supabase import create_client

from main.identity import content_hash

load_dotenv()

POOL_PATH = "main/chunk_pool.json"
if not os.path.exists(POOL_PATH):
    # Untracked runtime cache (commit 9740a7a); absent until a decomposition
    # populates it. Nothing to count -- and "0 components" would be a wrong
    # answer, not an empty one.
    raise SystemExit(f"{POOL_PATH} does not exist yet -- run a decomposition "
                     f"first, there is nothing to count.")

pool = json.load(open(POOL_PATH))
_sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
_rows = {p["slug"]: p for p in (_sb.table("problems").select(
    "slug, title, description, difficulty, solution").execute().data or [])}

total_components = 0
missing = []

for slug in REVIEWED_SLUGS:
    row = _rows.get(slug)
    entries = pool.get(content_hash(row), []) if row else []
    if not entries:
        missing.append(slug)
        continue
    # Uses the first pooled decomposition for that slug --
    # change index if a different one was reviewed
    total_components += len(entries[0]["chunks"])

print(f"Total components across {len(REVIEWED_SLUGS) - len(missing)} slugs: {total_components}")
if missing:
    print(f"WARNING: not found in pool, count manually: {missing}")