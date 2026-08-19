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

# chunk_pool.json is now keyed by a content hash of each problem's
# description + solution (main/identity.py:content_hash), not by slug --
# that migration happened after this script was originally written. This
# script only has slugs, and a slug alone can't reproduce the hash (the hash
# needs the full description and solution text), so a slug-based lookup will
# report every entry as "missing" against the new pool even once problems
# have been reprocessed. Fixing this requires querying Supabase for each
# slug's full problem row and hashing that, not just patching the lookup key.
pool = json.load(open("main/chunk_pool.json"))
total_components = 0
missing = []

for slug in REVIEWED_SLUGS:
    entries = pool.get(slug, [])
    if not entries:
        missing.append(slug)
        continue
    # Uses the first pooled decomposition for that slug --
    # change index if a different one was reviewed
    total_components += len(entries[0]["chunks"])

print(f"Total components across {len(REVIEWED_SLUGS) - len(missing)} slugs: {total_components}")
if missing:
    print(f"WARNING: not found in pool, count manually: {missing}")