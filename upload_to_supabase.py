import json
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"]
)

def upload_problems(path="problems_raw.json"):
    with open(path) as f:
        problems = json.load(f)

    print(f"Uploading {len(problems)} problems to Supabase...")
    success = 0
    skipped = 0

    for p in problems:
        try:
            supabase.table("problems").upsert(
                p, on_conflict="slug"  # skip duplicates
            ).execute()
            success += 1
            print(f"  ✅ {p['slug']}")
        except Exception as e:
            print(f"  ⚠️  Skipped {p['slug']}: {e}")
            skipped += 1

    print(f"\nDone. {success} uploaded, {skipped} skipped.")


if __name__ == "__main__":
    upload_problems()