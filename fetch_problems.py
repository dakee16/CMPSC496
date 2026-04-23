import json
import time
import re
import os
import requests
from dotenv import load_dotenv
from supabase import create_client
from ollama_client import chat

load_dotenv()

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"],
)

GRAPHQL_URL = "https://leetcode.com/graphql"
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://leetcode.com"
}

MODEL = "qwen2.5:7b-instruct"


# ---------------------------------------------------------------------------
# Fetch hints + Python template from LeetCode
# ---------------------------------------------------------------------------

def fetch_meta(slug: str) -> dict:
    """Fetch hints and Python code template for a problem."""
    query = """
    query questionData($titleSlug: String!) {
      question(titleSlug: $titleSlug) {
        hints
        codeSnippets {
          lang
          code
        }
      }
    }
    """
    try:
        r = requests.post(
            GRAPHQL_URL,
            json={"query": query, "variables": {"titleSlug": slug}},
            headers=HEADERS,
            timeout=15,
        )
        data = r.json()["data"]["question"]
        hints = data.get("hints", [])
        snippets = data.get("codeSnippets", [])
        py_template = next(
            (s["code"] for s in snippets if "Python" in s["lang"]), ""
        )
        return {"hints": hints, "python_template": py_template}
    except Exception as e:
        print(f"  ⚠️  Could not fetch meta for {slug}: {e}")
        return {"hints": [], "python_template": ""}


# ---------------------------------------------------------------------------
# Generate ground truth solution via local LLM
# ---------------------------------------------------------------------------

def generate_solution(title: str, description: str,
                       hints: list, template: str) -> str:
    """Use the local LLM to generate a reference Python solution."""
    hints_text = "\n".join(f"- {h}" for h in hints) if hints else "None provided."
    prompt = (
        f"You are an expert competitive programmer.\n\n"
        f"Problem: {title}\n\n"
        f"Description:\n{description[:1000]}\n\n"
        f"Hints:\n{hints_text}\n\n"
        f"Python function template to complete:\n{template}\n\n"
        "Write a clean, correct, complete Python solution. "
        "Return ONLY the Python code — no explanation, no markdown fences."
    )
    raw = chat(
        MODEL,
        "You are an expert competitive programmer. Return only clean Python code, no markdown.",
        [{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    # Strip any markdown fences the model adds
    raw = re.sub(r'```[\w]*\n?', '', raw)
    raw = re.sub(r'```', '', raw)
    return raw.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Only fetch for problems that don't have a solution yet
    res = (
        supabase.table("problems")
        .select("id, slug, title, description")
        .is_("solution", "null")
        .limit(500)
        .execute()
    )
    problems = res.data

    if not problems:
        print("✅ All problems already have solutions.")
        return

    print(f"Generating solutions for {len(problems)} problems...\n")

    for i, p in enumerate(problems):
        slug = p["slug"]
        print(f"[{i+1}/{len(problems)}] {slug}")

        # Get hints + template from LeetCode
        meta = fetch_meta(slug)
        time.sleep(0.8)  # polite rate limiting

        # Generate solution via LLM
        solution = generate_solution(
            p["title"],
            p["description"] or "",
            meta["hints"],
            meta["python_template"],
        )

        if not solution:
            print(f"  ⚠️  Empty solution — skipping.")
            continue

        # Save to Supabase
        try:
            supabase.table("problems").update(
                {"solution": solution}
            ).eq("id", p["id"]).execute()
            print(f"  ✅ Saved ({len(solution)} chars)")
        except Exception as e:
            print(f"  ⚠️  Failed to save: {e}")

        time.sleep(1.0)  # LLM cooldown

    print("\nDone. All solutions generated.")


if __name__ == "__main__":
    main()