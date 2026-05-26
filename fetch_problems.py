"""
fetch_problems.py — Scrape 500 free LeetCode problems via GraphQL.
Outputs problems_raw.json for upload_to_supabase.py to ingest.
"""
import json
import re
import time
import requests

GRAPHQL_URL = "https://leetcode.com/graphql"
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://leetcode.com",
}

LIST_QUERY = """
query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {
  problemsetQuestionList: questionList(
    categorySlug: $categorySlug
    limit: $limit
    skip: $skip
    filters: $filters
  ) {
    total: totalNum
    questions: data {
      titleSlug
      title
      difficulty
      topicTags { name }
      paidOnly: isPaidOnly
    }
  }
}
"""

CONTENT_QUERY = """
query questionData($titleSlug: String!) {
  question(titleSlug: $titleSlug) {
    content
  }
}
"""

TARGET = 500
PAGE_SIZE = 50


def strip_html(html: str) -> str:
    if not html:
        return ""
    text = re.sub(r"<[^>]+>", " ", html)
    replacements = {
        "&nbsp;": " ", "&lt;": "<", "&gt;": ">",
        "&amp;": "&", "&quot;": '"', "&#39;": "'",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return re.sub(r"\s+", " ", text).strip()


def graphql(query: str, variables: dict) -> dict:
    r = requests.post(
        GRAPHQL_URL,
        json={"query": query, "variables": variables},
        headers=HEADERS,
        timeout=20,
    )
    r.raise_for_status()
    return r.json()


def fetch_list(skip: int, limit: int) -> list[dict]:
    data = graphql(LIST_QUERY, {
        "categorySlug": "",
        "limit": limit,
        "skip": skip,
        "filters": {},
    })
    return data["data"]["problemsetQuestionList"]["questions"]


def fetch_content(slug: str) -> str:
    try:
        data = graphql(CONTENT_QUERY, {"titleSlug": slug})
        return (data["data"]["question"] or {}).get("content") or ""
    except Exception as e:
        print(f"  ⚠️  content fetch failed for {slug}: {e}")
        return ""


def main():
    collected = []
    skip = 0
    print(f"Fetching {TARGET} free LeetCode problems...\n")

    while len(collected) < TARGET:
        print(f"Page skip={skip}...")
        batch = fetch_list(skip, PAGE_SIZE)
        if not batch:
            print("No more problems available.")
            break

        for q in batch:
            if q.get("paidOnly"):
                continue
            if len(collected) >= TARGET:
                break
            slug = q["titleSlug"]
            print(f"  [{len(collected)+1}/{TARGET}] {slug}")
            content = fetch_content(slug)
            collected.append({
                "slug": slug,
                "title": q["title"],
                "difficulty": q["difficulty"],
                "description": strip_html(content),
                "topic_tags": [t["name"] for t in q.get("topicTags", [])],
                "source": "leetcode",
            })
            time.sleep(0.4)  # polite rate limit

        skip += PAGE_SIZE
        time.sleep(0.5)

    with open("problems_raw.json", "w") as f:
        json.dump(collected, f, indent=2)

    print(f"\n✅ Saved {len(collected)} problems to problems_raw.json")


if __name__ == "__main__":
    main()