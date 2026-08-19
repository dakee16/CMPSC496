from main.mutation import evaluate_oracle

def test_unrepairable_boundary_gap_is_weak():
    """A gap the free probes and LLM search genuinely cannot find must stay
    WEAK — it must not be silently excused as 'likely equivalent'."""
    solution = "def isBig(n):\n    return n > 100"
    problem = {"slug": "test", "title": "t", "description": "d", "solution": solution}
    # Neither test input is anywhere near the n=100 boundary, and neither is
    # any free-probe variant of 50 or 200 (0, 1, -1, ±50, ±200, offsets by 1).
    oracle = [{"input": [50], "expected": False}, {"input": [200], "expected": True}]

    result = evaluate_oracle(problem, oracle)
    assert result["strong"] is False, (
        f"boundary gap at n=100 was hidden (kill_rate={result['kill_rate']})")

if __name__ == "__main__":
    test_unrepairable_boundary_gap_is_weak()
    print("PASSED")