"""
Small eval harness: runs a fixed set of queries through the router and
prints the routing DECISION for each (score, margin, chosen route) without
calling any handler — so you can tune SIMILARITY_THRESHOLD / AMBIGUITY_MARGIN
in router.py and re-run this freely without spending any Gemini calls.

Includes clear-cut queries for every route plus deliberately ambiguous /
out-of-scope ones that should fall through to llm_fallback. This table is
also the evidence you'd put in a README or talk through in an interview.
"""

from app.router import SemanticRouter, validate_input

TEST_QUERIES = [
    # clear-cut — should route confidently to a real handler
    ("What's my account balance?", "balance_query"),
    ("How much money do I have left", "balance_query"),
    ("I need a refund for my last order", "refund_request"),
    ("Cancel my payment and give me my money back", "refund_request"),
    ("Show me my recent transactions", "transaction_history"),
    ("What did I spend this month", "transaction_history"),
    ("hello", "small_talk"),
    ("thanks a lot", "small_talk"),

    # ambiguous / out-of-scope — should fall back to the LLM
    ("Why does my card keep getting declined", "llm_fallback"),
    ("Can you explain how interest is calculated", "llm_fallback"),
    ("balance transfer refund history", "llm_fallback"),  # deliberately straddles 3 routes
    ("asdkjhasdkjh", "llm_fallback"),
]


def run():
    router = SemanticRouter(config_path="app/config/routes.yaml")
    correct = 0

    header = f"{'query':45} {'expected':20} {'got':20} {'score':>6} {'margin':>7}"
    print(header)
    print("-" * len(header))

    for query, expected in TEST_QUERIES:
        clean = validate_input(query)
        match = router.classify(clean)
        got = match.route if match.confident else "llm_fallback"
        margin = match.score - match.runner_up_score
        ok = "OK" if got == expected else "MISS"
        if got == expected:
            correct += 1
        print(f"{query[:45]:45} {expected:20} {got:20} {match.score:6.3f} {margin:7.3f}  {ok}")

    print(f"\n{correct}/{len(TEST_QUERIES)} matched expected route")


if __name__ == "__main__":
    run()
