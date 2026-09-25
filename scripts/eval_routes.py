"""
Small eval harness: runs a fixed set of queries through the router and
prints the routing DECISION for each (score, margin, chosen route) without
calling any handler — so you can tune SIMILARITY_THRESHOLD / AMBIGUITY_MARGIN
in router.py and re-run this freely without spending any Gemini calls.

This only exercises the embedding router in isolation (router.classify),
not the full pipeline in core.handle_query — so "not_confident" below
means "the embedding match alone wasn't good enough," regardless of
whether Gemini's classification attempt would go on to rescue it into a
real route. Includes clear-cut queries for every route plus deliberately
ambiguous / out-of-scope ones that should fail to confidently match. This
table is also the evidence you'd put in a README or talk through in an
interview.
"""

from app.router import SemanticRouter, validate_input

TEST_QUERIES = [
    # clear-cut — should route confidently to a real handler. Phrased
    # deliberately differently from the utterances in routes.yaml (synonyms,
    # informal phrasing, typos) rather than near-copies of them, so this is
    # a fair test of generalization rather than an exact-match check.
    ("Can you tell me my balance", "balance_query"),
    ("How much do I have in my account", "balance_query"),
    ("What's my current balance", "balance_query"),
    ("I want to check my balance", "balance_query"),
    ("Balance check please", "balance_query"),
    ("How much cash do I have right now", "balance_query"),
    ("What's the balance on my account", "balance_query"),
    ("Tell me how much money is in my account", "balance_query"),
    ("Current account balance", "balance_query"),
    ("How much do I have left to spend", "balance_query"),
    ("What is my available balance", "balance_query"),
    ("Need to see my balance", "balance_query"),
    ("How's my account looking", "balance_query"),
    ("Whats in my account rn", "balance_query"),
    ("how much money do i got", "balance_query"),
    ("account balance pls", "balance_query"),
    ("show me how much i have", "balance_query"),
    ("what's my balance at the moment", "balance_query"),
    ("I'd like to know my balance", "balance_query"),
    ("give me my balance", "balance_query"),

    ("Can I get a refund", "refund_request"),
    ("I need my money back", "refund_request"),
    ("Please refund this purchase", "refund_request"),
    ("This order was wrong, I want a refund", "refund_request"),
    ("How do I get a refund", "refund_request"),
    ("I'd like to return this and get my money back", "refund_request"),
    ("Reverse this charge please", "refund_request"),
    ("I want to cancel this transaction and get refunded", "refund_request"),
    ("Refund my last purchase", "refund_request"),
    ("This charge was a mistake, refund it", "refund_request"),
    ("Send my money back", "refund_request"),
    ("I'm not happy with this order, refund me", "refund_request"),
    ("Undo this payment", "refund_request"),
    ("Give me my money back for that order", "refund_request"),
    ("Requesting a refund on my recent purchase", "refund_request"),
    ("Can you reverse this payment", "refund_request"),
    ("I want a chargeback", "refund_request"),
    ("Please return my funds", "refund_request"),
    ("refund needed asap", "refund_request"),
    ("my order was cancelled, wheres my refund", "refund_request"),

    ("Show me what I've spent recently", "transaction_history"),
    ("Can I see my past transactions", "transaction_history"),
    ("List my recent purchases", "transaction_history"),
    ("What have I bought this week", "transaction_history"),
    ("Pull up my transaction history", "transaction_history"),
    ("Show me my spending", "transaction_history"),
    ("I want to see my last few payments", "transaction_history"),
    ("Can you show my recent activity", "transaction_history"),
    ("What transactions have I made", "transaction_history"),
    ("Give me a list of my payments", "transaction_history"),
    ("Show all my recent charges", "transaction_history"),
    ("What did I buy last week", "transaction_history"),
    ("recent purchases pls", "transaction_history"),
    ("show spending history", "transaction_history"),
    ("list of transactions this month", "transaction_history"),
    ("what's my spending been like", "transaction_history"),
    ("can i get my statement", "transaction_history"),
    ("show me my account activity", "transaction_history"),
    ("past payments list", "transaction_history"),
    ("transactions from last week", "transaction_history"),

    ("hey", "small_talk"),
    ("good morning", "small_talk"),
    ("what's up", "small_talk"),
    ("how's it going", "small_talk"),
    ("hiya", "small_talk"),
    ("yo", "small_talk"),
    ("greetings", "small_talk"),
    ("thank you so much", "small_talk"),
    ("appreciate it", "small_talk"),
    ("thanks for the help", "small_talk"),
    ("cheers", "small_talk"),
    ("nice one thanks", "small_talk"),
    ("hey there", "small_talk"),
    ("good evening", "small_talk"),
    ("howdy", "small_talk"),
    ("much appreciated", "small_talk"),
    ("thanks a bunch", "small_talk"),
    ("hello there", "small_talk"),
    ("sup", "small_talk"),
    ("have a good day", "small_talk"),

    # ambiguous / out-of-scope — the embedding router alone shouldn't be
    # confident about any of these (Gemini may still rescue them in the
    # full pipeline, but that's not what this script measures)
    ("Why does my card keep getting declined", "not_confident"),
    ("Can you explain how interest is calculated", "not_confident"),
    ("balance transfer refund history", "not_confident"),  # deliberately straddles 3 routes
    ("asdkjhasdkjh", "not_confident"),
    ("What's the exchange rate today", "not_confident"),
    ("Can I increase my credit limit", "not_confident"),
    ("How do I reset my password", "not_confident"),
    ("Is my account secure", "not_confident"),
    ("What are your business hours", "not_confident"),
    ("Can I speak to a human", "not_confident"),
    ("How do I close my account", "not_confident"),
    ("What's the weather like today", "not_confident"),
    ("Tell me a joke", "not_confident"),
    ("What is the meaning of life", "not_confident"),
    ("12345", "not_confident"),
    ("How do loans work here", "not_confident"),
    ("Can you help me invest my money", "not_confident"),
    ("What's your policy on overdrafts", "not_confident"),
    ("I lost my card, what do I do", "not_confident"),
    ("How do I update my address", "not_confident"),
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
        got = match.route if match.confident else "not_confident"
        margin = match.score - match.runner_up_score
        ok = "OK" if got == expected else "MISS"
        if got == expected:
            correct += 1
        print(f"{query[:45]:45} {expected:20} {got:20} {match.score:6.3f} {margin:7.3f}  {ok}")

    print(f"\n{correct}/{len(TEST_QUERIES)} matched expected route")


if __name__ == "__main__":
    run()
