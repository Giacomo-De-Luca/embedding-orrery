"""Counting the documents an embedding run truncates (the non-cancelling warning)."""

from backend.utils.truncation import TokenLengths, TruncationTracker


class FakeTokenizer:
    """Whitespace tokenizer recording the keyword arguments of each call."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, texts, **kwargs):
        self.calls.append(kwargs)
        return {"input_ids": [[0] * len(text.split()) for text in texts]}


def test_token_lengths_count_every_token_without_truncating():
    tokenizer = FakeTokenizer()

    assert TokenLengths.of(tokenizer, ["a b c", "d"]) == [3, 1]
    call = tokenizer.calls[0]
    assert call["truncation"] is False
    assert call["padding"] is False
    assert call["add_special_tokens"] is True


class TestTracker:
    def test_inactive_without_a_limit(self):
        tracker = TruncationTracker(None)

        assert tracker.active is False
        assert tracker.observe([10**6]) == 0
        assert tracker.warning() is None

    def test_counts_only_documents_longer_than_the_limit(self):
        tracker = TruncationTracker(4)

        assert tracker.observe([3, 4, 5, 9]) == 2
        assert (tracker.truncated, tracker.seen, tracker.longest) == (2, 4, 9)

    def test_counts_accumulate_across_batches(self):
        tracker = TruncationTracker(4)

        assert tracker.observe([5, 1]) == 1
        assert tracker.observe([1, 1]) == 0
        assert tracker.observe([6, 7]) == 2
        assert (tracker.truncated, tracker.seen, tracker.longest) == (3, 6, 7)

    def test_no_warning_while_nothing_was_truncated(self):
        tracker = TruncationTracker(512)
        tracker.observe([10, 512])

        assert tracker.warning() is None

    def test_warning_reports_counts_limit_and_longest(self):
        tracker = TruncationTracker(512)
        tracker.observe([600, 100, 1200])

        warning = tracker.warning()
        assert "2 of 3 documents" in warning
        assert "512-token" in warning
        assert "1,200" in warning


class TestCount:
    def test_counts_through_the_tokenizer(self):
        tracker = TruncationTracker(2)

        assert tracker.count(FakeTokenizer(), ["a b c", "d"]) == 1
        assert (tracker.truncated, tracker.seen) == (1, 2)

    def test_a_failing_tokenizer_disables_counting_instead_of_failing_the_job(self):
        def broken(texts, **kwargs):
            raise TypeError("not a text tokenizer")

        tracker = TruncationTracker(2)

        assert tracker.count(broken, ["a b c"]) == 0
        assert tracker.active is False
        assert tracker.count(FakeTokenizer(), ["a b c"]) == 0

    def test_a_missing_tokenizer_disables_counting(self):
        tracker = TruncationTracker(2)

        assert tracker.count(None, ["a b c"]) == 0
        assert tracker.active is False
