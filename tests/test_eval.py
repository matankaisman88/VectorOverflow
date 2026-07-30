from __future__ import annotations

import pytest

from so_rag.eval import recall_at_k, reciprocal_rank


def test_recall_at_k_perfect_match_at_rank_one() -> None:
    assert recall_at_k([10, 20, 30], [10, 20], k=3) == 1.0


def test_recall_at_k_match_outside_top_k() -> None:
    assert recall_at_k([1, 2, 3], [99], k=3) == 0.0


def test_recall_at_k_no_relevant_ids() -> None:
    assert recall_at_k([1, 2, 3], [], k=3) == 0.0


def test_recall_at_k_empty_retrieved() -> None:
    assert recall_at_k([], [1], k=3) == 0.0


def test_reciprocal_rank_first_relevant_at_rank_one() -> None:
    assert reciprocal_rank([5, 10, 15], [5]) == 1.0


def test_reciprocal_rank_first_relevant_at_rank_three() -> None:
    assert reciprocal_rank([1, 2, 5], [5]) == pytest.approx(1 / 3)


def test_reciprocal_rank_no_relevant_in_list() -> None:
    assert reciprocal_rank([1, 2, 3], [99]) == 0.0


def test_reciprocal_rank_empty_retrieved() -> None:
    assert reciprocal_rank([], [1]) == 0.0
