"""Reciprocal rank fusion: merging, weights, empty inputs."""

from src.search.rrf import fuse


def test_fusion_prefers_documents_in_both_lists():
    keyword = [(1, 9.0), (0, 8.0), (2, 7.0)]
    dense = [(2, 0.9), (1, 0.8), (3, 0.7)]
    fused = dict(fuse([keyword, dense], k=60))
    assert fused[1] > fused[0]
    assert fused[2] > fused[3]


def test_weights_change_the_order():
    keyword = [(0, 5.0), (1, 4.0)]
    dense = [(1, 0.9), (0, 0.1)]
    keyword_heavy = dict(fuse([keyword, dense], weights=[3.0, 1.0]))
    assert keyword_heavy[0] > keyword_heavy[1]


def test_empty_input():
    assert fuse([[], []]) == []
