from alignment_manifold.prompts import build_smoke_records


def test_smoke_prompt_shape_and_ids() -> None:
    records = build_smoke_records()
    assert len(records) == 200
    assert len({record["example_id"] for record in records}) == 200
    assert len({record["cluster_id"] for record in records}) == 50
    assert {record["category"] for record in records} == {
        "helpfulness",
        "instruction_following",
        "honesty",
        "safety",
    }


def test_each_semantic_cluster_has_four_variants() -> None:
    records = build_smoke_records()
    cluster_counts = {}
    for record in records:
        cluster_counts[record["cluster_id"]] = cluster_counts.get(record["cluster_id"], 0) + 1
    assert set(cluster_counts.values()) == {4}

