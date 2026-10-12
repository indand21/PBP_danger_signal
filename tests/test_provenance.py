from src.analysis.provenance import provenance_rows, provenance_markdown


def test_rows_cover_host_damage_params():
    names = {r["name"] for r in provenance_rows()}
    for p in ("k_path", "k_infl", "k_heal", "I50", "k_pers"):
        assert p in names


def test_markdown_is_a_table():
    md = provenance_markdown()
    assert md.startswith("| Parameter")
    assert "---" in md
