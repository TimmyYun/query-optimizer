"""Adapters must take the benchmarks' own predicates unchanged."""
from prepare_benchmark import dsb_extract_ranges, ops_to_range, stats_parse_subqueries


def test_dsb_extracts_plain_between_in_cents():
    sql = """
        and ws_wholesale_cost BETWEEN 34 AND 54
        and ws1.ws_list_price between 120 and 130
        and ws_list_price between 7.50 and 12.25)
    """
    assert dsb_extract_ranges(sql, "ws_wholesale_cost", 100) == [(3400, 5400)]
    assert dsb_extract_ranges(sql, "ws_list_price", 100) == [(12000, 13000), (750, 1225)]


def test_dsb_skips_ratio_and_other_tables():
    sql = """
        and ws_sales_price / ws_list_price BETWEEN 15 * 0.01 AND 35 * 0.01
        and cs_wholesale_cost BETWEEN 1 AND 21
    """
    assert dsb_extract_ranges(sql, "ws_list_price", 100) == []
    assert dsb_extract_ranges(sql, "ws_wholesale_cost", 100) == []


def test_ops_to_range_intersects_and_opens_to_column_domain():
    assert ops_to_range([(">=", 3), ("<=", 10)], 0, 100) == (3, 10)
    assert ops_to_range([(">", 3), ("<", 10)], 0, 100) == (4, 9)
    assert ops_to_range([(">=", 5)], -19, 192) == (5, 192)
    assert ops_to_range([("=", 7)], 0, 100) == (7, 7)


def test_stats_marks_only_truly_1d_subqueries_for_truth_check(tmp_path):
    f = tmp_path / "sub.sql"
    f.write_text(
        "SELECT COUNT(*) FROM users as u WHERE u.UpVotes>=0;||0||40325\n"
        "SELECT COUNT(*) FROM users as u WHERE u.Reputation>=1 AND u.CreationDate<='2014-09-11 14:33:06'::timestamp;||1||37520\n"
        "SELECT COUNT(*) FROM posts as p WHERE p.Score>=-1 AND p.Score<=10 AND p.ViewCount>=0;||2||100\n"
    )
    per_col = stats_parse_subqueries(f)
    assert per_col["users.UpVotes"] == [([(">=", 0)], 40325)]
    # the timestamp filter makes it 2-D: projected, but not used for the truth check
    assert per_col["users.Reputation"] == [([(">=", 1)], None)]
    assert per_col["posts.Score"] == [([(">=", -1), ("<=", 10)], None)]
    assert "users.CreationDate" not in per_col
