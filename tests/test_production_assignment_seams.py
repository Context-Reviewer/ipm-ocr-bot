from production_assignment_seams import (
    ProductionAssignmentRowState,
    allowed_assignment_outputs,
    inspect_production_assignment_seams,
    parse_active_assignment_rows,
)


class FakeRects:
    def __init__(self, rects=None):
        self.rects = rects or {}

    def get(self, key):
        return self.rects.get(key)


class FakeActions:
    def open_smelter_panel(self):
        return True

    def open_crafter_panel(self):
        return True


def test_parse_active_assignment_rows_aggregates_only_active_rows():
    rows = [
        ProductionAssignmentRowState(name="Aluminum Bar", active=True, backend="fake"),
        ProductionAssignmentRowState(name="Iron Bar", active=False, backend="fake"),
        ProductionAssignmentRowState(name="Aluminium Bar", active=True, backend="fake"),
        ProductionAssignmentRowState(name="", active=True, backend="fake"),
    ]

    parsed = parse_active_assignment_rows(rows, allowed_outputs=allowed_assignment_outputs("smelter"))

    assert parsed == {
        "Aluminium Bar": 2,
    }


def test_parse_active_assignment_rows_rejects_unknown_output():
    rows = [
        ProductionAssignmentRowState(name="Unknown Widget", active=True, backend="fake"),
    ]

    try:
        parse_active_assignment_rows(rows, allowed_outputs=allowed_assignment_outputs("crafter"))
    except ValueError as exc:
        assert str(exc) == "unsupported_assignment_row:Unknown Widget"
    else:
        raise AssertionError("expected ValueError for unsupported output")


def test_inspect_production_assignment_seams_reports_exact_missing_rects():
    seams = inspect_production_assignment_seams(
        rects=FakeRects(),
        actions=FakeActions(),
        visible_rows=2,
    )

    expected_missing_rects = [
        "PRODUCTION_PANEL_TEXT",
        "PRODUCTION_TOP_ANCHOR",
        "PRODUCTION_ROW1_READ",
        "PRODUCTION_ROW2_READ",
        "PRODUCTION_ROW1_ACTIVE",
        "PRODUCTION_ROW2_ACTIVE",
    ]
    expected_required_groups = {
        "panel_text": ["PRODUCTION_PANEL_TEXT"],
        "top_anchor": ["PRODUCTION_TOP_ANCHOR"],
        "row_labels": ["PRODUCTION_ROW1_READ", "PRODUCTION_ROW2_READ"],
        "row_active_indicators": ["PRODUCTION_ROW1_ACTIVE", "PRODUCTION_ROW2_ACTIVE"],
    }

    assert seams["smelter"] == {
        "feasible": False,
        "blocker": (
            "missing calibrated rects: PRODUCTION_PANEL_TEXT, PRODUCTION_TOP_ANCHOR, "
            "PRODUCTION_ROW1_READ, PRODUCTION_ROW2_READ, PRODUCTION_ROW1_ACTIVE, PRODUCTION_ROW2_ACTIVE"
        ),
        "reader_contract": (
            "read each visible production row as ProductionAssignmentRowState(name=<canonical output>, active=<bool>, backend=<reader>)"
        ),
        "parser_contract": "aggregate only active rows into dict[output_name, active_count]",
        "parser_helper_available": True,
        "required_rect_groups": expected_required_groups,
        "missing_rects": expected_missing_rects,
        "navigation": {
            "open_smelter_panel": True,
            "open_crafter_panel": True,
        },
    }
    assert seams["crafter"] == {
        "feasible": False,
        "blocker": (
            "missing calibrated rects: PRODUCTION_PANEL_TEXT, PRODUCTION_TOP_ANCHOR, "
            "PRODUCTION_ROW1_READ, PRODUCTION_ROW2_READ, PRODUCTION_ROW1_ACTIVE, PRODUCTION_ROW2_ACTIVE"
        ),
        "reader_contract": (
            "read each visible production row as ProductionAssignmentRowState(name=<canonical output>, active=<bool>, backend=<reader>)"
        ),
        "parser_contract": "aggregate only active rows into dict[output_name, active_count]",
        "parser_helper_available": True,
        "required_rect_groups": expected_required_groups,
        "missing_rects": expected_missing_rects,
        "navigation": {
            "open_smelter_panel": True,
            "open_crafter_panel": True,
        },
    }
