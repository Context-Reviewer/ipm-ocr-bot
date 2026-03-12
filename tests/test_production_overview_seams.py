from ipm.state import ProductionOverviewCardState
from production_overview_seams import (
    allowed_overview_outputs,
    parse_active_overview_cards,
    required_production_overview_rects,
)


def test_parse_active_overview_cards_counts_only_active_supported_outputs():
    cards = [
        ProductionOverviewCardState(slot_index=1, tab="smelt", output_name="Copper Bar", active=True, backend="fake"),
        ProductionOverviewCardState(slot_index=2, tab="smelt", output_name="Iron Bar", active=False, backend="fake"),
        ProductionOverviewCardState(slot_index=3, tab="smelt", output_name="Copper Bar", active=True, backend="fake"),
        ProductionOverviewCardState(slot_index=4, tab="smelt", output_name="", active=False, backend="fake"),
    ]

    assert parse_active_overview_cards(cards, allowed_outputs=allowed_overview_outputs("smelt")) == {
        "Copper Bar": 2,
    }


def test_required_production_overview_rects_matches_four_slot_contract():
    assert required_production_overview_rects() == [
        "PRODUCTION_CARD1",
        "PRODUCTION_CARD2",
        "PRODUCTION_CARD3",
        "PRODUCTION_CARD4",
    ]
