from ipm.domain_data import (
    ORE_NAMES,
    PLANET_NAMES,
    is_plausible_planet_title,
    normalize_ore_name,
    normalize_resource_row_name,
    normalize_planet_name,
    resource_row_name_reject_reason,
)


def test_ore_vocab_preserves_repo_canonical_silica():
    assert "Silica" in ORE_NAMES
    assert "Silicon" not in ORE_NAMES


def test_normalize_ore_name_maps_seed_alias_to_repo_canon():
    assert normalize_ore_name("Silicon") == "Silica"
    assert normalize_ore_name("Aluminium") == "Aluminum"


def test_resource_row_vocab_expands_beyond_strict_ore_canon():
    assert normalize_resource_row_name("Sulfur") == "Sulfur"
    assert normalize_resource_row_name("Lithium") == "Lithium"
    assert normalize_resource_row_name("Hydrogen") == "Hydrogen"


def test_resource_row_normalization_rejects_live_contamination_examples():
    assert resource_row_name_reject_reason('The ore or resource name visible in the row is "Sulfur."') == "prose_wrapper"
    assert resource_row_name_reject_reason('The resource name visible in the row is "Iron".') == "prose_wrapper"
    assert resource_row_name_reject_reason("Ship Speed") == "ui_text"
    assert resource_row_name_reject_reason("8.92 mkph") == "digit_text"
    assert resource_row_name_reject_reason("v. 24") == "digit_text"
    assert resource_row_name_reject_reason("100%") == "digit_text"
    assert normalize_resource_row_name("Cooper 390") == ""


def test_normalize_planet_name_handles_numbered_titles():
    assert normalize_planet_name("8. ACHERON") == "Acheron"
    assert "Acheron" in PLANET_NAMES


def test_normalize_planet_name_strips_leading_o_prefix_noise_for_known_title():
    assert normalize_planet_name("O. SOLVEIG") == "Solveig"


def test_normalize_planet_name_strips_colony_level_suffix():
    assert normalize_planet_name("1. BALOR Colony Lv 1") == "Balor"


def test_normalize_planet_name_keeps_existing_decoration_only_cases():
    assert normalize_planet_name(". YANGTZE") == "Yangtze"
    assert normalize_planet_name(". ACHERON") == "Acheron"


def test_planet_title_plausibility_rejects_sentence_and_keeps_operational_titles():
    assert is_plausible_planet_title("8. ACHERON") is True
    assert is_plausible_planet_title("8. ACHEAON") is True
    assert is_plausible_planet_title("The visible planet title is Water Planet.") is False
