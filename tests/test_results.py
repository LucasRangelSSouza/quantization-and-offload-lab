import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "results" / "summary.json").read_text())


def test_awq_is_much_smaller_than_nvfp4_and_fp8():
    assert S["weights_awq_gib"] < 0.7 * S["weights_nvfp4_gib"]
    assert abs(S["weights_nvfp4_gib"] - S["weights_fp8_gib"]) < 1.0  # this NVFP4 checkpoint is no smaller than FP8


def test_no_precision_is_detectably_worse():
    assert S["min_paired_p"] > 0.05 / 12  # no paired difference survives a Bonferroni correction over 12 tests


def test_on_card_routes_sit_near_one_terabyte_per_second():
    assert 700 < S["implied_gb_s_on_card_min"] and S["implied_gb_s_on_card_max"] < 1300


def test_offload_is_two_orders_of_magnitude_slower():
    assert S["implied_gb_s_streamed"] < 0.02 * S["implied_gb_s_on_card_min"]
    assert S["airllm_days_for_eval"] > 1
