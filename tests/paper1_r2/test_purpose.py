"""Protect generator/scorer mappings that previously selected fallback profiles."""

import pytest
import numpy as np

from paper1_r2.purpose import canonical_purpose, purpose_weight


@pytest.mark.parametrize('source,expected', [
    ('shopping', 'Shopping'), ('Visit Hospital or other health', 'Health visits'),
    ('Eating/Drinking', 'Eating/drinking'),
    ('Sport/Entertainment', 'Sport/entertainment'), ('Other Journey', 'Other journey'),
])
def test_generated_label_uses_the_intended_flexibility_profile(source, expected):
    name = canonical_purpose(source)
    assert name == expected


def test_typo_is_not_silently_assigned_other_journey():
    with pytest.raises(ValueError, match='Unrecognised'):
        canonical_purpose('shoppign')


def test_full_weight_includes_temporal_blend_and_purpose_charging_assumptions():
    shopping = [.65,.70,.90,.50,.80,.60,.80,.90,.50,.40]
    assert purpose_weight(np.array(shopping)) == pytest.approx(.6929)
    assert purpose_weight(np.array([shopping, shopping])).tolist() == pytest.approx([.6929,.6929])
    with pytest.raises(ValueError):
        purpose_weight(np.array(shopping), base_weights=(.4,.4,.4))
