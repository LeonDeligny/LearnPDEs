"""Explicit catalog of runnable scenarios, preserving public names and aliases."""

from __future__ import annotations

from collections.abc import Iterable
from types import MappingProxyType

from learnpdes import POTENTIAL_FLOW_SCENARIO, SOLENOIDAL_FLOW_SCENARIO
from learnpdes.scenarios import (
    airfoil,
    circular_couette,
    cosinus,
    cylinder,
    exponential,
    forced_linear,
    kovasznay,
    laplace,
    logistic,
    poiseuille,
    wind_tunnel,
)
from learnpdes.scenarios.base import Scenario


def make_registry(cases: Iterable[Scenario]) -> MappingProxyType[str, Scenario]:
    catalog = {}
    for case in cases:
        if case.name in catalog:
            raise ValueError(f'Duplicate scenario: {case.name}')
        catalog[case.name] = case
    return MappingProxyType(catalog)


SCENARIOS: MappingProxyType[str, Scenario] = make_registry(
    cases=(
        exponential.SCENARIO,
        forced_linear.SCENARIO,
        logistic.SCENARIO,
        cosinus.SCENARIO,
        laplace.SCENARIO,
        poiseuille.SCENARIO,
        kovasznay.SCENARIO,
        circular_couette.SCENARIO,
        cylinder.SCENARIO,
        wind_tunnel.SCENARIO,
        airfoil.POTENTIAL,
        airfoil.STREAMFUNCTION,
    )
)


def get_scenario(name: str) -> Scenario:
    """Accept canonical CLI names and the original Python flow identifiers."""
    name = {
        POTENTIAL_FLOW_SCENARIO: 'potential-flow',
        SOLENOIDAL_FLOW_SCENARIO: 'solenoidal-flow',
    }.get(name, name)
    try:
        return SCENARIOS[name]
    except KeyError:
        raise ValueError(
            f'Unknown scenario {name!r}. Choose from: {", ".join(SCENARIOS)}'
        ) from None
