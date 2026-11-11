from datetime import date

import pandas as pd
import pytest

from spend_aware.core.models import Transaction
from spend_aware.data.personas import PERSONAS, generate

START = date(2025, 1, 1)


def test_same_seed_same_output():
    a, ta = generate(PERSONAS["student"], START, months=3, seed=1)
    b, tb = generate(PERSONAS["student"], START, months=3, seed=1)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ta, tb)


def test_different_seed_different_output():
    a, _ = generate(PERSONAS["student"], START, months=3, seed=1)
    b, _ = generate(PERSONAS["student"], START, months=3, seed=2)
    assert not a.equals(b)


def test_rent_appears_weekly():
    tx, truth = generate(PERSONAS["student"], START, months=3)
    rent_ids = truth.query("series_name == 'rent'").transaction_id
    rent_dates = tx[tx.id.isin(rent_ids)].date.sort_values()
    gaps = rent_dates.diff().dt.days.dropna()
    assert len(gaps) >= 10
    assert gaps.between(6, 8).all()


@pytest.mark.parametrize("name", PERSONAS)
def test_all_personas_generate(name):
    tx, truth = generate(PERSONAS[name], START, months=6)
    assert len(tx) > 0
    assert len(truth) == len(tx)
    assert tx.date.min() >= pd.Timestamp(START)
    assert tx.date.max() < pd.Timestamp(2025, 7, 1)
    assert truth.is_recurring.any()
    assert (~truth.is_recurring).any()


@pytest.mark.parametrize("name", PERSONAS)
def test_ids_unique(name):
    tx, _ = generate(PERSONAS[name], START, months=6)
    assert tx.id.is_unique


@pytest.mark.parametrize("name", PERSONAS)
def test_every_truth_row_matches_a_transaction(name):
    tx, truth = generate(PERSONAS[name], START, months=6)
    assert set(truth.transaction_id) == set(tx.id)


def test_bills_never_on_weekends():
    for name, persona in PERSONAS.items():
        tx, truth = generate(persona, START, months=6)
        recurring_ids = truth.query("is_recurring").transaction_id
        days = tx[tx.id.isin(recurring_ids)].date.dt.day_of_week
        assert (days < 5).all(), name


def _running_balance(name: str, months: int = 9, seed: int = 42) -> pd.Series:
    persona = PERSONAS[name]
    tx, _ = generate(persona, START, months=months, seed=seed)
    return persona.opening_balance + tx.groupby("date").amount.sum().cumsum()


def test_four_personas_defined():
    assert set(PERSONAS) == {"student", "salaried", "gig", "edge"}


def test_edge_dips_below_zero():
    assert (_running_balance("edge") < 0).any()


def test_salaried_never_overdrawn():
    assert (_running_balance("salaried") > 0).all()


def test_monthly_bills_track_the_calendar():
    tx, truth = generate(PERSONAS["salaried"], START, months=9)
    salary_ids = truth.query("series_name == 'salary'").transaction_id
    salary_dates = tx[tx.id.isin(salary_ids)].date
    assert len(salary_dates) == 9
    # Paid on the 15th, or the following Monday (16th/17th) when the 15th is a weekend.
    assert salary_dates.dt.day.between(15, 17).all()
    assert salary_dates.dt.month.is_unique


def test_sign_convention():
    tx, _ = generate(PERSONAS["student"], START, months=6)
    assert (tx.loc[tx.category == "income", "amount"] > 0).all()
    assert (tx.loc[tx.category.isin(["rent", "groceries"]), "amount"] < 0).all()


def test_gig_has_no_recurring_income():
    _, truth = generate(PERSONAS["gig"], START, months=6)
    assert not truth.series_name.isin(["pay", "salary"]).any()


def test_truth_labels_are_consistent():
    for name, persona in PERSONAS.items():
        _, truth = generate(persona, START, months=6)
        rec = truth[truth.is_recurring]
        assert rec.series_name.notna().all(), name
        assert rec.cadence_days.notna().all(), name
        assert truth[~truth.is_recurring].series_name.isna().all(), name


@pytest.mark.parametrize("name", PERSONAS)
def test_rows_validate_against_transaction_model(name):
    tx, _ = generate(PERSONAS[name], START, months=3)
    for row in tx.to_dict("records"):
        t = Transaction(**{**row, "date": row["date"].date(), "amount": str(row["amount"])})
        assert t.amount == round(t.amount, 2)


