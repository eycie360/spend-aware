"""Synthetic bank-history generator with ground-truth labels.

Each persona's recurring series and discretionary habits are described as data
(``PERSONAS``). ``generate`` turns a persona into a transactions DataFrame plus a
ground-truth DataFrame that says which transactions belong to which recurring series.

Run as a module to write CSVs::

    python -m spend_aware.data.personas --persona student
"""

import re
import string
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Persona specs use dates relative to this reference. ``generate`` re-anchors them to
# whatever ``start`` it is given, so the same persona works for any time window.
REFERENCE_START = date(2025, 1, 1)

MONTHLY_CADENCES = range(28, 32)  # cadence_days in this range means "same day each month"


@dataclass
class RecurringSpec:
    merchant: str
    description: str
    amount: float  # negative for bills, positive for income
    cadence_days: int  # 7, 14, 30 (approx monthly), 365
    start: date
    amount_jitter: float = 0.0  # e.g. 0.08 = ±8% (power bills vary)
    day_jitter: int = 0  # e.g. 1 = may land a day early or late
    category: str = "bills"
    series_name: str | None = None  # ground-truth label; defaults to a slug of merchant

    @property
    def series(self) -> str:
        return self.series_name or re.sub(r"[^a-z0-9]+", "_", self.merchant.lower()).strip("_")


@dataclass
class Persona:
    name: str
    opening_balance: float
    recurring: list[RecurringSpec]
    discretionary: list[
        dict
    ]  # e.g. {"merchant": "countdown", "mean": -85, "sd": 25, "per_week": 2}
    one_offs: list[dict] = field(default_factory=list)  # {"merchant", "description", "amount"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shift_weekend(d: date) -> date:
    """Bills that fall on Saturday/Sunday are processed the following Monday."""
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _bank_desc(template: str, rng: np.random.Generator) -> str:
    """Fill ``{ref}`` (6 alphanumerics) and ``{num}`` (4 digits) with random values."""
    alphabet = list(string.ascii_uppercase + string.digits)
    out = template
    while "{ref}" in out:
        out = out.replace("{ref}", "".join(rng.choice(alphabet, size=6)), 1)
    while "{num}" in out:
        out = out.replace("{num}", f"{rng.integers(0, 10_000):04d}", 1)
    return out


def _recurring_dates(
    spec: RecurringSpec, start: date, end: date, rng: np.random.Generator
) -> list[date]:
    """All posting dates for one series in ``[start, end)``."""
    offset = (spec.start - REFERENCE_START).days
    scheduled: list[date] = []

    if spec.cadence_days in MONTHLY_CADENCES:
        # Same calendar day each month. Always offset from the anchor (not the previous
        # date) so a bill on the 31st doesn't drift to the 28th after February.
        anchor = pd.Timestamp(start.year, start.month, 1) + pd.Timedelta(days=spec.start.day - 1)
        k = 0
        while True:
            d = (anchor + pd.DateOffset(months=k)).date()
            if d >= end:
                break
            if d >= start:
                scheduled.append(d)
            k += 1
    else:
        d = start + timedelta(days=offset)
        while d < end:
            scheduled.append(d)
            d += timedelta(days=spec.cadence_days)

    dates = []
    for d in scheduled:
        if spec.day_jitter:
            d += timedelta(days=int(rng.integers(-spec.day_jitter, spec.day_jitter + 1)))
        d = _shift_weekend(d)
        if start <= d < end:
            dates.append(d)
    return dates


def _lognormal(rng: np.random.Generator, mean: float, sd: float, size: int) -> np.ndarray:
    """Right-skewed positive draws with the given mean and standard deviation."""
    sigma2 = np.log1p((sd / mean) ** 2)
    mu = np.log(mean) - sigma2 / 2
    return rng.lognormal(mu, np.sqrt(sigma2), size)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def generate(
    persona: Persona, start: date, months: int, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (transactions, ground_truth)."""
    rng = np.random.default_rng(seed)
    end = (pd.Timestamp(start) + pd.DateOffset(months=months)).date()
    rows: list[dict] = []

    # 1. Recurring series
    for spec in persona.recurring:
        for d in _recurring_dates(spec, start, end, rng):
            amount = spec.amount * (1 + rng.normal(0, spec.amount_jitter))
            rows.append(
                {
                    "date": d,
                    "amount": round(amount, 2),
                    "description": _bank_desc(spec.description, rng),
                    "merchant": spec.merchant,
                    "category": spec.category,
                    "is_recurring": True,
                    "series_name": spec.series,
                    "cadence_days": spec.cadence_days,
                }
            )

    # 2. Discretionary spending (and irregular income): Poisson count per day
    n_days = (end - start).days
    for item in persona.discretionary:
        mean, sd = item["mean"], item["sd"]
        sign = 1.0 if mean > 0 else -1.0
        weekend_boost = item.get("weekend_boost", 1.5)
        lo, hi = item.get("min", 1.0), item.get("max", np.inf)
        for i in range(n_days):
            d = start + timedelta(days=i)
            rate = item["per_week"] / 7 * (weekend_boost if d.weekday() >= 5 else 1.0)
            count = rng.poisson(rate)
            if count == 0:
                continue
            for amt in np.clip(_lognormal(rng, abs(mean), sd, count), lo, hi):
                rows.append(
                    {
                        "date": d,
                        "amount": round(sign * float(amt), 2),
                        "description": _bank_desc(item.get("description", item["merchant"]), rng),
                        "merchant": item["merchant"],
                        "category": item.get("category", "discretionary"),
                        "is_recurring": False,
                        "series_name": None,
                        "cadence_days": None,
                    }
                )

    # 3. One-off surprises on random dates
    for item in persona.one_offs:
        d = start + timedelta(days=int(rng.integers(0, n_days)))
        rows.append(
            {
                "date": d,
                "amount": round(item["amount"], 2),
                "description": _bank_desc(item["description"], rng),
                "merchant": item["merchant"],
                "category": item.get("category", "one_off"),
                "is_recurring": False,
                "series_name": None,
                "cadence_days": None,
            }
        )

    # 4. Stable order and IDs
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "merchant", "amount"], kind="mergesort").reset_index(drop=True)
    df["id"] = [f"{persona.name}-{i:05d}" for i in range(len(df))]
    df["account_id"] = f"{persona.name}-acc"

    transactions = df[["id", "account_id", "date", "amount", "description", "merchant", "category"]]
    truth = df[["id", "is_recurring", "series_name", "cadence_days"]].rename(
        columns={"id": "transaction_id"}
    )
    truth = truth.astype({"cadence_days": "Int64"})
    return transactions.copy(), truth.copy()


# ---------------------------------------------------------------------------
# Persona definitions (see plan/phase-01-synthetic-data.md, Step 2)
# ---------------------------------------------------------------------------

_R = REFERENCE_START  # 2025-01-01, a Wednesday

PERSONAS: dict[str, Persona] = {
    # Weekly rent vs fortnightly pay.
    "student": Persona(
        name="student",
        opening_balance=800.0,
        recurring=[
            RecurringSpec(
                "Cafe Payroll",
                "CAFE CO LTD WAGES {num}",
                675.0,
                14,
                _R + timedelta(1),
                amount_jitter=0.06,
                category="income",
                series_name="pay",
            ),
            RecurringSpec(
                "Tenancy Services",
                "RENT FLAT 3 REF {num}",
                -230.0,
                7,
                _R,
                category="rent",
                series_name="rent",
            ),
            RecurringSpec(
                "Spotify",
                "SPOTIFY P{ref} STOCKHOLM",
                -13.99,
                30,
                date(2025, 1, 9),
                category="subscriptions",
            ),
            RecurringSpec(
                "One NZ",
                "ONE NZ PREPAY {ref} AUCKLAND",
                -30.0,
                30,
                date(2025, 1, 20),
                category="phone",
            ),
        ],
        discretionary=[
            {
                "merchant": "Countdown",
                "description": "COUNTDOWN {num} AUCKLAND",
                "mean": -40,
                "sd": 15,
                "per_week": 2,
                "category": "groceries",
            },
            {
                "merchant": "Uber Eats",
                "description": "UBER *EATS {ref} AUCKLAND",
                "mean": -18,
                "sd": 7,
                "per_week": 0.7,
                "category": "takeaways",
            },
        ],
        one_offs=[
            {
                "merchant": "Warehouse Stationery",
                "description": "WAREHOUSE STATIONERY {num}",
                "amount": -120.0,
            },
            {
                "merchant": "IRD",
                "description": "IRD TAX REFUND {ref}",
                "amount": 80.0,
                "category": "refund",
            },
        ],
    ),
    # The "easy" baseline: monthly salary, calendar-monthly bills, comfortable buffer.
    "salaried": Persona(
        name="salaried",
        opening_balance=6000.0,
        recurring=[
            RecurringSpec(
                "Acme Corp Payroll",
                "ACME CORP LTD SALARY {num}",
                5200.0,
                30,
                date(2025, 1, 15),
                category="income",
                series_name="salary",
            ),
            RecurringSpec(
                "ASB Home Loan",
                "ASB HOME LOAN {ref} REPAYMENT",
                -2400.0,
                30,
                date(2025, 1, 1),
                category="mortgage",
                series_name="mortgage",
            ),
            RecurringSpec(
                "Mercury Energy",
                "MERCURY ENERGY DD {ref}",
                -180.0,
                30,
                date(2025, 1, 22),
                amount_jitter=0.15,
                category="utilities",
                series_name="power",
            ),
            RecurringSpec(
                "2degrees Broadband",
                "2DEGREES BROADBAND {ref}",
                -85.0,
                30,
                date(2025, 1, 8),
                category="utilities",
                series_name="internet",
            ),
            RecurringSpec(
                "CityFitness",
                "CITYFITNESS {num} WELLINGTON",
                -15.0,
                7,
                _R + timedelta(5),
                category="health",
                series_name="gym",
            ),
            RecurringSpec(
                "Netflix",
                "NETFLIX.COM {ref} LOS GATOS",
                -22.99,
                30,
                date(2025, 1, 12),
                category="subscriptions",
                series_name="netflix",
            ),
        ],
        discretionary=[
            {
                "merchant": "Countdown",
                "description": "COUNTDOWN {num} WELLINGTON",
                "mean": -110,
                "sd": 35,
                "per_week": 2,
                "category": "groceries",
            },
            {
                "merchant": "Z Energy",
                "description": "Z ENERGY {num} WELLINGTON",
                "mean": -75,
                "sd": 15,
                "per_week": 0.8,
                "category": "transport",
                "min": 20,
            },
            {
                "merchant": "Cafe",
                "description": "MEJIRO CAFE {num} WELLINGTON",
                "mean": -12,
                "sd": 5,
                "per_week": 3,
                "category": "eating_out",
            },
            {
                "merchant": "Restaurant",
                "description": "SP* LOGAN BROWN {ref}",
                "mean": -85,
                "sd": 40,
                "per_week": 0.6,
                "category": "eating_out",
            },
        ],
        one_offs=[
            {"merchant": "Mitre 10", "description": "MITRE 10 MEGA {num}", "amount": -340.0},
            {
                "merchant": "Airbnb",
                "description": "AIRBNB * {ref} SAN FRANCISCO",
                "amount": -620.0,
                "category": "travel",
            },
            {
                "merchant": "Noel Leeming",
                "description": "NOEL LEEMING REFUND {num}",
                "amount": 80.0,
                "category": "refund",
            },
        ],
    ),
    # Hard mode: no payday. Income arrives as 1-4 irregular payouts a week.
    "gig": Persona(
        name="gig",
        opening_balance=2000.0,
        recurring=[
            RecurringSpec(
                "Tenancy Services",
                "RENT 14B ROOM REF {num}",
                -350.0,
                7,
                _R + timedelta(1),
                category="rent",
                series_name="rent",
            ),
            RecurringSpec(
                "Skinny Mobile",
                "SKINNY MOBILE {ref}",
                -40.0,
                30,
                date(2025, 1, 5),
                category="phone",
                series_name="phone",
            ),
        ],
        discretionary=[
            # Irregular income is modelled as discretionary: it is NOT recurring in the
            # ground truth, which is exactly what makes this persona hard.
            {
                "merchant": "Uber",
                "description": "UBER BV PAYOUT {ref}",
                "mean": 230,
                "sd": 100,
                "per_week": 3.0,
                "min": 50,
                "max": 400,
                "weekend_boost": 1.0,
                "category": "income",
            },
            {
                "merchant": "Pak'nSave",
                "description": "PAK N SAVE {num} AUCKLAND",
                "mean": -65,
                "sd": 30,
                "per_week": 1.5,
                "category": "groceries",
            },
            {
                "merchant": "Gull",
                "description": "GULL {num} AUCKLAND",
                "mean": -55,
                "sd": 20,
                "per_week": 1.2,
                "category": "transport",
                "min": 15,
            },
            {
                "merchant": "McDonald's",
                "description": "MCDONALDS {num} AUCKLAND",
                "mean": -16,
                "sd": 6,
                "per_week": 1.5,
                "category": "takeaways",
            },
        ],
        one_offs=[
            {
                "merchant": "Repco",
                "description": "REPCO {num} AUCKLAND",
                "amount": -600.0,
                "category": "car",
            },
            {
                "merchant": "Trade Me",
                "description": "TRADE ME SALE {ref}",
                "amount": 150.0,
                "category": "income",
            },
        ],
    ),
    # Tight budget: fortnightly pay, big weekly rent, car loan the day before payday.
    "edge": Persona(
        name="edge",
        opening_balance=300.0,
        recurring=[
            RecurringSpec(
                "Warehouse Payroll",
                "THE WAREHOUSE LTD WAGES {num}",
                1400.0,
                14,
                _R + timedelta(1),
                category="income",
                series_name="pay",
            ),
            RecurringSpec(
                "Barfoot Rentals",
                "BARFOOT RENT {ref}",
                -500.0,
                7,
                _R,
                category="rent",
                series_name="rent",
            ),
            RecurringSpec(
                "UDC Finance",
                "UDC FINANCE LOAN {ref}",
                -150.0,
                14,
                _R,
                category="loan",
                series_name="car_loan",
            ),
            RecurringSpec(
                "AA Insurance",
                "AA INSURANCE DD {ref}",
                -95.0,
                30,
                date(2025, 1, 14),
                category="insurance",
                series_name="insurance",
            ),
        ],
        discretionary=[
            {
                "merchant": "Pak'nSave",
                "description": "PAK N SAVE {num} HAMILTON",
                "mean": -40,
                "sd": 15,
                "per_week": 1.5,
                "category": "groceries",
            },
            {
                "merchant": "BP",
                "description": "BP CONNECT {num} HAMILTON",
                "mean": -40,
                "sd": 12,
                "per_week": 0.6,
                "category": "transport",
                "min": 15,
            },
            {
                "merchant": "KFC",
                "description": "KFC {num} HAMILTON",
                "mean": -15,
                "sd": 6,
                "per_week": 0.5,
                "category": "takeaways",
            },
        ],
        one_offs=[
            {
                "merchant": "Vet",
                "description": "HAMILTON VET CLINIC {num}",
                "amount": -280.0,
                "category": "pets",
            },
            {
                "merchant": "Countdown",
                "description": "COUNTDOWN REFUND {num}",
                "amount": 35.0,
                "category": "refund",
            },
        ],
    ),
}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--persona", choices=PERSONAS.keys(), required=True)
    p.add_argument("--months", type=int, default=9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start", type=date.fromisoformat, default=REFERENCE_START)
    p.add_argument("--out", default="data/synthetic")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tx, truth = generate(PERSONAS[args.persona], args.start, args.months, args.seed)
    tx_path = out / f"{args.persona}_transactions.csv"
    truth_path = out / f"{args.persona}_truth.csv"
    tx.to_csv(tx_path, index=False)
    truth.to_csv(truth_path, index=False)
    print(f"Wrote {len(tx)} transactions to {tx_path}")
    n_recurring = int(truth.is_recurring.sum())
    print(f"Wrote {len(truth)} truth rows to {truth_path} ({n_recurring} recurring)")

