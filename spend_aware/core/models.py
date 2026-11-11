from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    id: str
    account_id: str
    date: date
    amount: Decimal = Field(description="Negative = money out, positive = money in")
    description: str
    merchant: str | None = None
    category: str | None = None

