# spend-aware

## The Problem
Most budgeting apps tell you what you already spent. Very few tell you what's about to happen. 

People get hit with overdraft fees, failed payments and stress because rent, bills and subscriptions bunch up before payday. THe project answers one question everyday:
> "Will I make it to payday - and if not, why?"

## Features
| Feature | Description |
| :--- | :---: | 
| 30-day balance forecast | Most likely balance per day, with an uncertainty range  | 
| Overdraft risk alert | Probability of dropping below $0, and which payments cause it | 
| Safe-to-spend today | How much you can spend without putting upcoming bills at risk |
| Upcoming bills calendar | Recurring payments detected automatically, with next expected date |
| What-if simulator | "If I buy this $300 thing today, what happens?" |
| Synthetic data personas | Realistic fake financial profiles for demo and testing |

## Tech Stack
* **Language:** Python 3.12
* **Data & modelling:** pandas, NumPy,scikit-learn, statsmodels
* **Backend:** FastAPI
* **Database:** SQLite (dev) &rarr; PostgreSQL (prod)
* **Frontend:** Streamlit
* **Testing/CI:** pytest, GitHub Actions
* **Deployment:** Docker + Render/Fly.io