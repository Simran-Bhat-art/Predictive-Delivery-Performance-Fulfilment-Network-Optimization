# Delivery Risk & Fulfilment Network Optimizer

A decision-support dashboard for e-commerce delivery. It predicts which shipments will arrive late, converts that risk into a cost, and uses optimization models to test seller allocation and warehouse locations. Built on the public Olist Brazilian E-Commerce dataset (2016–2018).

Project for Digital Supply Chain Management and Operations Research.

## What it does

1. **Analyses** where and why deliveries are late (by state, distance and review score).
2. **Predicts** the probability that each shipment is late, using only information known at order time.
3. **Optimizes** two decisions using freight cost plus expected late-delivery penalty:
   - Seller-to-region assignment (transportation LP)
   - Warehouse location (MILP)
4. **Presents** everything in an interactive Streamlit dashboard with what-if controls.

## Dashboard tabs

| Tab | What it shows |
|---|---|
| Overview | KPIs, insight cards, late rate by state (map and bar chart), review score on time vs late |
| Risk Model | Logistic regression vs gradient boosting, feature importance, highest-risk shipments |
| Network Optimizer | Model 1 (seller assignment) and Model 2 (warehouse map), compared with the current network |
| What-if | Total cost against penalty level and warehouse limit |
| Paper results | Tables and 300-dpi figures for the report (ROC/PR, calibration, SHAP, cost comparison) |

Sidebar controls: late-delivery penalty (% of average order value), maximum warehouses, capacity multiplier, warehouse fixed cost.

## Method in brief

- **Unit of analysis:** a shipment (order-seller pair); 97,811 delivered shipments.
- **Label:** late if the actual delivery timestamp is after the estimated delivery timestamp.
- **Split:** chronological, oldest 80% for training and newest 20% for testing.
- **Models:** logistic regression (baseline) and LightGBM, plus SHAP for explanations.
- **Risk-adjusted cost:** `freight + penalty × late probability`, with the penalty set to a share of the average order value (default 25%).
- **Optimization:** PuLP with the CBC solver, at state level.

## Key results (default settings)

- 8.0% of shipments are late; review scores average 4.27 on time and 2.57 when late.
- Late rates are highest in northeastern states (AL, MA, PI, CE, SE).
- Prediction is modest: ROC-AUC 0.710 and PR-AUC 0.107 against a 5.2% base rate in the test period.
- Model 1 cuts freight by about 16.7% and expected late orders by about 11.8%; Model 2 (warehouses in SP and MG) cuts total cost by about 5.1%.
- **Risk-aware and cost-only optimization give almost the same result** (total cost differs by 0.02%), because freight and delivery risk both rise with distance.

## Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
python3 -m pip install -r requirements.txt
```

**Get the data:** download the Brazilian E-Commerce Public Dataset by Olist from Kaggle (https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), unzip it, and place the `olist_*_dataset.csv` files in a folder named `data` next to `app.py`. Without them the app runs on synthetic demo data and shows a warning.

**Run:**

```bash
python3 -m streamlit run app.py
```

The app opens at http://localhost:8501.

### Troubleshooting

- **LightGBM error on macOS (`libomp.dylib`):** run `brew install libomp`. The app falls back to scikit-learn's gradient boosting if LightGBM cannot load.
- **SHAP figure missing:** run `python3 -m pip install shap`.
- **`No module named ...`:** make sure the virtual environment is active and re-run the install command.

## Project structure

```
app.py              # data preparation, models, optimization and dashboard
requirements.txt    # Python dependencies
data/               # Olist CSV files (not included)
```

## Assumptions and limitations

- The late-delivery penalty, capacity multiplier and warehouse opening cost are assumptions; Olist contains no refund or warehouse data.
- Model 1 treats products as interchangeable between sellers, and Model 2 ignores the seller-to-warehouse leg, so savings are optimistic scenario estimates.
- Nodes are Brazilian states with straight-line distances; costs and risk for unobserved routes are estimated from distance.
- The model over-predicts risk in the test period (7.4% predicted against 5.2% observed), and seller history is only approximately leakage-free.
- Data covers 2016–2018 and may not reflect current logistics conditions.

## Tech stack

Python, pandas, NumPy, scikit-learn, LightGBM, SHAP, PuLP (CBC), Plotly, Matplotlib, Streamlit.

## Authors

Vaishnavi Dhundale, Simran Bhat, Arya Pathak, Siddhesh Tandale, Geetanjali Yadav
Department of Artificial Intelligence and Data Science, BRACT's Vishwakarma Institute of Information Technology, Pune, India

## Data source

Olist, "Brazilian E-Commerce Public Dataset by Olist," Kaggle.
