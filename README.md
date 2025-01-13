# Market Anomaly Detection Dashboard 🎯

A Streamlit-powered web application that analyzes financial market indicators to predict market anomalies and potential crash scenarios using machine learning.

## Features

- Real-time market risk assessment
- Historical comparison analysis
- Key market indicator tracking
- Research-backed feature importance explanations
- Interactive data visualization

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:

bash
git clone <repository-url>
cd market-anomaly-detection

2. Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

## Required Files

Ensure these files are in your project root directory:

- `FormattedData.csv`: Historical market data
- `xgb_weights.pkl`: Trained model weights

## Running the Application

1. With your virtual environment activated, run:

```bash
streamlit run app.py
```

2. The dashboard will open automatically in your default browser at `http://localhost:8501`

## Data Sources

The dashboard uses several key market indicators:

- VIX Index (Market Volatility)
- EONIA Rate (European Overnight Index)
- JPY Currency (Japanese Yen)
- MXRU Index (MSCI Russia)
- Bond Volatility Indicators

## Model Information

The application uses a trained XGBoost model to:

- Predict market anomaly probability
- Calculate relative risk levels
- Identify key contributing factors
- Analyze historical patterns

## Troubleshooting

Common issues and solutions:

1. **ModuleNotFoundError**

   ```bash
   pip install -r requirements.txt
   ```

2. **FileNotFoundError**

   - Check that `FormattedData.csv` and `xgb_weights.pkl` are present
   - Verify file paths in `app.py`

3. **Port Conflict**
   ```bash
   streamlit run app.py --server.port 8502
   ```

## Dependencies

Core requirements:

```
streamlit>=1.24.0
pandas>=1.5.0
plotly>=5.13.0
scikit-learn>=1.0.2
xgboost>=1.7.3
```

Full list in `requirements.txt`

## Development

To modify the dashboard:

1. Make changes to `app.py`
2. The app will auto-reload when files are saved
3. Use `st.experimental_rerun()` for manual refresh

## Support

For issues and feature requests:

1. Check existing GitHub issues
2. Create a new issue with:
   - Error message/screenshot
   - Steps to reproduce
   - Expected behavior
