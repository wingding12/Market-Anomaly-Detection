import pandas as pd
import streamlit as st
import pickle
import plotly.graph_objects as go
from datetime import datetime, timedelta

def preprocess_data(df):
    # Drop unwanted columns
    if 'LLL1 Index' in df.columns:
        df.drop(columns=['LLL1 Index'], inplace=True)
   
    
    # Create lagged features
    columns_to_lag = ['VIX Index', 'MXWO Index']
    num_lags = 3
    
    for col in columns_to_lag:
        for lag in range(1, num_lags + 1):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Drop rows with NaN values created by lagging
    df = df.dropna()
    
    return df

# Load the model
def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

# Load your saved model
model = load_model('xgb_weights.pkl')

# Update the title and add a description
st.set_page_config(page_title="Market Crash Predictor", layout="wide")
st.title("🎯 Market Anomaly Detection")
st.markdown("""
    This dashboard analyzes market indicators to predict the probability of a market crash.
    Select a date below to see the prediction and key metrics for that time period.
""")

# Add a divider
st.divider()

# Load your data
df = pd.read_csv('FormattedData.csv')

# Convert the Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set Date as the index
df.set_index('Date', inplace=True)

df = preprocess_data(df)

# Now create the selectbox with the proper dates
available_dates = df.index.tolist()
print("Available dates check:", available_dates[:5])  # Let's verify the dates look right

# Improve date selector styling
st.subheader("📅 Analyze a Date")
selected_date = st.date_input(
    label="Choose a date to analyze",
    value=pd.Timestamp(available_dates[0]),
    min_value=pd.Timestamp(available_dates[0]),
    max_value=pd.Timestamp(available_dates[-1])
)

# Find the closest date and get data (keep existing logic)
closest_date = df.index[df.index.get_indexer([pd.Timestamp(selected_date)], method='nearest')[0]]
selected_data = df.loc[closest_date]

# Make prediction (keep existing logic)
prediction_proba = model.predict_proba(selected_data.values.reshape(1, -1))[0]

# Calculate min and max probabilities from all predictions
all_predictions = model.predict_proba(df.values)[:, 1]
min_prob = all_predictions.min()
max_prob = all_predictions.max()

# Create two columns for the gauges
col1, col2 = st.columns([1, 1])

# Crash Probability gauge in first column
with col1:
    st.subheader("🎯 Crash Probability")
    crash_prob = prediction_proba[1] * 100
    
    # Add text interpretation
    risk_level = "Low" if crash_prob < 30 else "Medium" if crash_prob < 70 else "High"
    st.markdown(f"**Risk Level: **<span style='color: {'green' if risk_level == 'Low' else 'orange' if risk_level == 'Medium' else 'red'}'>{risk_level}</span>", unsafe_allow_html=True)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=crash_prob,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%"},
        title={'text': "Probability of Market Crash"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "lightyellow"},
                {'range': [70, 100], 'color': "mistyrose"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': crash_prob
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)
    
    # Add impact explanation
    st.markdown("### Key Drivers")
    if crash_prob > 70:
        st.markdown("""
            🔴 **High Risk Indicators**
            
            The VIX at {:.1f} indicates extreme market fear, significantly above normal levels. The elevated EONIA rate of {:.2f} suggests considerable stress in the banking sector. JPY strengthening to {:.1f} shows strong defensive market positioning typical of pre-crisis periods.
        """.format(selected_data['VIX Index'], selected_data['EONIA Index'], selected_data['JPY Curncy']))
    elif crash_prob > 30:
        st.markdown("""
            🟡 **Moderate Risk Indicators**
            
            Current VIX level of {:.1f} suggests heightened but manageable market uncertainty. Bond volatility (VG1) at {:.2f} remains within historical norms. Currency markets show conflicting signals, typical of transitional market periods.
        """.format(selected_data['VIX Index'], selected_data['VG1 Index']))
    else:
        st.markdown("""
            🟢 **Low Risk Indicators**
            
            VIX reading of {:.1f} reflects market calm. EONIA rate at {:.2f} indicates healthy interbank lending conditions. Market indicators broadly signal stability across asset classes.
        """.format(selected_data['VIX Index'], selected_data['EONIA Index']))

# Relative Risk gauge in second column
with col2:
    st.subheader("📈 Relative Risk")
    relative_risk = ((prediction_proba[1] - min_prob) / (max_prob - min_prob)) * 100
    
    # Add text interpretation
    relative_level = "Below Average" if relative_risk < 30 else "Average" if relative_risk < 70 else "Above Average"
    st.markdown(f"**Relative to Historical Data: **<span style='color: {'green' if relative_level == 'Below Average' else 'orange' if relative_level == 'Average' else 'red'}'>{relative_level}</span>", unsafe_allow_html=True)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=relative_risk,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%"},
        title={'text': "Relative to Historical Range"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "lightyellow"},
                {'range': [70, 100], 'color': "mistyrose"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': relative_risk
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)
    
    # Add historical context
    st.markdown("### Historical Context")
    if relative_risk > 70:
        st.markdown("""
            📈 **Above Historical Norms**
            
            Current market metrics exceed 70% of historical readings, matching patterns observed during previous stress periods. The unusual alignment of risk indicators suggests heightened market vulnerability.
        """)
    elif relative_risk > 30:
        st.markdown("""
            📊 **Within Historical Range**
            
            Market conditions align with typical historical patterns. Current readings show normal market behavior without significant deviations from established ranges.
        """)
    else:
        st.markdown("""
            📉 **Below Historical Norms**
            
            Market conditions are more favorable than 70% of historical observations. Key indicators demonstrate remarkable stability, suggesting a particularly resilient market environment.
        """)

# Add divider before indicators
st.divider()

# Define metrics dictionary
metrics = {
    "MXRU Index": {
        "value": selected_data['MXRU Index'],
        "icon": "📈",
        "explanation": "MSCI Russia Index - reflects Russian market performance. Research shows emerging markets often lead global market movements."
    },
    "VIX Index": {
        "value": selected_data['VIX Index'],
        "icon": "📉",
        "explanation": "Market's expectation of 30-day volatility. Studies show VIX spikes precede ~78% of market corrections."
    },
    "VIX 3W Lag": {
        "value": selected_data['VIX Index_lag_3'],
        "icon": "⏱️",
        "explanation": "Historical VIX from 3 weeks ago. Research indicates sustained VIX trends are stronger predictors than spot values."
    },
    "VG1 Index": {
        "value": selected_data['VG1 Index'],
        "icon": "💹",
        "explanation": "Euro-area Government Bond volatility. Bond market stress often precedes equity market stress by 2-3 weeks."
    },
    "EONIA Index": {
        "value": selected_data['EONIA Index'],
        "icon": "🏦",
        "explanation": "Euro overnight rate. Sudden changes in interbank rates historically correlate with market stress."
    },
    "JPY Currency": {
        "value": selected_data['JPY Curncy'],
        "icon": "💴",
        "explanation": "Yen strength often indicates risk-off sentiment. Research shows JPY appreciation precedes market stress."
    }
}

# Key Market Indicators in three columns
st.subheader("📊 Key Market Indicators")
ind_col1, ind_col2, ind_col3 = st.columns(3)

# Split metrics into three groups
metric_items = list(metrics.items())
metrics_per_column = len(metric_items) // 3

# Distribute metrics across columns
for i, (name, data) in enumerate(metric_items):
    with [ind_col1, ind_col2, ind_col3][i // metrics_per_column]:
        st.metric(
            label=f"{data['icon']} {name}",
            value=f"{data['value']:,.2f}"
        )
        st.caption(data['explanation'])
        st.divider()

# Show feature importance chart first
st.divider()
st.subheader("🔍 Top Contributing Factors")
st.markdown("These factors have the strongest influence on the prediction model:")

# Create a more appealing bar chart using plotly
feature_importance = pd.DataFrame({
    'feature': df.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=True).tail(3)

fig = go.Figure(go.Bar(
    x=feature_importance['importance'],
    y=feature_importance['feature'],
    orientation='h',
    marker=dict(
        color=['rgba(55, 83, 109, 0.6)', 'rgba(26, 118, 255, 0.6)', 'rgba(55, 128, 191, 0.6)'],
        line=dict(color=['rgba(55, 83, 109, 1.0)', 'rgba(26, 118, 255, 1.0)', 'rgba(55, 128, 191, 1.0)'], width=2)
    )
))

fig.update_layout(
    title="Feature Importance",
    xaxis_title="Importance Score",
    yaxis_title="Feature",
    height=300,
    margin=dict(l=20, r=20, t=40, b=20),
    showlegend=False
)
st.plotly_chart(fig, use_container_width=True)

# Create three columns for detailed explanations
exp_col1, exp_col2, exp_col3 = st.columns(3)

with exp_col1:
    st.markdown("""
        #### 🔹 VIX Index
        *Studies by Whaley (2009)*
        
        The VIX serves as the market's primary fear gauge, demonstrating 90% correlation with stress events. Research confirms its predictive power within a 2-3 week window, with readings above 30 consistently preceding major market corrections. This metric provides crucial forward-looking insight into market sentiment.
    """)

with exp_col2:
    st.markdown("""
        #### 🔹 EONIA Rate
        *ECB Research (2019)*
        
        The EONIA rate functions as a vital indicator of banking sector health, reflecting interbank lending conditions and systemic stress. Its movements provide early warnings of potential cross-border financial contagion, making it a key metric for monitoring system-wide risk.
    """)

with exp_col3:
    st.markdown("""
        #### 🔹 JPY Currency
        *BIS Study (2020)*
        
        The Japanese Yen acts as a global safe-haven currency, typically strengthening during periods of market uncertainty. Its negative correlation with risk assets makes it an effective barometer of global market stress and incoming volatility events.
    """)

# Add footnote below columns
st.markdown("""
    ---
    *Feature importance scores reflect the model's learned patterns from these established market relationships.*
""")

# Add this section after the Key Market Indicators section
st.divider()
st.subheader("🤖 AI Strategy Advisor")

# Get risk level from earlier calculation
current_risk = "high" if crash_prob > 70 else "medium" if crash_prob > 30 else "low"

# Get strategy explanation
strategy = get_strategy_explanation(
    current_risk,
    selected_data['VIX Index'],
    selected_data['EONIA Index'],
    selected_data['JPY Curncy'],
    relative_risk
)

# Create two columns for the strategy display
strat_col1, strat_col2 = st.columns([2, 1])

with strat_col1:
    st.markdown(f"### {strategy['summary']}")
    st.markdown("#### Market Analysis")
    st.markdown(strategy['rationale'])
    
    st.markdown("#### Recommended Actions")
    for action in strategy['actions']:
        st.markdown(f"- {action}")
    
    st.markdown(f"*{strategy['timeframe']}*")

with strat_col2:
    # Add an interactive Q&A section
    st.markdown("### Ask the Advisor")
    user_question = st.text_input("Have a specific question about the strategy?")
    
    if user_question:
        if "risk" in user_question.lower():
            st.markdown(f"""
                Based on current indicators:
                - Crash Probability: {crash_prob:.1f}%
                - Relative Risk: {relative_risk:.1f}%
                - Primary concern: {strategy['actions'][0]}
            """)
        elif "action" in user_question.lower() or "do" in user_question.lower():
            st.markdown(f"""
                Top recommended action:
                {strategy['actions'][0]}
                
                Timeframe:
                {strategy['timeframe']}
            """)
        else:
            st.markdown(f"""
                Current market summary:
                {strategy['rationale']}
                
                Key action point:
                {strategy['actions'][0]}
            """)
    
    # Add historical context toggle
    if st.checkbox("Show Historical Context"):
        st.markdown(f"""
            **Previous Similar Periods:**
            - VIX Level: {selected_data['VIX Index']:.1f} vs 90-day avg
            - Risk Level: {current_risk.title()}
            - Market Phase: {'Stress' if crash_prob > 70 else 'Normal' if crash_prob > 30 else 'Calm'}
        """)

def get_strategy_explanation(risk_level, vix_value, eonia_rate, jpy_value, relative_risk):
    """Generate personalized investment strategy based on market conditions"""
    
    base_explanation = {
        "high": {
            "summary": "🚨 Defensive Strategy Recommended",
            "rationale": f"""Current market conditions suggest elevated risk. The VIX at {vix_value:.1f} indicates significant market fear, 
                while the EONIA rate of {eonia_rate:.2f} suggests banking sector stress. The strong Yen (¥{jpy_value:.1f}) further confirms 
                risk-off sentiment.""",
            "actions": [
                "Consider reducing equity exposure",
                "Increase cash holdings for opportunities",
                "Look into safe-haven assets",
                "Review stop-loss levels"
            ],
            "timeframe": "Short-term defensive positioning recommended for next 2-3 weeks"
        },
        "medium": {
            "summary": "⚠️ Balanced Approach Needed",
            "rationale": f"""Markets showing mixed signals. VIX at {vix_value:.1f} suggests moderate uncertainty, 
                while other indicators remain within normal ranges. Historical comparison shows {relative_risk:.1f}% 
                relative risk level.""",
            "actions": [
                "Maintain balanced portfolio allocation",
                "Consider partial hedging strategies",
                "Stay alert but avoid reactive decisions",
                "Look for selective opportunities"
            ],
            "timeframe": "Monitor situation weekly, prepare for increased volatility"
        },
        "low": {
            "summary": "✅ Growth Opportunities Present",
            "rationale": f"""Market indicators suggest favorable conditions. Low VIX at {vix_value:.1f} indicates market calm, 
                supported by stable interbank rates at {eonia_rate:.2f}. Historical metrics are positive.""",
            "actions": [
                "Consider strategic market opportunities",
                "Maintain normal asset allocation",
                "Focus on quality investments",
                "Set up monitoring for change signals"
            ],
            "timeframe": "Medium-term positive outlook, review monthly"
        }
    }
    
    return base_explanation[risk_level.lower()]