import streamlit as st
import json
import pandas as pd
import plotly.express as px
from pathlib import Path

# --- Configuration ---
st.set_page_config(page_title="Atlas Live Dashboard", page_icon="📈", layout="wide")
st.title("Atlas Autonomous Live Engine")

LIVE_DIR = Path(__file__).parent.parent / "research" / "live"
ACTIVE_PORT_FILE = LIVE_DIR / "portfolios" / "active_portfolio.json"
HISTORY_FILE = LIVE_DIR / "performance_history.json"
REPORTS_DIR = LIVE_DIR / "reports"
JOURNAL_DIR = LIVE_DIR / "journal"
TRADES_FILE = LIVE_DIR / "trades" / "trade_ledger.json"

# --- Data Loaders ---
@st.cache_data(ttl=3600)
def load_json_file(file_path):
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def load_latest_md(directory, prefix=""):
    if directory.exists():
        files = sorted(list(directory.glob("*.md")), reverse=True)
        if prefix:
            files = [f for f in files if f.name.startswith(prefix)]
        if files:
            with open(files[0], "r", encoding="utf-8") as f:
                return f.read()
    return "No report generated yet."

# --- Alerts & Warnings ---
def display_health_warnings():
    dq_report = load_latest_md(REPORTS_DIR, "Data_Quality_Report")
    pi_report = load_latest_md(REPORTS_DIR, "Portfolio_Integrity_Report")
    
    if "FAIL ❌" in dq_report:
        st.error("⚠️ CRITICAL: Data Quality Checks Failed. Check Data Quality Status tab.")
    if "FAIL ❌" in pi_report:
        st.error("⚠️ CRITICAL: Portfolio Integrity Checks Failed. Check Portfolio Health tab.")

display_health_warnings()

# --- Layout ---
tabs = st.tabs([
    "Overview", 
    "Portfolio Live", 
    "Performance Track", 
    "Trade History", 
    "Decision Journal",
    "Portfolio Health",
    "Data Quality Status",
    "Research Reports"
])

port = load_json_file(ACTIVE_PORT_FILE)
hist = load_json_file(HISTORY_FILE)
ledger = load_json_file(TRADES_FILE)

with tabs[0]:  # Overview
    st.header("System Overview")
    if hist and hist.get("daily_snapshots"):
        latest_snap = hist["daily_snapshots"][-1]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Portfolio Value", f"₹ {latest_snap['portfolio_value']:,.2f}", 
                      f"{(latest_snap['portfolio_value']/hist['initial_capital'] - 1)*100:+.2f}%")
        with col2:
            nifty_ret = ((latest_snap.get('nifty_close', 1) / hist.get('initial_nifty', 1)) - 1) * 100
            st.metric("NIFTY Benchmark", f"₹ {latest_snap.get('nifty_close', 0):,.2f}", f"{nifty_ret:+.2f}%")
        with col3:
            excess = ((latest_snap['portfolio_value']/hist['initial_capital'] - 1)*100) - nifty_ret
            st.metric("Excess Return (Alpha)", f"{excess:+.2f}%")
            
        metrics = hist.get("current_metrics", {})
        if metrics:
            st.markdown("### Key Performance Indicators")
            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            m_col1.metric("CAGR", f"{metrics.get('cagr', 0):.2f}%")
            m_col2.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
            m_col3.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")
            m_col4.metric("Win Rate", f"{metrics.get('win_rate', 0):.1f}%")
            m_col5.metric("Rolling 1Y Return", f"{metrics.get('rolling_return_1y', 0):.2f}%")
            
    else:
        st.warning("Performance history not available yet. Waiting for first Daily MTM.")

with tabs[1]:  # Portfolio Live
    st.header("Live Portfolio Holdings")
    if port:
        st.write(f"**Rebalance Date:** {port.get('date')}")
        st.write(f"**Optimization Scheme:** {port.get('weighting_scheme')}")
        st.write(f"**Cash Balance:** ₹ {port.get('cash_balance', 0):,.2f}")
        
        df_pos = pd.DataFrame(port['positions'])
        if not df_pos.empty:
            df_pos['weight_%'] = df_pos['weight'] * 100
            st.dataframe(df_pos[['symbol', 'shares', 'entry_price', 'weight_%', 'factor_score']], use_container_width=True)
            
            fig = px.pie(df_pos, values='weight', names='symbol', title='Target Capital Allocation')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Active portfolio not generated yet.")

with tabs[2]:  # Performance Track
    st.header("Equity Curve")
    if hist and hist.get("daily_snapshots"):
        df_hist = pd.DataFrame(hist['daily_snapshots'])
        df_hist['date'] = pd.to_datetime(df_hist['date'])
        
        df_hist['Atlas'] = (df_hist['portfolio_value'] / hist['initial_capital']) * 100
        if 'nifty_close' in df_hist.columns and hist.get('initial_nifty'):
            df_hist['NIFTY'] = (df_hist['nifty_close'] / hist['initial_nifty']) * 100
            plot_df = df_hist[['date', 'Atlas', 'NIFTY']].melt(id_vars='date', var_name='Asset', value_name='Indexed Value')
        else:
            plot_df = df_hist[['date', 'Atlas']].melt(id_vars='date', var_name='Asset', value_name='Indexed Value')
            
        fig2 = px.line(plot_df, x='date', y='Indexed Value', color='Asset', title='Indexed Performance (Base=100)')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Not enough history to plot equity curve.")

with tabs[3]:  # Trade History
    st.header("Trade Ledger")
    if ledger:
        df_ledger = pd.DataFrame(ledger)
        df_ledger['timestamp'] = pd.to_datetime(df_ledger['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        # Style dataframe slightly
        st.dataframe(df_ledger, use_container_width=True)
    else:
        st.info("No trades recorded yet.")

with tabs[4]:  # Decision Journal
    st.markdown(load_latest_md(JOURNAL_DIR, "Decision_Journal_"))

with tabs[5]:  # Portfolio Health
    st.markdown(load_latest_md(REPORTS_DIR, "Portfolio_Integrity_Report"))

with tabs[6]:  # Data Quality Status
    st.markdown(load_latest_md(REPORTS_DIR, "Data_Quality_Report"))

with tabs[7]:  # Research Reports
    st.markdown(load_latest_md(REPORTS_DIR, "20")) # match 202X reports
