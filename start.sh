#!/bin/bash
# Start Streamlit in the background
streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0 &

# Start the autonomous scheduler in the foreground
python scheduler.py
