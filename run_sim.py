from config.settings import get_settings
from database.connection import init_db
from analytics.live.paper_trader import PaperTrader
from analytics.live.performance_tracker import PerformanceTracker
from analytics.live.report_generator import ReportGenerator
from core.logging import setup_logging

def run_simulation():
    setup_logging(log_level="INFO")
    
    settings = get_settings()
    init_db(settings.database_url, echo=False)
    
    print("--- 1. Running Paper Trader ---")
    trader = PaperTrader()
    trader.run_rebalance()
    
    print("--- 2. Running Performance Tracker ---")
    tracker = PerformanceTracker()
    tracker.run_daily_mtm()
    
    print("--- 3. Running Report Generator ---")
    report = ReportGenerator()
    report.generate_monthly_report()
    report.generate_decision_journal()
    
    print("--- Simulation Complete ---")

if __name__ == "__main__":
    run_simulation()
