import time
import schedule
from datetime import date
from core.logging import get_logger, setup_logging
from analytics.live.market_sync import MarketSynchronizer
from analytics.live.performance_tracker import PerformanceTracker
from analytics.live.paper_trader import PaperTrader
from analytics.live.report_generator import ReportGenerator

logger = get_logger(__name__)

def daily_market_sync():
    logger.info("=== SCHEDULER: Initiating Daily Market Sync ===")
    sync = MarketSynchronizer()
    success = sync.run_daily_sync()
    if success:
        logger.info("=== SCHEDULER: Initiating Daily Performance MTM ===")
        tracker = PerformanceTracker()
        tracker.run_daily_mtm()
        
        # Check if today is the 1st of the month (or first trading day)
        # Simplified: We just rebalance if today is the 1st of the month
        if date.today().day == 1:
            logger.info("=== SCHEDULER: 1st of Month Detected. Initiating Monthly Rebalance ===")
            trader = PaperTrader()
            trader.run_rebalance()
            
            logger.info("=== SCHEDULER: Generating Monthly Research Report ===")
            report = ReportGenerator()
            report.generate_monthly_report()
            
def start_scheduler():
    setup_logging(log_level="INFO")
    
    # Centralized Database Initialization
    from config.settings import get_settings
    from database.connection import init_db
    settings = get_settings()
    init_db(settings.database_url, echo=False)
    
    logger.info("Starting Atlas Autonomous Scheduler...")
    logger.info("Scheduled Daily Market Sync at 18:00 IST (Post-Market Close).")
    
    # Schedule daily at 6:00 PM (18:00) server time
    schedule.every().day.at("18:00").do(daily_market_sync)
    
    # Keep running forever
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    start_scheduler()
