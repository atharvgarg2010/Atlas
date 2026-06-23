import time
from core.logging import get_logger
from config.settings import get_settings
from data.data_manager import DataManager

logger = get_logger(__name__)

class MarketSynchronizer:
    def __init__(self):
        self.dm = DataManager()
        
    def run_daily_sync(self):
        logger.info("Starting Autonomous Daily Market Sync...")
        start_time = time.time()
        
        with self.dm.db.session() as s:
            from database.models.market_data import SymbolMetadata
            from sqlalchemy import select
            # We sync all symbols that are actively tracked in the database
            active_symbols = s.scalars(select(SymbolMetadata.symbol).where(SymbolMetadata.status == "ACTIVE")).all()
            
        if not active_symbols:
            logger.warning("No ACTIVE symbols found in database to sync.")
            return False
            
        logger.info(f"Initiating bulk sync for {len(active_symbols)} symbols...")
        results = self.dm.sync_universe(active_symbols, max_workers=5)
        
        success_count = sum(1 for v in results.values() if v)
        elapsed = time.time() - start_time
        
        logger.info(f"Daily Market Sync Complete in {elapsed:.2f}s. Successful: {success_count}/{len(active_symbols)}")
        return success_count > 0

if __name__ == "__main__":
    from core.logging import setup_logging
    setup_logging(log_level="INFO")
    sync = MarketSynchronizer()
    sync.run_daily_sync()
