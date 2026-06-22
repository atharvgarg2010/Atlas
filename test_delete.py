from datetime import date
from database.connection import init_db, get_db
from database.models.factors import FactorRanking
from sqlalchemy import select, delete
from config.settings import get_settings

settings = get_settings()
init_db(settings.database_url, echo=True)
db = get_db()

ranking_date = date.today()

with db.session() as s:
    count_before = s.scalar(select(FactorRanking).where(FactorRanking.ranking_date == ranking_date))
    print("Count before delete:", count_before)
    
    s.execute(delete(FactorRanking).where(FactorRanking.ranking_date == ranking_date))
    
    count_after = s.scalar(select(FactorRanking).where(FactorRanking.ranking_date == ranking_date))
    print("Count after delete:", count_after)
