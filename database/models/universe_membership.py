from datetime import date
from sqlalchemy import Date, String, Integer
from sqlalchemy.orm import Mapped, mapped_column

from database.connection import Base

class UniverseMembership(Base):
    """
    Tracks historical point-in-time membership of symbols in specific indices.
    Prevents survivorship bias by ensuring we only backtest on symbols that
    were actually in the index on the backtest date.
    """
    __tablename__ = "universe_membership"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String, index=True, nullable=False)
    index_name: Mapped[str] = mapped_column(String, index=True, nullable=False)  # e.g., 'NIFTY100'
    start_date: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, index=True, nullable=True) # Null if currently active

    def __repr__(self) -> str:
        end_str = self.end_date if self.end_date else 'Present'
        return f"<UniverseMembership {self.symbol} in {self.index_name} ({self.start_date} to {end_str})>"
