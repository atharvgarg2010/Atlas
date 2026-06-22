from abc import ABC, abstractmethod
import pandas as pd

class BaseFactor(ABC):
    """
    Abstract base class for all factors in the Atlas Factor Engine.
    """
    
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, symbol_data: pd.DataFrame, benchmark_data: pd.DataFrame | None = None) -> float | None:
        """
        Compute the raw factor score for a given symbol.

        Args:
            symbol_data: DataFrame containing historical OHLCV and indicators for the symbol.
                         Assumed to be sorted by date (oldest first).
            benchmark_data: DataFrame containing historical data for the benchmark (e.g., ^NSEI).
                            Only used if the factor requires it (e.g., Relative Strength).

        Returns:
            The raw factor score, or None if data is insufficient.
        """
        pass
