"""
repositories/stock_repository.py
=================================
Data access layer for the `stocks` table.

Purpose:
    CRUD operations for the master stock registry.
    All callers must use this class — no direct ORM queries outside repositories.

Dependencies:
    database.connection.get_db
    database.models.Stock  (added in Sprint 2)

Inputs:
    Symbol strings, sector names, exchange codes.

Outputs:
    Stock ORM instances or lists thereof.

Failure Scenarios:
    - Duplicate symbol insert → IntegrityError (unique constraint on symbol)
    - DB unreachable → DatabaseConnectionError from connection layer
"""

from __future__ import annotations

# TODO (Sprint 2): Implement when database/models.py is created.
#
# class StockRepository:
#     def __init__(self, db: DatabaseManager) -> None: ...
#     def get_by_symbol(self, symbol: str) -> Stock | None: ...
#     def get_all_active(self) -> list[Stock]: ...
#     def upsert(self, symbol: str, name: str, sector: str) -> Stock: ...
#     def deactivate(self, symbol: str) -> None: ...
