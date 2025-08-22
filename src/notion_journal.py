# src/integrations/notion_journal.py
from __future__ import annotations

import os, logging
from dataclasses import dataclass
from typing import Optional, Literal
from datetime import datetime, timezone
from notion_client import Client as NotionClient

def _now_iso() -> str:
    # If you set TZ=Europe/Zurich in docker-compose, Python will still give UTC here;
    # Notion stores timezone-aware date; using UTC is fine and portable.
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

@dataclass
class TradeEvent:
    event: Literal["OPEN", "CLOSE"]
    symbol: str
    direction: Optional[Literal["BUY", "SELL"]] = None
    order_type: Optional[str] = None
    lots: Optional[float] = None
    volume_units: Optional[int] = None
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_id: Optional[int] = None
    order_id: Optional[int] = None
    account_id: Optional[int] = None
    timestamp: Optional[str] = None
    status: Optional[str] = None
    pnl: Optional[float] = None
    note: Optional[str] = None

class NotionJournal:
    """Fire-and-forget journaling to a Notion database. Safe to no-op if disabled."""
    def __init__(self, secret: str, database_id: str, enabled: bool = True):
        self.enabled = enabled and bool(secret and database_id)
        self.db_id = database_id
        self.client = NotionClient(auth=secret) if self.enabled else None
        if not self.enabled:
            logging.info("[Notion] disabled (missing creds or NOTION_ENABLED=0)")

    @classmethod
    def from_env(cls) -> "NotionJournal":
        enabled = str(os.getenv("NOTION_ENABLED", "1")).strip().lower() not in {"0","false","no","off"}
        return cls(
            secret=os.getenv("NOTION_SECRET", ""),
            database_id=os.getenv("NOTION_DB_ID", ""),
            enabled=enabled,
        )

    def log_trade(self, evt: TradeEvent) -> Optional[str]:
        if not self.enabled:
            return None
        try:
            ts = evt.timestamp or _now_iso()
            title = f"{evt.event} {evt.symbol} {evt.direction or ''}".strip()
            props = {
                "Title":       {"title": [{"text": {"content": title}}]},
                "Symbol":      {"rich_text": [{"text": {"content": evt.symbol}}]},
                "Event":       {"select": {"name": evt.event}},
                "Direction":   {"select": {"name": evt.direction}} if evt.direction else None,
                "Order Type":  {"select": {"name": evt.order_type}} if evt.order_type else None,
                "Lots":        {"number": evt.lots} if evt.lots is not None else None,
                "Volume Units":{"number": evt.volume_units} if evt.volume_units is not None else None,
                "Price":       {"number": evt.price} if evt.price is not None else None,
                "Stop Loss":   {"number": evt.stop_loss} if evt.stop_loss is not None else None,
                "Take Profit": {"number": evt.take_profit} if evt.take_profit is not None else None,
                "Position ID": {"rich_text": [{"text": {"content": str(evt.position_id)}}]} if evt.position_id else None,
                "Order ID":    {"rich_text": [{"text": {"content": str(evt.order_id)}}]} if evt.order_id else None,
                "Account ID":  {"rich_text": [{"text": {"content": str(evt.account_id)}}]} if evt.account_id else None,
                "Timestamp":   {"date": {"start": ts}},
                "Status":      {"rich_text": [{"text": {"content": evt.status or ''}}]} if evt.status else None,
                "PnL":         {"number": evt.pnl} if evt.pnl is not None else None,
                "Note":        {"rich_text": [{"text": {"content": evt.note or ''}}]} if evt.note else None,
            }
            props = {k:v for k,v in props.items() if v is not None}
            res = self.client.pages.create(parent={"database_id": self.db_id}, properties=props)
            page_id = res.get("id")
            logging.info(f"[Notion] journaled trade page_id={page_id}")
            return page_id
        except Exception as e:
            logging.error(f"[Notion] failed to journal trade: {e}")
            return None
