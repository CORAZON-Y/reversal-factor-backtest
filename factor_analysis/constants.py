"""Project constants and raw data field definitions."""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "basic_data"
OUTPUT_DIR = ROOT_DIR / "output"

DAILY_DATA_FILE = "daily_data.parquet"
INDUSTRY_FILE = "industry.parquet"
ST_FILE = "st.parquet"
SUSPENSION_FILE = "停牌.parquet"

OPEN_COL = "DY-ADJ_AF-OPEN_PRICE_2"
CLOSE_COL = "DY-ADJ_AF-CLOSE_PRICE_2"
MARKET_VALUE_COL = "DY-BASIC-MARKET_VALUE"
FLOAT_MARKET_VALUE_COL = "DY-BASIC-NEG_MARKET_VALUE"
CHG_STATUS_COL = "DY-IND-CHG_STATUS"

DAILY_COLUMNS = [
    OPEN_COL,
    CLOSE_COL,
    MARKET_VALUE_COL,
    FLOAT_MARKET_VALUE_COL,
    CHG_STATUS_COL,
]

RAW_DAILY_RENAME = {
    OPEN_COL: "open",
    CLOSE_COL: "close",
    MARKET_VALUE_COL: "market_value",
    FLOAT_MARKET_VALUE_COL: "float_market_value",
    CHG_STATUS_COL: "chg_status",
}

RAW_INDUSTRY_RENAME = {
    "TYPE_ID": "industry_id",
    "LEVEL1_NAME": "industry_level1",
    "LEVEL2_NAME": "industry_level2",
    "LEVEL3_NAME": "industry_level3",
}

SUSPENDED_STATUS = -1
LIMIT_STATUS = {2, 3, 5, 6}
