"""
Finance information about companies
Using yahoo finance
"""
import yfinance as yf
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta

from forge.sdk.memory.memstore_tools import add_ability_memory

from ..forge_log import ForgeLogger
from .registry import ability

logger = ForgeLogger(__name__)

ability(
    name="get_ticker_info",
    description="Get information about a specific ticker symbol",
    parameters=[
        {
            "name": "ticker_symbol",
            "description": "ticker symbol",
            "type": "string",
            "required": True
        }
    ],
    output_type="str"
)
async def get_ticker_info(
    agent,
    task_id: str,
    ticker_symbol: str
) -> str:
    # get ticker then financial information as dict
    # dict has structure where timestamp are keys
    try:
        stock = yf.Ticker(ticker_symbol)
        stock_info = json.dumps(stock.get_info())

        add_ability_memory(
            task_id,
            stock_info,
            "get_financials_year",
            agent.memstore
        )
    except Exception as err:
        logger.error(f"get_ticker_info failed: {err}")
        raise err

    return stock_info

@ability(
    name="get_ticker_financials_year",
    description=f"Get financial information of a stock market ticker",
    parameters=[
        {
            "name": "ticker_symbol",
            "description": "ticker symbol",
            "type": "string",
            "required": True
        },
        {
            "name": "year",
            "description": "year wanted for financials",
            "type": "integer",
            "required": True
        }
    ],
    output_type="str"
)
async def get_financials_year(
    agent,
    task_id: str,
    ticker_symbol: str,
    year: int
) -> str:
    # get ticker then financial information as dict
    # dict has structure where timestamp are keys
    year_financial_data = None

    try:
        stock = yf.Ticker(ticker_symbol)
        financials_dict = stock.financials.to_dict()

        # get financial informatio and return dict
        for timestamp, financial_data in financials_dict.items():
            key_year = int(str(timestamp).split("-")[0])

            if key_year == year:
                year_financial_data = financial_data

                add_ability_memory(
                    task_id,
                    str(financial_data),
                    "get_financials_year",
                    agent.memstore
                )
    except Exception as err:
        logger.error(f"get_financials_year failed: {err}")
        raise err

    if year_financial_data:
        json_data = json.dumps(year_financial_data)
        return json_data
    else:
        return "Financial information not found. Try searching the internet."

