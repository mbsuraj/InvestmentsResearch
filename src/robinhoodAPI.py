import base64
import os
import datetime
import json
from typing import Any, Dict, Optional
import uuid
import requests
from cryptography.hazmat.primitives.asymmetric import ed25519
import robin_stocks.robinhood as r
import pyotp

class StockAPITrading:
    def __init__(self,):
        """
        Initializes the StockAPITrading class and logs into the Robinhood account.
        """
        self.username = os.environ["ROBINHOOD_USERNAME"]
        self.password = os.environ["ROBINHOOD_PASSWORD"]
        self.mfa_key = os.environ["ROBINHOOD_MFA"]
        self.logged_in = False
        self.login()

    def login(self):
        """
        Logs in to the Robinhood account using provided credentials.
        """
        try:
            totp = self.generate_totp()
            # r.login(self.username, self.password, mfa_code=totp)
            r.login(self.username, self.password, by_sms=True)
            self.logged_in = True
        except Exception as e:
            raise Exception(f"Failed to log in to Robinhood: {e}")

    def generate_totp(self):
        """
        Generates a TOTP code using the provided MFA key.
        """
        totp = pyotp.TOTP(self.mfa_key).now()
        return totp

    def get_stock_price(self, symbol: str):
        """
        Fetches the current bid and ask price for the specified stock.
        :param symbol: Stock ticker symbol (e.g., 'AAPL').
        :return: Dictionary with 'bid_price' and 'ask_price'.
        """
        try:
            ask_price = r.stocks.get_latest_price(symbol, includeExtendedHours=True, priceType="ask_price")
            bid_price = r.stocks.get_latest_price(symbol, includeExtendedHours=True, priceType="bid_price")
            return {
                "bid_price": bid_price[0],
                "ask_price": ask_price[0]
            }
        except Exception as e:
            raise Exception(f"Error fetching stock price for {symbol}: {e}")

    def execute_order(self, symbol: str, amountInDollars: float, side: str, order_type: str = "market"):
        """
        Executes a buy or sell order for the specified stock.
        :param symbol: Stock ticker symbol (e.g., 'AAPL').
        :param quantity: Number of shares (can be fractional).
        :param side: 'buy' or 'sell'.
        :param order_type: Type of order ('market' or 'limit').
        :return: Order response.
        """
        try:
            if side.lower() not in ["buy", "sell"]:
                raise ValueError("Side must be 'buy' or 'sell'.")

            if order_type.lower() == "market":
                if side.lower() == "buy":
                    return r.orders.order_buy_fractional_by_price(
                        symbol, amountInDollars, timeInForce="gfd"
                    )
                elif side.lower() == "sell":
                    return r.orders.order_sell_fractional_by_price(
                        symbol, amountInDollars, timeInForce="gfd"
                    )
            else:
                raise ValueError("Currently, only market orders are supported.")
        except Exception as e:
            raise Exception(f"Error executing {side} order for {symbol}: {e}")

    def logout(self):
        """
        Logs out of the Robinhood account.
        """
        try:
            r.logout()
            self.logged_in = False
        except Exception as e:
            raise Exception(f"Failed to log out of Robinhood: {e}")

class CryptoAPITrading:
    def __init__(self):
        self.api_key = os.environ['API_KEY'].strip()
        private_bytes = base64.b64decode(os.environ['BASE64_PRIVATE_KEY'].strip())
        # Note that the cryptography library used here only accepts a 32 byte ed25519 private key
        self.private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_bytes[:32])
        self.base_url = "https://trading.robinhood.com"

    @staticmethod
    def _get_current_timestamp() -> int:
        return int(datetime.datetime.now(tz=datetime.timezone.utc).timestamp())

    @staticmethod
    def get_query_params(key: str, *args: Optional[str]) -> str:
        if not args:
            return ""

        params = []
        for arg in args:
            params.append(f"{key}={arg}")

        return "?" + "&".join(params)

    def make_api_request(self, method: str, path: str, body: str = "") -> Any:
        timestamp = self._get_current_timestamp()
        headers = self.get_authorization_header(method, path, body, timestamp)
        url = self.base_url + path

        try:
            response = {}
            if method == "GET":
                response = requests.get(url, headers=headers, timeout=10)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=json.loads(body), timeout=10)
            return response.json()
        except requests.RequestException as e:
            print(f"Error making API request: {e}")
            return None

    def get_authorization_header(
            self, method: str, path: str, body: str, timestamp: int
    ) -> Dict[str, str]:
        message_to_sign = f"{self.api_key}{timestamp}{path}{method}{body}"
        signature = self.private_key.sign(message_to_sign.encode("utf-8"))

        return {
            "x-api-key": self.api_key,
            "x-signature": base64.b64encode(signature).decode("utf-8"),
            "x-timestamp": str(timestamp),
        }

    def get_account(self) -> Any:
        path = "/api/v1/crypto/trading/accounts/"
        return self.make_api_request("GET", path)

    # The symbols argument must be formatted in trading pairs, e.g "BTC-USD", "ETH-USD". If no symbols are provided,
    # all supported symbols will be returned
    def get_trading_pairs(self, *symbols: Optional[str]) -> Any:
        query_params = self.get_query_params("symbol", *symbols)
        path = f"/api/v1/crypto/trading/trading_pairs/{query_params}"
        return self.make_api_request("GET", path)

    # The asset_codes argument must be formatted as the short form name for a crypto, e.g "BTC", "ETH". If no asset
    # codes are provided, all crypto holdings will be returned
    def get_holdings(self, *asset_codes: Optional[str]) -> Any:
        query_params = self.get_query_params("asset_code", *asset_codes)
        path = f"/api/v1/crypto/trading/holdings/{query_params}"
        return self.make_api_request("GET", path)

    # The symbols argument must be formatted in trading pairs, e.g "BTC-USD", "ETH-USD". If no symbols are provided,
    # the best bid and ask for all supported symbols will be returned
    def get_best_bid_ask(self, *symbols: Optional[str]) -> Any:
        query_params = self.get_query_params("symbol", *symbols)
        path = f"/api/v1/crypto/marketdata/best_bid_ask/{query_params}"
        return self.make_api_request("GET", path)

    # The symbol argument must be formatted in a trading pair, e.g "BTC-USD", "ETH-USD"
    # The side argument must be "bid", "ask", or "both".
    # Multiple quantities can be specified in the quantity argument, e.g. "0.1,1,1.999".
    def get_estimated_price(self, symbol: str, side: str, quantity: str) -> Any:
        path = f"/api/v1/crypto/marketdata/estimated_price/?symbol={symbol}&side={side}&quantity={quantity}"
        return self.make_api_request("GET", path)

    def place_order(
            self,
            client_order_id: str,
            side: str,
            order_type: str,
            symbol: str,
            order_config: Dict[str, str],
    ) -> Any:
        body = {
            "client_order_id": client_order_id,
            "side": side,
            "type": order_type,
            "symbol": symbol,
            f"{order_type}_order_config": order_config,
        }
        path = "/api/v1/crypto/trading/orders/"
        return self.make_api_request("POST", path, json.dumps(body))

    def cancel_order(self, order_id: str) -> Any:
        path = f"/api/v1/crypto/trading/orders/{order_id}/cancel/"
        return self.make_api_request("POST", path)

    def get_order(self, order_id: str) -> Any:
        path = f"/api/v1/crypto/trading/orders/{order_id}/"
        return self.make_api_request("GET", path)

    def get_orders(self) -> Any:
        path = "/api/v1/crypto/trading/orders/"
        return self.make_api_request("GET", path)

def execute_crypto_trade_in_dollars(symbol: str, side: str, dollar_amount: float, trading_client=CryptoAPITrading()):
    api_trading_client = trading_client

    # Step 1: Get the current price of the cryptocurrency
    client_side = 'bid' if side == 'buy' else 'sell'
    estimated_price = api_trading_client.get_estimated_price(symbol, client_side, "1")
    current_price = float(estimated_price["results"][0]["price"])  # Assume price is in the response

    # Step 2: Calculate the quantity of the cryptocurrency to trade
    asset_quantity = dollar_amount / current_price
    asset_quantity_str = f"{asset_quantity:.5f}"  # Ensure the quantity is formatted correctly

    # Step 3: Place the order
    order = api_trading_client.place_order(
        str(uuid.uuid4()),
        side,  # "buy" or "sell"
        "market",
        symbol,
        {"asset_quantity": asset_quantity_str}
    )
    return order

def execute_stock_trade_in_dollars(symbol: str, side: str, dollar_amount: float, trading_client: StockAPITrading):
    api_trading_client = trading_client

    # Step 1: Get the current price of the cryptocurrency
    client_side = 'bid_price' if side == 'buy' else 'ask_price'
    estimated_price = api_trading_client.get_stock_price(symbol)
    current_price = float(estimated_price[client_side])  # Assume price is in the response

    # Step 2: Calculate the quantity of the cryptocurrency to trade
    asset_quantity = round(dollar_amount / current_price, 8)
    asset_quantity_str = f"{asset_quantity:.5f}"  # Ensure the quantity is formatted correctly

    # Step 3: Place the order
    order = api_trading_client.execute_order(symbol=symbol,
                                             amountInDollars=dollar_amount,
                                             side=side,
                                             order_type="market")
    api_trading_client.logout()
    return order