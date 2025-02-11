import requests
import traceback as tb
from pydantic import BaseModel
from typing import Optional, Union


class SCScorerInput(BaseModel):
    # mirroring the (default) wrapper; convenient to turn into a client library
    smiles: str


class SCScorerResponse(BaseModel):
    # mirroring the (default) wrapper, but without BaseResponse (semi-hardcode)
    status_code: int
    message: str
    result: Optional[float]


class SCScorerAPI:
    """SCScorer API to be used as an SCScorer"""
    def __init__(self, default_url: str):
        self.default_url = default_url
        self.session = requests.Session()

    def __call__(self, smiles: str, url: str = None) -> Union[float, None]:
        if not url:
            url = self.default_url

        input = {"smiles": smiles}

        SCScorerInput(**input)                      # merely validate the input
        try:
            response = self.session.post(url=url, json=input).json()
            SCScorerResponse(**response)            # merely validate the response
        except requests.exceptions.ConnectionError:
            # Handle the connection error appropriately
            print("Connection error for SCScorerAPI:")
            tb.print_exc()

            return None
        except Exception:
            # Handle any other exception that might occur
            print("An error occurred for SCScorerAPI:")
            tb.print_exc()
            print(smiles)

            return None

        result = response["result"]

        return result
