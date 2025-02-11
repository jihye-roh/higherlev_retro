import requests
import traceback as tb
from pydantic import BaseModel
from typing import Dict, List, Optional, Union


class SCScorerBatchInput(BaseModel):
    # mirroring the (default) wrapper; convenient to turn into a client library
    smiles: List[str]


class SCScorerBatchResponse(BaseModel):
    # mirroring the (default) wrapper, but without BaseResponse (semi-hardcode)
    status_code: int
    message: str
    result: Optional[Dict[str, float]]


class SCScorerBatchAPI:
    """SCScorerBatch API to be used as an SCScorer for batch query"""
    def __init__(self, default_url: str):
        self.default_url = default_url
        self.session = requests.Session()

    def __call__(self, smiles_list: List[str], url: str = None
                 ) -> Union[Dict[str, float], None]:
        if not url:
            url = self.default_url

        input = {"smiles": smiles_list}

        SCScorerBatchInput(**input)                 # merely validate the input
        try:
            response = self.session.post(url=url, json=input).json()
            SCScorerBatchResponse(**response)       # merely validate the response
        except requests.exceptions.ConnectionError:
            # Handle the connection error appropriately
            print("Connection error for SCScorerBatchAPI:")
            tb.print_exc()

            return None
        except Exception:
            # Handle any other exception that might occur
            print("An error occurred for SCScorerBatchAPI:")
            tb.print_exc()

            return None

        result = response["result"]

        return result
