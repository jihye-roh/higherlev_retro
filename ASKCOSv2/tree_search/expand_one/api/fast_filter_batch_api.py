import requests
from pydantic import BaseModel
from typing import List, Optional


class FastFilterBatchInput(BaseModel):
    # mirroring the (default) wrapper; convenient to turn into a client library
    rxn_smiles: List[str]


class FastFilterBatchResponse(BaseModel):
    # mirroring the (default) wrapper, but without BaseResponse (semi-hardcode)
    status_code: int
    message: str
    result: Optional[List[float]]


class FastFilterBatchAPI:
    """fast filter Batch API to be used as a fast filter for batch query"""
    def __init__(self, default_url: str):
        self.default_url = default_url
        self.session = requests.Session()

    def __call__(self, rxn_smiles: List[str], url: str = None
                 ) -> Optional[List[float]]:
        if not url:
            url = self.default_url

        input = {"rxn_smiles": rxn_smiles}

        FastFilterBatchInput(**input)               # merely validate the input
        try:
            response = self.session.post(url=url, json=input).json()
            FastFilterBatchResponse(**response)     # merely validate the response
        except requests.ConnectionError as e:
            # Handle the connection error appropriately
            print("Connection error:", e)

            return None
        except Exception as e:
            # Handle any other exception that might occur
            print("An error occurred:", e)

            return None

        try:
            scores = response["result"]
        except KeyError:
            print("score not found in result")
            scores = None

        return scores
