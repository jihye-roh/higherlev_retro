import requests
import traceback as tb
from pydantic import BaseModel
from typing import Dict, List, Optional


class HistorianInput(BaseModel):
    # mirroring the (default) wrapper; convenient to turn into a client library
    smiles: str
    template_sets: Optional[List[str]] = None
    canonicalize: bool


class HistorianResponse(BaseModel):
    # mirroring the (default) wrapper, but without BaseResponse (semi-hardcode)
    as_reactant: int
    as_product: int


class HistorianAPI:
    """Historian API to be used as a Historian"""
    def __init__(self, default_url: str):
        self.default_url = default_url
        self.session = requests.Session()

    def __call__(
        self,
        smiles: str,
        template_sets: Optional[List[str]] = None,
        canonicalize: bool = False,
        url: str = None
    ) -> Dict[str, int]:
        if not url:
            url = self.default_url

        input = {
            "smiles": smiles,
            "template_sets": template_sets,
            "canonicalize": canonicalize
        }

        HistorianInput(**input)                     # merely validate the input
        try:
            response = self.session.post(url=url, params=input).json()
            HistorianResponse(**response)           # merely validate the response
        except requests.ConnectionError as e:
            # Handle the connection error appropriately
            print("Connection error for HistorianAPI:")
            tb.print_exc()

            return {"as_reactant": 0, "as_product": 0}
        except Exception as e:
            # Handle any other exception that might occur
            print("An error occurred for HistorianAPI:")
            tb.print_exc()

            return {"as_reactant": 0, "as_product": 0}

        return response
