import requests
from pydantic import BaseModel
from typing import List, Optional, Tuple


class GetTopClassBatchInput(BaseModel):
    smiles: List[str]
    level: int = 2
    threshold: float = 0.2


class GetTopClassBatchResponse(BaseModel):
    # mirroring the (default) wrapper, but without BaseResponse (semi-hardcode)
    status_code: int
    message: str
    result: Optional[List[List[str]]]


class ReactionClassificationAPI:
    """ReactionClassification API to be used as a reaction classifier"""
    def __init__(self, url: str):
        self.default_url = url
        self.session = requests.Session()

    def __call__(
        self,
        smiles_list: List[str],
        level: int = 2,
        threshold: float = None,
        url: str = None
    ) -> Optional[List[Tuple[str, str]]]:
        if not url:
            url = self.default_url

        input = {
            "smiles": smiles_list,
            "level": level,
            "threshold": threshold
        }

        GetTopClassBatchInput(**input)              # merely validate the input
        response = self.session.post(url=url, json=input).json()
        GetTopClassBatchResponse(**response)        # merely validate the response

        result = response["result"]

        return result
