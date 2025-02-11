import requests
from pydantic import BaseModel
from typing import List, Optional


class FastFilterInput(BaseModel):
    # mirroring the (default) wrapper; convenient to turn into a client library
    smiles: List[str]


class FastFilterOutcome(BaseModel):
    smiles: str
    template_ids: list
    num_examples: int


class FastFilterResult(BaseModel):
    rank: float
    outcome: FastFilterOutcome
    score: float
    prob: float


class FastFilterResponse(BaseModel):
    # mirroring the (default) wrapper, but without BaseResponse (semi-hardcode)
    status_code: int
    message: str
    result: Optional[FastFilterResult]


class FastFilterAPI:
    """fast filter API to be used as a fast filter"""
    def __init__(self, default_url: str):
        self.default_url = default_url
        self.session = requests.Session()

    def __call__(self, reactant_smiles: str, target: str, url: str = None
                 ) -> Optional[float]:
        if not url:
            url = self.default_url

        input = {"smiles": [reactant_smiles, target]}

        FastFilterInput(**input)                # merely validate the input
        try:
            response = self.session.post(url=url, json=input).json()
            FastFilterResponse(**response)      # merely validate the response
        except requests.ConnectionError as e:
            # Handle the connection error appropriately
            print("Connection error:", e)

            return None
        except Exception as e:
            # Handle any other exception that might occur
            print("An error occurred:", e)

            return None

        try:
            score = response["result"]["score"]
        except KeyError:
            print("score not found in result")
            score = None

        return score
