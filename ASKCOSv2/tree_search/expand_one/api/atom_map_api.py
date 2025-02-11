import requests
from pydantic import BaseModel
from typing import List, Optional


class AtomMapInput(BaseModel):
    # mirroring the wrappers; convenient to turn into a client library
    backend: str = "rxnmapper"
    smiles: List[str]


class AtomMapResponse(BaseModel):
    # mirroring the wrappers, but without BaseResponse (semi-hardcode)
    status_code: int
    message: str
    result: Optional[List[str]]


class AtomMapAPI:
    """atom map API to be used as an atom mapper"""
    def __init__(self, url: str, backend: str = "rxnmapper"):
        self.default_url = url
        self.default_backend = backend

    def __call__(self, smiles: List[str], backend: str = None, url: str = None
                 ) -> Optional[List[str]]:
        if not url:
            url = self.default_url

        if not backend:
            backend = self.default_backend

        input = {
            "backend": backend,
            "smiles": smiles
        }

        AtomMapInput(**input)                   # merely validate the input
        response = requests.post(url=url, json=input).json()
        AtomMapResponse(**response)             # merely validate the response

        result = response["result"]

        return result

