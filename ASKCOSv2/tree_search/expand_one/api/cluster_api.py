import requests
import traceback as tb
from pydantic import BaseModel
from typing import Dict, List, Optional, Tuple, Union


class ClusterSetting(BaseModel):
    feature: str = "original"
    cluster_method: str = "kmeans"
    fp_type: str = "morgan"
    fp_length: int = 512
    fp_radius: int = 1
    classification_threshold: float = 0.2


class ClusterInput(ClusterSetting):
    # mirroring the (default) wrapper; convenient to turn into a client library
    original: str
    outcomes: List[str]
    scores: List[float] = None


class ClusterResponse(BaseModel):
    # mirroring the (default) wrapper, but without BaseResponse (semi-hardcode)
    status_code: int
    message: str
    result: List[Union[List[int], Dict[str, str]]]


class ClusterAPI:
    """Cluster API to be used as a clusterer"""
    def __init__(self, default_url: str):
        self.default_url = default_url
        self.session = requests.Session()

    def __call__(
        self,
        original: str,
        outcomes: List[str],
        scores: List[float] = None,
        cluster_setting: ClusterSetting = ClusterSetting(),
        url: str = None
    ) -> Optional[Tuple[List[int], Dict[str, str]]]:
        if not url:
            url = self.default_url

        input = {
            "original": original,
            "outcomes": outcomes,
            "scores": scores,
            "feature": cluster_setting.feature,
            "cluster_method": cluster_setting.cluster_method,
            "fp_type": cluster_setting.fp_type,
            "fp_length": cluster_setting.fp_length,
            "fp_radius": cluster_setting.fp_radius,
            "classification_threshold": cluster_setting.classification_threshold
        }

        ClusterInput(**input)                   # merely validate the input
        try:
            response = self.session.post(url=url, json=input).json()
            ClusterResponse(**response)         # merely validate the response
        except requests.ConnectionError as e:
            # Handle the connection error appropriately
            print("Connection error for ClusterAPI:")
            tb.print_exc()

            return [], {}
        except Exception as e:
            # Handle any other exception that might occur
            print("An error occurred for ClusterAPI:")
            tb.print_exc()

            return [], {}

        cluster_ids, name_dict = response["result"]

        return cluster_ids, name_dict
