"""API server entrypoint.

This FastAPI server dynamically registers model endpoints for flares linguistic models and mobility travel time models.
"""

from typing import List
from fastapi import FastAPI
from src.schemas.schemas import TextRequest, ReliabilitySample
from src.services.flares_service import flares_models
from src.services.mobility_service import mobility_models
from pydantic import create_model

app = FastAPI(title="Models Server")

# -------------------------
# dynamic flares model endpoints
# -------------------------

def _make_wh_endpoint(model_name: str):
    def endpoint(requests: List[TextRequest]):
        return flares_models.predict_wh_batch(model_name, requests)
    return endpoint


def _make_reliability_endpoint(model_name: str):
    def endpoint(requests: List[ReliabilitySample]):
        return flares_models.predict_reliability_batch(model_name, requests)
    return endpoint


for model_name, model_type in flares_models.model_types.items():
    route = f"/flares/{model_name}"
    if model_type == "5w1h":
        app.add_api_route(route, _make_wh_endpoint(model_name), methods=["POST"], summary=f"5W1H model: {model_name}")
    else:
        app.add_api_route(route, _make_reliability_endpoint(model_name), methods=["POST"], summary=f"Reliability model: {model_name}")


# -------------------------
# dynamic mobility model endpoints
# -------------------------

def _build_dynamic_request_model(
    model_name: str,
) -> type:
    """
    Build request schema dynamically from model config columns.
    """

    columns = mobility_models.get_model_columns(model_name)

    fields: Dict[str, Any] = {}

    for col in columns:
        # basic inference
        if col.endswith("_id"):
            fields[col] = (str, ...)
        elif col in {"hour", "day", "month", "year"}:
            fields[col] = (int, ...)
        else:
            fields[col] = (float, ...)

    return create_model(
        f"{model_name}Request",
        **fields,
    )


def _make_mobility_endpoint(target: str):

    RequestModel = _build_dynamic_request_model(target)

    def endpoint(requests: List[RequestModel]):
        payload = [r.model_dump() for r in requests]
        return mobility_models.predict_batch(target, payload)

    return endpoint


for target in mobility_models.list_models():

    route = f"/mobility/{target}"

    app.add_api_route(
        route,
        _make_mobility_endpoint(target),
        methods=["POST"],
        summary=f"Mobility model: {target}",
    )

# -------------------------
# list models
# -------------------------
@app.get("/models")
def models():
    return {
        "flares": flares_models.list_models(),
        "mobility": mobility_models.list_models(),
    }
