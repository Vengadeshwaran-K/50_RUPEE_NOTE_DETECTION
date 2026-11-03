from fastapi import APIRouter, Request

from ..schemas import HealthResponse, ModelInfoResponse

router = APIRouter(prefix="/api", tags=["status"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Backend liveness and whether the ONNX model is loaded."""
    service = request.app.state.detector
    return HealthResponse(status="ok", model_loaded=service.loaded, model_error=service.error)


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    """Target class, model status, input size and inference backend."""
    return ModelInfoResponse(**request.app.state.detector.info())
