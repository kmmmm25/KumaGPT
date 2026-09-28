from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router
from .config import get_settings
from .db import Base, engine
from .services.inference import InferenceService
from .services.kumagpt_model import KumaGPTInference

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    model = KumaGPTInference(settings)
    model.load()
    app.state.inference = InferenceService(model, settings.inference_timeout_seconds)
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins, allow_methods=["*"], allow_headers=["*"])
app.include_router(router, prefix=settings.api_prefix)


@app.get("/health")
def health():
    return {"status": "ok", "model_backend": settings.model_backend}
