import asyncio

from .kumagpt_model import KumaGPTInference


class InferenceService:
    def __init__(self, model: KumaGPTInference, timeout: float):
        self.model = model
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(1)

    async def generate(self, messages: list[dict[str, str]]) -> str:
        async with self.semaphore:
            return await asyncio.wait_for(asyncio.to_thread(self.model.generate, messages), self.timeout)
