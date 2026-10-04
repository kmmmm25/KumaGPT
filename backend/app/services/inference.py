import asyncio

from .kumagpt_model import KumaGPTInference


class InferenceService:
    def __init__(self, model: KumaGPTInference, timeout: float):
        self.model = model
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(1)

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        top_k: int | None = None,
    ) -> str:
        async with self.semaphore:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self.model.generate,
                    messages,
                    temperature=temperature,
                    top_k=top_k,
                ),
                self.timeout,
            )
