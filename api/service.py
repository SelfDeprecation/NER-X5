import logging
from time import time
from typing import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.concurrency import run_in_threadpool

from api import schemas, helpers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model on startup...")
    try:
        helpers.model_manager.load_model()
        logger.info("Model loaded successfully and ready to serve requests")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        raise
    yield
    logger.info("Shutting down...")

app = FastAPI(title='X5 entity allocator', lifespan=lifespan)

def get_model_predict():
    return helpers.model_manager.predict

@app.get('/')
def root():
    return {
        'status': 'ok'
    }

@app.get("/health-check")
def healthcheck():
    return {
        'status': 'OK'
    }

@app.post('/api/predict')
async def predict(
    request: schemas.PredictRequestModel, model_predict: Callable = Depends(get_model_predict)
):
    start_time = time()

    prediction = await run_in_threadpool(model_predict, request.input)

    time_taken = (time() - start_time) * 1000
    logger.info(f"  {time_taken:.2f} ms spent on '{request.input}': {prediction}")

    return prediction
