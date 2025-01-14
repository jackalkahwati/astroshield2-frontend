from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/')
async def read_root():
    return {'message': 'Welcome to AstroShield API'}

handler = Mangum(app, lifespan='off', api_gateway_base_path=None, strip_stage_path=True)

__all__ = ['handler']
