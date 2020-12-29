import subprocess

import uvicorn
import nest_asyncio

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


nest_asyncio.apply()
APP = FastAPI()


@APP.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "message": str(exc),
        },
    )


class Serve:
    def __init__(self, name=None, port=5000, host="localhost", log_level="info"):
        self.name = name
        self.port = port
        self.host = host
        self.log_level = log_level

    def __call__(self, routes=None):
        assert isinstance(
            routes, dict
        ), f"routes should be dict but passed {type(routes)}"
        for name, route in routes.items():
            route, _type = route
            assert isinstance(name, str), f"name should be string."
            _wrapper = getattr(APP, _type)(name)
            _wrapper(route)

    def init(
        self,
    ):
        uvicorn.run(APP, host=self.host, port=self.port, log_level=self.log_level)
