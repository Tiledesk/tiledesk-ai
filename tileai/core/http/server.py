import tileai.shared.const
import logging
import os
import json

import asyncio
import multiprocessing
from time import sleep
from threading import Event

from functools import reduce, wraps
import concurrent.futures

from tileai.shared.exceptions import ErrorResponse

import tileai.shared.const as const

from queue import Queue

import aiohttp
import jsonschema
from sanic import Sanic, response
from sanic.request import Request
from sanic.response import HTTPResponse
from sanic_cors import CORS
from sanic import Blueprint
from http import HTTPStatus

from typing import (
    Any,
    Callable,
    DefaultDict,
    List,
    Optional,
    Text,
    Union,
    Dict,
    TYPE_CHECKING,
    NoReturn,
    Coroutine,
)

logger = logging.getLogger(__name__)

def configure_cors(
    app: Sanic, cors_origins: Union[Text, List[Text], None] = ""
) -> None:
    """Configure CORS origins for the given app. Presa da RASA"""

    # Workaround so that socketio works with requests from other origins.
    # https://github.com/miguelgrinberg/python-socketio/issues/205#issuecomment-493769183
    app.config.CORS_AUTOMATIC_OPTIONS = True
    app.config.CORS_SUPPORTS_CREDENTIALS = True
    app.config.CORS_EXPOSE_HEADERS = "filename"

    CORS(
        app, resources={r"/*": {"origins": cors_origins or ""}}, automatic_options=True
    )


def async_callback_url(f: Callable[..., Coroutine]) -> Callable:
    """Decorator to enable async request handling.

    If the incoming HTTP request specified a `webhook_url` json parameter, the request
    will return immediately with a 204 while the actual request response will
    be sent to the `webhook_url`. If an error happens, the error payload will also
    be sent to the `webhook_url`.

    Args:
        f: The request handler function which should be decorated.

    Returns:
        The decorated function.
    """

    @wraps(f)
    async def decorated_function(
        request: Request, *args: Any, **kwargs: Any
    ) -> HTTPResponse:
        
       
        #webhook_url
        #
        if not "webhook_url" in request.json.keys():
            return await f(request, *args, **kwargs)
        webhook_url = request.json["webhook_url"]
        #print("Webhook url",webhook_url )
        # Only process request asynchronously if the user specified a `webhook_url`
        # json keys.
        if not webhook_url:
            return await f(request, *args, **kwargs)

        async def wrapped() -> None:
            try:
                
                result: HTTPResponse = await f(request, *args, **kwargs)
                print(result)
                payload: Dict[Text, Any] = dict(
                    data=result.body, headers={"Content-Type": result.content_type}
                )
                logger.debug(
                    "Asynchronous processing of request was successful. "
                    "Sending result to webhook URL."
                )

            except Exception as e:
                if not isinstance(e, ErrorResponse):
                    logger.error(e)
                    e = ErrorResponse(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        "UnexpectedError",
                        f"An unexpected error occurred. Error: {e}",
                    )
                # If an error happens, we send the error payload to the `callback_url`
                payload = dict(json=e.error_info)
                logger.error(
                    "Error happened when processing request asynchronously. "
                    "Sending error to callback URL."
                )
            async with aiohttp.ClientSession() as session:
                await session.post(webhook_url, raise_for_status=True, **payload)

        # Run the request in the background on the event loop
        request.app.add_task(wrapped())

        # The incoming request will return immediately with a 204
        return response.empty()

    return decorated_function



def run_in_thread(f: Callable[..., Coroutine]) -> Callable:
    """Decorator which runs request on a separate thread.

    Some requests (e.g. training or cross-validation) are computional intense requests.
    This means that they will block the event loop and hence the processing of other
    requests. This decorator can be used to process these requests on a separate thread
    to avoid blocking the processing of incoming requests.

    Args:
        f: The request handler function which should be decorated.

    Returns:
        The decorated function.
    """

    @wraps(f)
    async def decorated_function(
        request: Request, *args: Any, **kwargs: Any
    ) -> HTTPResponse:
        # Use a sync wrapper for our `async` function as `run_in_executor` only supports
        # sync functions
        def run() -> HTTPResponse:
            return asyncio.run(f(request, *args, **kwargs))

        
       
        with concurrent.futures.ThreadPoolExecutor() as pool:
            #from os import getpid
            #from os import getppid
            #from threading import current_thread
            #print("==================================================")
            #print(f'Worker pid={getpid()}, ppid={getppid()} thread={current_thread().name}')
            #print("==================================================")
            future = await request.app.loop.run_in_executor(pool, run)
            #future = request.app.loop.run_in_executor(pool, run)
            #sleep(1)
            #was_cancelled = future.cancel()
            #print(f'Second task was cancelled: {was_cancelled}')
            #await future
            
            return future

    return decorated_function

def validate_request_body(request: Request, error_message: Text) -> None:
    """Check if `request` has a body."""
    if not request.body:
        raise ErrorResponse(HTTPStatus.BAD_REQUEST, "BadRequest", error_message)

def create_app(
    app : Sanic = None,
    cors_origins: Union[Text, List[Text], None] = "*",
    auth_token: Optional[Text] = None,
    response_timeout: int = tileai.shared.const.DEFAULT_RESPONSE_TIMEOUT,
    jwt_secret: Optional[Text] = None,
    #jwt_method: Text = "HS256",
    endpoints: Optional[Text] = None,
) -> Sanic:
    """Class representing a Tileai HTTP server."""

    
    
    app = Sanic(name="tileai_server")
    app.config.RESPONSE_TIMEOUT = response_timeout
    configure_cors(app, cors_origins)

    add_root_route(app)
   
    """

    # Setup the Sanic-JWT extension
    if jwt_secret and jwt_method:
        # `sanic-jwt` depends on having an available event loop when making the call to
        # `Initialize`. If there is none, the server startup will fail with
        # `There is no current event loop in thread 'MainThread'`.
        try:
            _ = asyncio.get_running_loop()
        except RuntimeError:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)

        # since we only want to check signatures, we don't actually care
        # about the JWT method and set the passed secret as either symmetric
        # or asymmetric key. jwt lib will choose the right one based on method
        app.config["USE_JWT"] = True
        Initialize(
            app,
            secret=jwt_secret,
            authenticate=authenticate,
            algorithm=jwt_method,
            user_id="username",
        )
    """
   
  
    

    # Initialize shared object of type unsigned int for tracking
    # the number of active training processes
    app.ctx.active_training_processes = multiprocessing.Value("I", 0)
    

    #@app.exception(ErrorResponse)
    #async def handle_error_response(
    #    request: Request, exception: ErrorResponse
    #) -> HTTPResponse:
    #    return response.json(exception.error_info, status=exception.status)


    
    
    @app.get("/version")
    async def version(request: Request) -> HTTPResponse:
        """Respond with the version number of the installed Tileai."""
        from tileai.cli import __version__
        return response.json(
            {
                "version": __version__,
            }
        )

    @app.get("/status")
    #@requires_auth(app, auth_token)
    #@ensure_loaded_agent(app)
    async def status(request: Request) -> HTTPResponse:
        """Respond with the model name and the fingerprint of that model."""
        print(app.ctx)

        if hasattr(app.ctx, "models"):

            return response.json(
                {
                    "model_file": app.ctx.agent.processor.model_filename,
                    "model_id": app.ctx.agent.model_id,
                    "num_active_training_jobs": app.ctx.active_training_processes.value,
                }
            )
        else:
            return response.json(
                {
                    "model_file":"nessun modello ancora presente"
                    
                }
            )
    

    @app.post("/model/enqueuetrain")
    #@requires_auth(app, auth_token)
    @async_callback_url
    async def queuetrain(request: Request) -> HTTPResponse:

       
        validate_request_body(
            request, "You must provide configuration and training data in the request body in order to train your model.")
                
      
        nlu = request.json
        
       
        modelpath = nlu["model"]
        print(modelpath)
        nlu_str = json.dumps(nlu)#.encode('utf-8')
        from tileai.core.model_training import del_old_model
                
        try:
            async with app.ctx.redis as red:
                await del_old_model(redis_conn=red, model=const.MODEL_PATH_ROOT+modelpath)
                res = await red.publish('train',str(nlu_str))
            return response.json({"message":"Task is in queue","n client":res})
        
        except ErrorResponse as e:
            raise e
        
        except Exception as e:
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Queue error",
                f"An unexpected error occurred during training. Error: {e}",
            )
        




    @app.post("/model/train")
    #@requires_auth(app, auth_token)
    @async_callback_url
    @run_in_thread
    async def train(request: Request) -> HTTPResponse:

        
        print(request.json)
    
        validate_request_body(
            request, "You must provide configuration and training data in the request body in order to train your model.")
                

        nlu = request.json
        modelpath = request.json["model"]

        try:
            with app.ctx.active_training_processes.get_lock(): 
                app.ctx.active_training_processes.value += 1

            from tileai.core.model_training import train


            # pass `None` to run in default executor
            training_result = train(nlu, const.MODEL_PATH_ROOT+modelpath)

            if training_result.model:
                filename = os.path.basename(training_result.model)
                
                print(filename)
                print(training_result.performanceindex)
                return response.json({"model":training_result.model}) # da valutare se restituire parametri , "performance":training_result.performanceindex

            else:
                raise ErrorResponse(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "TrainingError",
                    "Ran training, but it finished without a trained model.",
                )
        except ErrorResponse as e:
            raise e
        
        except Exception as e:
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "TrainingError",
                f"An unexpected error occurred during training. Error: {e}",
            )
        finally:
            with app.ctx.active_training_processes.get_lock():
                app.ctx.active_training_processes.value -= 1

    

    @app.post("/model/parse")
    #@requires_auth(app, auth_token)
    #@ensure_loaded_agent(app)
    async def parse(request: Request) -> HTTPResponse:
        validate_request_body(
            request,
            "No text message defined in request_body. Add text message to request body "
            "in order to obtain the intent and extracted entities.",
        )
        
        model = request.json.get("model")
        text = request.json.get("text")

        try:
            from tileai.core.model_training import query, http_query
            if(app.ctx.redis is None):
                label, risult_dict = query(model, text)
            else:
                label, risult_dict = await http_query(redis_conn=app.ctx.redis, model=model, query_text=text)

           

            return response.json(risult_dict)

        except Exception as e:
            print(e)
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "ParsingError",
                f"An unexpected error occurred. Error: {e}",
            )

   
    return app



    #@requires_auth(app, auth_token)


def add_root_route(app: Sanic) -> None:
    """Add '/' route to return hello."""

    @app.get("/")
    async def hello(request: Request) -> HTTPResponse:
        """Check if the server is running and responds with the version."""
        from tileai.cli import __version__
        return response.text("Hello from Tileai: " + __version__)
    
    @app.get("/test1")
    async def testredis(request: Request) -> HTTPResponse:
        """Check if the server is running and responds with the version."""
        async with app.ctx.redis as r:
            await r.set("key1", "value1")
            result = await r.get("key1")
        from sanic.response import text
        return text(str(result))

