import tileai.shared.const
import logging
import os
import json

import asyncio
import multiprocessing
import pickle
from time import sleep
from threading import Event

from functools import reduce, wraps
import concurrent.futures

from tileai.shared.exceptions import ErrorResponse

#import tileai.shared.const as const

from queue import Queue

import aiohttp
import jsonschema
from sanic import Sanic, response
from sanic.request import Request
from sanic.response import HTTPResponse
from sanic_cors import CORS
from sanic import Blueprint
from aioredis import from_url
import redis.asyncio as redis
import aioredis

from http import HTTPStatus

from inspect import isawaitable

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

def requires_auth(
    app: Sanic, token: Optional[Text] = None
) -> Callable[["SanicView"], "SanicView"]:
    """Wraps a request handler with token authentication."""

    
    def decorator(f: "SanicView") -> "SanicView":
        def bot_id_from_args(args: Any, kwargs: Any) -> Optional[Text]:
            argnames = rasa.shared.utils.common.arguments_of(f)

            try:
                sender_id_arg_idx = argnames.index("bot_id")
                if "bot_id" in kwargs:  # try to fetch from kwargs first
                    return kwargs["bot_id"]
                if sender_id_arg_idx < len(args):
                    return args[sender_id_arg_idx]
                return None
            except ValueError:
                return None
    
        async def sufficient_scope(
            request: Request, *args: Any, **kwargs: Any
        ) -> Optional[bool]:
            
            jwt_data =   await request.app.ctx.auth.extract_payload(request)
            
            user = jwt_data.get("user", {})

            username = user.get("username", None)
            role = user.get("role", None)

            if role == "admin":
                return True
            elif role == "user":
                bot_id = bot_id_from_args(args, kwargs)
                return bot_id is not None and username == bot_id
            else:
                return False

        @wraps(f)
        async def decorated(
            request: Request, *args: Any, **kwargs: Any
        ) -> response.HTTPResponse:

            #print("USE_JWT",app.config.get("USE_JWT"))
            # noinspection PyProtectedMember
            if app.config.get(
                "USE_JWT"
            ) and await request.app.ctx.auth.is_authenticated(request):
                if await sufficient_scope(request, *args, **kwargs):      
                    result = f(request, *args, **kwargs)
                    return await result if isawaitable(result) else result
                    
                raise ErrorResponse(
                    HTTPStatus.FORBIDDEN,
                    "NotAuthorized",
                    "User has insufficient permissions."
                    
                )
            elif token is None and app.config.get("USE_JWT") is None:
                # authentication is disabled
                result = f(request, *args, **kwargs)
                ret = await result if isawaitable(result) else result
                return ret
            raise ErrorResponse(
                HTTPStatus.UNAUTHORIZED,
                "NotAuthenticated",
                "User is not authenticated."
            )

        return decorated

    return decorator

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
                
                #print(result)
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

async def authenticate(_: Request) -> NoReturn:
    """Callback for authentication failed."""
    raise exceptions.AuthenticationFailed(
        "Direct JWT authentication not supported. You should already have "
        "a valid JWT from an authentication provider, Tileai will just make "
        "sure that the token is valid, but not issue new tokens."
    )

def create_app(
    app : Sanic = None,
    cors_origins: Union[Text, List[Text], None] = "*",
    auth_token: Optional[Text] = None,
    response_timeout: Optional[int] = tileai.shared.const.DEFAULT_RESPONSE_TIMEOUT,
    jwt_secret: Optional[Text] = None,
    jwt_method: Text = "HS256",
    endpoints: Optional[Text] = None,
    port: Optional[int] = tileai.shared.const.DEFAULT_SERVER_PORT,
    redis_url : Optional[Text] = None,
) -> Sanic:
    """Class representing a Tileai HTTP server."""

   
    app = Sanic(name="tileai_server")
    app.config.RESPONSE_TIMEOUT = response_timeout
    configure_cors(app, cors_origins)

   

    
    
   
  
    

    # Initialize shared object of type unsigned int for tracking
    # the number of active training processes
    app.ctx.active_training_processes = multiprocessing.Value("I", 0)
    add_root_route(app)
    protocol = "http"

    #@app.exception(ErrorResponse)
    #async def handle_error_response(
    #    request: Request, exception: ErrorResponse
    #) -> HTTPResponse:
    #    return response.json(exception.error_info, status=exception.status)

    


    async def reader(channel: aioredis.client.Redis):    
        while True:
            try:
                messages = await channel.xreadgroup(
                    groupname=tileai.shared.const.STREAM_CONSUMER_GROUP,
                    consumername=tileai.shared.const.STREAM_CONSUMER_NAME,
                    streams={tileai.shared.const.STREAM_NAME: '>'},
                    count=1,
                    block=0  # Set block to 0 for non-blocking
                )
                

                for stream, message_data in messages:
                    for message in message_data:
                        message_id, message_values = message
                        #print(f"Received message {message_id}: {message_values}")
                        import ast
                        

                        byte_str = message_values[b"train"]

                        dict_str = byte_str.decode("UTF-8")
                        nlu_json = ast.literal_eval(dict_str)
                        #nlu_msg = message_values
                    
                        #nlu_str = nlu_msg.decode('utf-8')
                    
                        #nlu_json = mydata["train"]#json.loads(nlu_str)#.decode('utf-8')
                        #print(nlu_json)
                        modelpath = nlu_json["model"]
                        base_url=f"{protocol}://localhost:{port}"
                        url_post = base_url+ "/model/train"
                        # A POST request to tthe API
                        
                        
                        async with aiohttp.ClientSession() as session:
                            
                            async with app.shared_ctx.redis as red:
                                await red.set(modelpath, "running")
                                print("task runninng", modelpath)
                            
                            async with session.post(url_post,  json = nlu_json) as response:
                                post_response = await response.text()
                                #print(post_response)
                            
                            #async with app.ctx.redis as red:
                            #    await red.set(modelpath, "completedaaaa")
                            #    print("task completed", modelpath)

                            #########
                            #async with app.ctx.redis as red:
                                #await red.set(modelpath, "error")
                                #print("task error", modelpath)
                    
                    

                    """
                    
                    async with app.ctx.redis as red:
                        await red.set(modelpath, "running")
                        print("task runninng", modelpath)
                        
                    import tileai.core.http.trainutil as trainutil

                    result = await trainutil.train(nlu_json)
                    

                    async with app.ctx.redis as red:
                        await red.set(modelpath, "competed")
                        #print("task completed", modelpath)
                    
                    ##### INVIO post su webhook se esiste webhook altrimenti nulla
                   
                    payload: Dict[Text, Any] = dict(
                        data=json.dumps(result.result()), headers={"Content-Type": "application/json"}
                    )

                    if "webhook_url" in nlu_json.keys():
                        webhook_url = nlu_json["webhook_url"]    
                        async with aiohttp.ClientSession() as session:
                            await session.post(webhook_url, raise_for_status=True, **payload)  

                    """
                    
          
          
                    #post_response = await asyncio.post(url_post, json=result.result(), content_type="application/json")
            except ErrorResponse as e:
                if "webhook_url" in nlu_json.keys():
                    webhook_url = nlu_json["webhook_url"]
                    payload = dict(json=e.error_info) 
                    async with aiohttp.ClientSession() as session:
                            await session.post(webhook_url, raise_for_status=True, **payload)  
                
                async with app.shared_ctx.redis as red:
                    await red.set(modelpath, "error")
                    print("task error", modelpath)       
            
            except Exception as e:
                if "webhook_url" in nlu_json.keys():
                    webhook_url = nlu_json["webhook_url"]
                    payload = dict(json=e.error_info) 
                    async with aiohttp.ClientSession() as session:
                            await session.post(webhook_url, raise_for_status=True, **payload)  
                #print(e)
                async with app.shared_ctx.redis as red:
                    await red.set(modelpath, "error")
                    print("task error", modelpath)
                    # Process the message here

                    # Acknowledge the message to remove it from the stream
                    await channel.xack(
                        tileai.shared.const.STREAM_NAME, 
                        tileai.shared.const.STREAM_CONSUMER_GROUP, 
                        message_id)

               
                
                
                
           
                
    

           
        
    @app.listener("main_process_start")
    async def main_process_start(app):
        #await setup_redis(app)
        #await setup_jwtsecret_url(app,loop)
        #await redis_consumer(app,loop) 
        print("listener_1")
                
        
    @app.listener('after_server_start') 
    async def redis_consumer(_app:Sanic, _loop):
        #print("redis consumer")
        
        sub = _app.shared_ctx.redis
        try:
            await sub.xgroup_create(
                tileai.shared.const.STREAM_NAME, 
                tileai.shared.const.STREAM_CONSUMER_GROUP, 
                id="0", 
                mkstream=True)
        except aioredis.exceptions.ResponseError as e:
            if "BUSYGROUP Consumer Group name already exists" not in str(e):
                raise
            
        #setattr(app.shared_ctx, "stream", stream)
        #await sub.subscribe('train')    
        future_reader = asyncio.create_task(reader(sub)) 
        await future_reader
    
    @app.listener("before_server_start")
    async def setup_jwtsecret_url(_app: Sanic, _loop):
        setattr(_app.ctx, "jwt_secret", jwt_secret)
        
    @app.before_server_start
    async def setup_redis(app):
        #import multiprocessing
        #import pickle
        #print("1")
        if not redis_url:
            raise ValueError("You must specify a redis_url or set the  Sanic config variable")
        logger.info("[sanic-redis] connecting")
        _redis = await from_url(redis_url)
        #app.shared_ctx.redis= multiprocessing.RawArray('c', pickle.dumps(_redis))
        setattr(app.shared_ctx, "redis", _redis)
        setattr(app.shared_ctx, "redis_url", redis_url)
        
        #conn = _redis
    
   
    @app.listener('after_server_stop')
    async def close_redis(_app, _loop):
        logger.info("[sanic-redis] closing")
        await app.shared_ctx.redis.close()
        
        
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
















    @app.exception(ErrorResponse)
    async def handle_error_response(
        request: Request, exception: ErrorResponse
    ) -> HTTPResponse:
        #multiprocessing.RawArray('c', pickle.dumps(exception.error_info))
        return response.json(exception.error_info, status=exception.status)
    
    
    @app.get("/version")
    async def version(request: Request) -> HTTPResponse:
        """Respond with the version number of the installed Tileai."""
        from tileai.cli import __version__
        return response.json(
            {
                "version": __version__,
            }
        )

    @app.get("/status/<modelid>")
    @requires_auth(app, auth_token)
    async def status(request: Request, modelid:None) -> HTTPResponse:
        """Respond with the model name and the fingerprint of that model."""
        
        async with app.shared_ctx.redis as red:
                modelstatus = await red.get(modelid)
        if modelstatus: #hasattr(app.ctx, "models"):
            #return response.json(
            #    {
            #        "model_file": app.ctx.agent.processor.model_filename,
            #        "model_id": app.ctx.agent.model_id,
            #        "num_active_training_jobs": app.ctx.active_training_processes.value,
            #    }
            #)
            return response.json(
                {
                    "model": modelid,
                    "status": modelstatus.decode('utf-8')
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
            async with app.shared_ctx.redis as red:
                
                await del_old_model(redis_conn=red, model=tileai.shared.const.MODEL_PATH_ROOT+modelpath)
                
                await red.set(modelpath, "equeued")
                print("task enqueued", modelpath)
                #res = await red.publish('train',str(nlu_str))
                res = await red.xadd(tileai.shared.const.STREAM_NAME, {"train":nlu_str} , id="*")
                
            return response.json({"message":"Task is in queue","n client":1})
        
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

        
        #print(request.json)
    
        validate_request_body(
            request, "You must provide configuration and training data in the request body in order to train your model.")
                

        nlu = request.json
        modelpath = request.json["model"]

        try:
            with app.ctx.active_training_processes.get_lock(): 
                app.ctx.active_training_processes.value += 1

            from tileai.core.model_training import train
           
            training_result = train(nlu, tileai.shared.const.MODEL_PATH_ROOT+modelpath)
         
            if training_result.model:
                filename = os.path.basename(training_result.model)
                
                print(filename)
                print(training_result.performanceindex)
                _redis = await from_url(app.shared_ctx.redis_url)
                await _redis.set(modelpath, "completed")
                print("task completed", modelpath)
               
                return response.json({"model":training_result.model}) # da valutare se restituire parametri , "performance":training_result.performanceindex

            else:
            
                raise ErrorResponse(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "TrainingError",
                    "Ran training, but it finished without a trained model.",
                )
        except ErrorResponse as e:
            _redis = await from_url(app.shared_ctx.redis_url)
            await _redis.set(modelpath, "error")
                
            raise e
        
        except Exception as e:
           
            print(e)
            _redis = await from_url(app.shared_ctx.redis_url)
            await _redis.set(modelpath, "error")
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
            if(app.shared_ctx.redis is None):
                label, risult_dict = query(model, text)
            else:
                label, risult_dict = await http_query(redis_conn=app.shared_ctx.redis, model=model, query_text=text)

           

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

