import asyncio
import aiohttp
import logging
import uuid
import os
from functools import partial
from typing import Any, List, Optional, Text, Union, Dict
import concurrent.futures

import json
import requests


import tileai.shared.const
#from tileai.core.http import server


from sanic import Sanic
from asyncio import AbstractEventLoop

from aioredis import from_url
import redis.asyncio as redis
from tileai.core.http.server  import authenticate

from sanic_jwt import Initialize, exceptions

logger = logging.getLogger()  # get the root logger




def serve_application(
    app : Sanic = None,
    model_path: Optional[Text] = None,
    endpoints: Optional[Text] = None,
    port: int = tileai.shared.const.DEFAULT_SERVER_PORT,
    redis_url : Optional[Text] = None,
    jwt_secret : Optional[Text] = None,
    jwt_method: Text = "HS256",
    cors: Optional[Union[Text, List[Text]]] = None,
    auth_token: Optional[Text] = None,
    response_timeout: int = tileai.shared.const.DEFAULT_RESPONSE_TIMEOUT,
    request_timeout: Optional[int] = None,
    
) -> None:
    
    #print(model_path)
    #app = configure_app(
    #    app,
    #    cors,
    #    auth_token,
    #    response_timeout,
    #    port=port,
    #    endpoints=endpoints,
    #    request_timeout=request_timeout,
    #)

    #app.register_listener(
    #    partial(load_agent_on_start, model_path, endpoints, remote_storage),
    #    "before_server_start",
    #)
    async def reader(channel: redis.client.PubSub):
        while True:
        #thread = sub.run_in_thread(sleep_time=0.001)
            message = await channel.get_message(ignore_subscribe_messages=True, timeout=0.1)
            if message is not None and isinstance(message, dict) and message.get('type') == 'message':
                nlu_msg = message.get('data')
                
                nlu_str = nlu_msg.decode('utf-8')
                
                nlu_json = json.loads(nlu_str)#.decode('utf-8')
                print(nlu_json)
                base_url=f"{protocol}://localhost:{port}"
                url_post = base_url+ "/model/train"
                # A POST request to tthe API
                async with aiohttp.ClientSession() as session:
                    async with session.post(url_post,  json = nlu_json) as response:
                        post_response = await response.text()
          
          
                #post_response = await asyncio.post(url_post, json=nlu_str, content_type="application/json")
                print(post_response)


                
        
    @app.listener('after_server_start') 
    async def redis_consumer(_app:Sanic, _loop):
        sub = _app.ctx.redis.pubsub()
        await sub.subscribe('train')    
        future_reader = asyncio.create_task(reader(sub)) 
        #await future_reader
    
    @app.listener("before_server_start")
    async def setup_jwtsecret_url(_app: Sanic, _loop):
        setattr(_app.ctx, "jwt_secret", jwt_secret)
        
    @app.listener("before_server_start")
    async def setup_redis(_app: Sanic, _loop):
        if not redis_url:
            raise ValueError("You must specify a redis_url or set the {} Sanic config variable".format(config_name))
        logger.info("[sanic-redis] connecting")
        _redis = await from_url(redis_url)
        setattr(_app.ctx, "redis", _redis)
        conn = _redis

   


    @app.listener('after_server_stop')
    async def close_redis(_app, _loop):
        logger.info("[sanic-redis] closing")
        await app.ctx.redis.close()

    
    print("HTTPRUN", jwt_secret)
    
    
    protocol = "http"
    interface = tileai.shared.const.DEFAULT_SERVER_INTERFACE

    print(f"Starting Tileai server on {protocol}://{interface}:{port}")

    import multiprocessing
    workers = multiprocessing.cpu_count()

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




    app.run(
        host=interface,
        port=port,
        debug=True,
        #dev=True,
        auto_reload=False,
        workers=1,
        single_process=False
       
    )

    
   




