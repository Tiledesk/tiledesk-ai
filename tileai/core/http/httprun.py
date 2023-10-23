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
from tileai.shared.exceptions import ErrorResponse


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
            try:
                message = await channel.get_message(ignore_subscribe_messages=True, timeout=0.1)
                if message is not None and isinstance(message, dict) and message.get('type') == 'message':
                    nlu_msg = message.get('data')
                
                    nlu_str = nlu_msg.decode('utf-8')
                
                    nlu_json = json.loads(nlu_str)#.decode('utf-8')
                    print(nlu_json)
                    modelpath = nlu_json["model"]
                    base_url=f"{protocol}://localhost:{port}"
                    url_post = base_url+ "/model/train"
                    # A POST request to tthe API
                    
                    
                    async with aiohttp.ClientSession() as session:
                        
                        async with app.ctx.redis as red:
                            await red.set(modelpath, "running")
                            print("task runninng", modelpath)
                        
                        async with session.post(url_post,  json = nlu_json) as response:
                            post_response = await response.text()
                            print("==================AAAAAAAAA")
                            print(post_response)
                            print("==================AAAAAAA")
                            
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
                
                async with app.ctx.redis as red:
                    await red.set(modelpath, "error")
                    print("task error", modelpath)       
            
            except Exception as e:
                if "webhook_url" in nlu_json.keys():
                    webhook_url = nlu_json["webhook_url"]
                    payload = dict(json=e.error_info) 
                    async with aiohttp.ClientSession() as session:
                            await session.post(webhook_url, raise_for_status=True, **payload)  
                print(e)
                async with app.ctx.redis as red:
                    await red.set(modelpath, "error")
                    print("task error", modelpath)
            
                



                
        
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
        setattr(_app.ctx, "redis_url", redis_url)
        conn = _redis
    
   


   


    @app.listener('after_server_stop')
    async def close_redis(_app, _loop):
        logger.info("[sanic-redis] closing")
        await app.ctx.redis.close()

    
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

    
   




