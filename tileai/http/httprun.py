import asyncio
import logging
import uuid
import os
from functools import partial
from typing import Any, List, Optional, Text, Union, Dict



import shared.const
from tileai import server


from sanic import Sanic
from asyncio import AbstractEventLoop

logger = logging.getLogger()  # get the root logger




def configure_app(
    cors: Optional[Union[Text, List[Text], None]] = None,
    auth_token: Optional[Text] = None,
    response_timeout: int = shared.const.DEFAULT_RESPONSE_TIMEOUT,
    port: int = shared.const.DEFAULT_SERVER_PORT,
    endpoints: Optional[Text] = None,
    log_file: Optional[Text] = None,
    request_timeout: Optional[int] = None,
) -> Sanic:
    """Run the agent."""
    #aggiunta del log per l'http server
    #rasa.core.utils.configure_file_logging(
    #    logger, log_file, use_syslog, syslog_address, syslog_port, syslog_protocol
    #)

    
    app = server.create_app(
        cors_origins=cors,
        auth_token=auth_token,
        response_timeout=response_timeout,
        endpoints=endpoints,
        )
  
#Aggiungere la capacità di loggare vedi l'approccio di rasa
    #if logger.isEnabledFor(logging.DEBUG):
        #rasa.core.utils.list_routes(app)

    #async def configure_async_logging() -> None:
    #    if logger.isEnabledFor(logging.DEBUG):
    #        rasa.utils.io.enable_async_loop_debugging(asyncio.get_event_loop())

    #app.add_task(configure_async_logging)


    return app


def serve_application(
    model_path: Optional[Text] = None,
    endpoints: Optional[Text] = None,
    port: int = shared.const.DEFAULT_SERVER_PORT,
    cors: Optional[Union[Text, List[Text]]] = None,
    auth_token: Optional[Text] = None,
    response_timeout: int = shared.const.DEFAULT_RESPONSE_TIMEOUT,
    request_timeout: Optional[int] = None,
    
) -> None:
    
    print(model_path)
    app = configure_app( 
        cors,
        auth_token,
        response_timeout,
        port=port,
        endpoints=endpoints,
        request_timeout=request_timeout,
    )

    protocol = "http"
    interface = shared.const.DEFAULT_SERVER_INTERFACE

    print(f"Starting Tileai server on {protocol}://{interface}:{port}")
######
   
    #app.register_listener(
    #    partial(load_agent_on_start, model_path, endpoints, remote_storage),
    #    "before_server_start",
    #)
    #app.register_listener(close_resources, "after_server_stop")

    #number_of_workers = rasa.core.utils.number_of_sanic_workers(
    #    endpoints.lock_store if endpoints else None
    #)

    

    #rasa.utils.common.update_sanic_log_level(
    #    log_file, use_syslog, syslog_address, syslog_port, syslog_protocol
    #)
    print(interface, port)
    app.run(
        host=interface,
        port=port,
        
        
    )


