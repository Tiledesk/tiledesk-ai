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
   

    
    app = server.create_app(
        cors_origins=cors,
        auth_token=auth_token,
        response_timeout=response_timeout,
        endpoints=endpoints,
        )
  
#Aggiungere la capacità di loggare vedi l'approccio di rasa
    


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

   
    
    print(interface, port)
    app.run(
        host=interface,
        port=port,
        
        
    )


