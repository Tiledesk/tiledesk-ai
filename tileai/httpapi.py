from typing import Any, Text, Dict, Union, List, Optional
import json
from sanic import Sanic

from tileai.core.http import server

app = server.create_app()

 


def run(**kwargs: "Dict[Text, Any]") -> None:
    
    import tileai.core.http.httprun
    
    _endpoints = "0.0.0.0"
    
   
    kwargs = tileai.shared.utils.minimal_kwargs(
        kwargs, tileai.core.http.httprun.serve_application
    )
    print("httpapi ",kwargs)
    
    tileai.core.http.httprun.serve_application(
        app=app,
        model_path="models/new",
        endpoints=_endpoints,
        #port=port,
        **kwargs
    )