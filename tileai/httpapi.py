from typing import Any, Text, Dict, Union, List, Optional
import json
#from sanic import Sanic

from tileai.core.http import server

#app = server.create_app()

 


def run(**kwargs: "Dict[Text, Any]") -> None:

    #app = server.create_app()

    import tileai.core.http.httprun
    
    _endpoints = "0.0.0.0"
    
   
    kwargs = tileai.shared.utils.minimal_kwargs(
        kwargs, tileai.core.http.httprun.serve_application
    )
    print("httpapi ",kwargs)
    
    #tileai.core.http.httprun.serve_application(
    #    app=app,
    #    model_path="models/new",
    #    endpoints=_endpoints,
    #    #port=port,
    #    **kwargs
    #)
    
    import multiprocessing
    import pickle
    workers = multiprocessing.cpu_count()
    
    from sanic.worker.loader import AppLoader
    from sanic import Sanic
    from functools import partial
    import tileai
    
    loader = AppLoader(factory =partial(server.create_app, **kwargs))
    app = loader.load()
    app.prepare(host='0.0.0.0',port=5100, debug=True, auto_reload=False, workers=workers,single_process=False)
    Sanic.serve(primary=app, app_loader=loader)