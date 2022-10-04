# tiledesk-ai
Tiledesk Module for AI

This module uses a simple Feed Forward Network implemented using PyTorch (more to come in the future) to understand the user intent.


## Use with the command line

> NOTE: recommended Python version >= 3.9

### Install

We recommend to install a python3 virtual env and install tiledesk-ai on it.
 ```
 pip install virtualenv
 python3 -m venv tileai
 source ./tileai/bin/activate
 ```

Create your working folder:

```
mkdir tiledeskai
cd tiledeskai
```

Now synchronize with the source repo:

```
git clone https://github.com/Tiledesk/tiledesk-ai.git
cd tiledesk-ai
pip install -r requirements.txt
```

For developement, use:
```
python setup.py develop
```
This command will install the in-place the program. You can edit the script files and test them from the command line.

> **NOTE**: **PRODUCTION ENVIRONMENT** If are not interested in customize the code to improve/modify the module you can just
> use the production command:
>
> **python setup.py install**


### nlu.json

```json
{
	"configuration": {
		"algo": "auto"
	},
	"nlu": [{	
      "intent":"hello_intent",
      "examples":["Hi","Hello", "..."]
	},

	]
}
```

### Train
To train the model use the *tileai* command.

*tileai* command synthax:

```
> tileai train [-f nlu_filepath] [-o model_file_path]
```

*nlu_filepath* defaults to local */domain/nlu.json* file.

Train example:

> tileai train -f domain/nlu.json -o models/my_trained_model

### Query

```shell
> tileai query [-m model path] -t "question"
```

## HTTP server

### Run the HTTP server

You can run the tiledesk-ai module as a web app, launching the HTTP server.
Default HTTP server port is 6006. You can change the port using the _-p port_ option.

```shell
> tileai run [-p port]
```

### Query from HTTP server
To train your model from http server:
```
POST http://localhost:port/train
```
```json
{
	"configuration": {
		"algo": "auto"
	},
	"nlu": [{	
      "intent":"hello_intent",
      "examples":["Hi","Hello", "..."]
	},

	]
}
```

To query your model
```shell
POST http://localhost:port/model/parse
```
```json
{
"model":"models/<name of the model>",
  "text":"..."
}
```

## APIs

TODO
