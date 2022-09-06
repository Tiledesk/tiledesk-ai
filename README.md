# tiledesk-ai
Tiledesk Module for AI

This module uses a simple Feed Forward Network implemented using PyTorch (more to come in the future) to understand the user intent.

## Use with the command line

### Install

```
pip install -r requirements.txt
```

```
>python setup.py install
```
For developement, use:
```
>python setup.py develop
```
to install in-place the program. You can edit the script files and test them from the command line. 


### nlu.json

```json
{
	"configuration": {
		"algo": "auto"
	},
	"nlu": [{	
      "intent":"hello_intent",
      "examples":["Hi","Hello", "..."]
	},...

	]
}
```

### Train
Defaults to local file named nlu.json

```
> tileai train [-f filepath] [-o output model]
```
### Query

```shell
> tileai query -q "who are you"
```

## HTTP server

### Run the HTTP server

You can run the tiledesk-ai module as a web app, launching the HTTP server.
Default HTTP server port is 6006. You can change the port using the _-p port_ option.

```shell
> tileai run [-p port]
```

### Query from HTTP server

```shell
> http://localhost?query=who%20are%20you
```

## APIs

TODO
