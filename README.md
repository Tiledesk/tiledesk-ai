# tiledesk-ai
Tiledesk Module for AI

## Use with the command line

### nlu.json

```json
{
	"configuration": {
		"algo": "auto"
	},
	"intents": []
}
```

### Train
Defaults to local file named nlu.json

```
> tileai train [-f filepath]
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
