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

## APIs

### Run web app

Default port is 6006

```shell
> tileai run [-p port]
```

### Query from web app

```shell
> http://localhost?query=who%20are%20you
```

