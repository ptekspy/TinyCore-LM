# Runtime / Server API Spec

## Components

```txt
VSCode Extension
  -> tinycored local server
  -> Python model server initially
  -> tinycore.cpp runtime later
```

## Local server endpoints

### Health

```http
GET /health
```

Response:

```json
{"ok":true,"model_loaded":true,"runtime":"python|native"}
```

### Generate

```http
POST /generate
```

Request:

```json
{
  "prompt": "...",
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.95,
  "stop": ["</assistant>"]
}
```

Response:

```json
{"text":"...","tokens":[],"metrics":{"tokens_per_sec":0}}
```

### Chat

```http
POST /chat
```

Request:

```json
{
  "messages": [
    {"role":"system","content":"..."},
    {"role":"user","content":"..."}
  ],
  "tools": []
}
```

### Agent step

```http
POST /agent/step
```

Returns either final answer or tool call.

```json
{
  "type": "tool_call",
  "tool": "read_file",
  "args": {"path":"src/index.ts"}
}
```

## Streaming

Use WebSocket or SSE later:

```txt
/generate/stream
/chat/stream
```

## Server implementation policy

MVP can wrap any existing model to build agent shell. TinyCore model integration becomes mandatory after Python model generates.
