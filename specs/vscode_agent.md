# VSCode Agent Spec

## Purpose

`TinyCore Code` is the editor interface for TinyCore-LM and the agent server.

## MVP features

Commands:

```txt
TinyCore: Ask
TinyCore: Explain Selected Code
TinyCore: Fix Current Error
TinyCore: Generate Tests
TinyCore: Apply Suggested Patch
TinyCore: Open Agent Chat
```

Views:

```txt
Chat webview
Diff review panel
Run log/output panel
Model status indicator
```

## Extension architecture

```txt
extension.ts
  -> command registry
  -> chat panel
  -> agent client
  -> diff provider
  -> workspace context provider
```

## Agent communication

Extension talks to local `tinycored` server over HTTP/WebSocket.

## Repo context provider

Collect:

```txt
workspace root
open file
selection
diagnostics
git diff
package manager
test scripts
```

Do not send entire repo by default. Use search/index tools.

## Human review

Even though the user may want high autonomy, the extension must support accept/reject diffs.
