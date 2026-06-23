# Codex Agent Engineer Role

Build the agent shell so the model can become useful in VSCode.

Rules:

- All edits via patches.
- Read/search before edit.
- Run tests after edit when possible.
- Never invent files; list/search first.
- Keep tool protocol JSON-serialisable.
- Make model backend swappable.

Initial backend can be mocked or an existing model. Later use TinyCore via tinycored.
