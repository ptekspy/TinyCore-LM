import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
  const ask = vscode.commands.registerCommand('tinycore.ask', async () => {
    const prompt = await vscode.window.showInputBox({ prompt: 'Ask TinyCore' });
    if (!prompt) return;
    const editor = vscode.window.activeTextEditor;
    const selection = editor ? editor.document.getText(editor.selection) : '';
    // TODO: send prompt + selection + workspace context to tinycored.
    vscode.window.showInformationMessage('TinyCore request queued locally.');
  });

  const explain = vscode.commands.registerCommand('tinycore.explainSelection', async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;
    const selection = editor.document.getText(editor.selection);
    // TODO: call tinycored /chat endpoint.
    console.log(selection);
  });

  context.subscriptions.push(ask, explain);
}

export function deactivate() {}
