import * as vscode from 'vscode';
import { TinycoredClient } from './agentClient.js';

export function activate(context: vscode.ExtensionContext): void {
  const health = vscode.commands.registerCommand('tinycore.health', async () => {
    const client = makeClient();
    const result = await client.health();
    vscode.window.showInformationMessage(`TinyCore server: ${result.ok ? 'ok' : 'not ok'} (${result.runtime})`);
  });

  const ask = vscode.commands.registerCommand('tinycore.ask', async () => {
    const prompt = await vscode.window.showInputBox({ prompt: 'Ask TinyCore' });
    if (!prompt) {
      return;
    }
    const client = makeClient();
    const result = await client.chat({
      messages: [{ role: 'user', content: prompt }],
      ...generationSettings('askMaxTokens'),
    });
    if (result.type === 'message') {
      vscode.window.showInformationMessage(trimForMessage(result.content));
    } else {
      vscode.window.showInformationMessage(`TinyCore tool result: ${result.ok ? 'ok' : 'failed'}`);
    }
  });

  const explain = vscode.commands.registerCommand('tinycore.explainSelection', async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showWarningMessage('Open a file and select code first.');
      return;
    }
    const selection = editor.document.getText(editor.selection);
    if (!selection) {
      vscode.window.showWarningMessage('Select code to explain.');
      return;
    }
    const client = makeClient();
    const result = await client.chat({
      messages: [
        { role: 'system', content: 'Explain the selected code concisely.' },
        { role: 'user', content: selection },
      ],
      ...generationSettings('explainMaxTokens'),
    });
    vscode.window.showInformationMessage(result.type === 'message' ? trimForMessage(result.content) : 'TinyCore returned a tool result.');
  });

  context.subscriptions.push(health, ask, explain);
}

export function deactivate(): void {}

function makeClient(): TinycoredClient {
  const config = vscode.workspace.getConfiguration('tinycore');
  return new TinycoredClient(config.get<string>('serverUrl', 'http://127.0.0.1:8787'));
}

function generationSettings(maxTokenKey: 'askMaxTokens' | 'explainMaxTokens'): {
  max_tokens: number;
  temperature: number;
  top_k: number;
  seed: number;
} {
  const config = vscode.workspace.getConfiguration('tinycore');
  return {
    max_tokens: config.get<number>(maxTokenKey, maxTokenKey === 'askMaxTokens' ? 160 : 200),
    temperature: config.get<number>('temperature', 0),
    top_k: config.get<number>('topK', 0),
    seed: config.get<number>('seed', 1337),
  };
}

function trimForMessage(content: string): string {
  return content.length > 240 ? `${content.slice(0, 237)}...` : content;
}
