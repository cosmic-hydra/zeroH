import * as path from "path";
import * as vscode from "vscode";
import { BridgeResponse, ZerohBridge } from "./bridge";

let output: vscode.OutputChannel;

export function activate(context: vscode.ExtensionContext): void {
  output = vscode.window.createOutputChannel("zeroH");
  const bridge = new ZerohBridge(context);

  const register = (id: string, fn: () => Promise<void>) =>
    context.subscriptions.push(
      vscode.commands.registerCommand(id, () =>
        fn().catch((err) =>
          vscode.window.showErrorMessage(`zeroH: ${String(err)}`)
        )
      )
    );

  register("zeroh.remember", () => remember(bridge));
  register("zeroh.ingestActiveFile", () => ingestActiveFile(bridge));
  register("zeroh.verifySelection", () => verifySelection(bridge));
  register("zeroh.recall", () => recall(bridge));
  register("zeroh.answerFromMemory", () => answerFromMemory(bridge));
  register("zeroh.complete", () => complete(bridge));

  context.subscriptions.push(output);
}

export function deactivate(): void {
  /* nothing to clean up: the bridge spawns short-lived processes */
}

/** Run a bridge command inside a progress notification, surfacing errors. */
async function run(
  title: string,
  fn: () => Promise<BridgeResponse>
): Promise<BridgeResponse | undefined> {
  return vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title },
    async () => {
      const res = await fn();
      if (!res.ok) {
        output.appendLine(`[error] ${res.code ?? "error"}: ${res.error}`);
        vscode.window.showErrorMessage(`zeroH: ${res.error}`);
        return undefined;
      }
      return res;
    }
  );
}

async function remember(bridge: ZerohBridge): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  const selected = editor?.document.getText(editor.selection).trim();
  const content = await vscode.window.showInputBox({
    prompt: "Fact to remember (stored durably in zeroH memory)",
    value: selected || "",
    ignoreFocusOut: true,
  });
  if (!content) {
    return;
  }
  const res = await run("zeroH: remembering…", () =>
    bridge.call("remember", { content, source: "vscode" })
  );
  if (res) {
    vscode.window.showInformationMessage("zeroH: fact remembered.");
  }
}

async function ingestActiveFile(bridge: ZerohBridge): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage("zeroH: no active editor to ingest.");
    return;
  }
  const document = editor.document.getText();
  if (!document.trim()) {
    vscode.window.showWarningMessage("zeroH: the active file is empty.");
    return;
  }
  const source = path.basename(editor.document.fileName) || "document";
  const res = await run("zeroH: ingesting document…", () =>
    bridge.call("ingest", { document, source })
  );
  if (res) {
    vscode.window.showInformationMessage(
      `zeroH: ingested ${res.result.chunks} chunk(s) from ${source}.`
    );
  }
}

async function verifySelection(bridge: ZerohBridge): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  const text = editor?.document.getText(editor.selection).trim();
  if (!text) {
    vscode.window.showWarningMessage("zeroH: select some text to verify.");
    return;
  }
  const res = await run("zeroH: verifying against memory…", () =>
    bridge.call("verify", { text })
  );
  if (res) {
    showAnswer("Verify", res.result);
  }
}

async function recall(bridge: ZerohBridge): Promise<void> {
  const query = await vscode.window.showInputBox({
    prompt: "Recall memories for…",
    ignoreFocusOut: true,
  });
  if (!query) {
    return;
  }
  const res = await run("zeroH: recalling…", () =>
    bridge.call("recall", { query })
  );
  if (!res) {
    return;
  }
  const results: Array<{ content: string; source: string; score: number }> =
    res.result.results || [];
  if (results.length === 0) {
    vscode.window.showInformationMessage("zeroH: no relevant memories found.");
    return;
  }
  const picked = await vscode.window.showQuickPick(
    results.map((r) => ({
      label: r.content,
      description: `score ${r.score} · ${r.source}`,
    })),
    { placeHolder: `Top ${results.length} memories for "${query}"` }
  );
  if (picked) {
    output.show(true);
    output.appendLine(`[recall] ${picked.label}`);
  }
}

async function answerFromMemory(bridge: ZerohBridge): Promise<void> {
  const query = await vscode.window.showInputBox({
    prompt: "Ask (answered extractively from memory, no LLM)",
    ignoreFocusOut: true,
  });
  if (!query) {
    return;
  }
  const res = await run("zeroH: answering from memory…", () =>
    bridge.call("answer", { query })
  );
  if (res) {
    showAnswer("Answer From Memory", res.result);
  }
}

async function complete(bridge: ZerohBridge): Promise<void> {
  const query = await vscode.window.showInputBox({
    prompt: "Ask (grounded completion via your configured LLM)",
    ignoreFocusOut: true,
  });
  if (!query) {
    return;
  }
  const res = await run("zeroH: generating grounded answer…", () =>
    bridge.call("complete", { query }, { withLlm: true })
  );
  if (res) {
    showAnswer("Grounded Answer", res.result);
  }
}

/** Render an Answer payload to the output channel. */
function showAnswer(label: string, answer: any): void {
  output.show(true);
  output.appendLine("");
  output.appendLine(`=== zeroH · ${label} ===`);
  output.appendLine(answer.text || "");
  output.appendLine(
    `grounded=${answer.grounded} · confidence=${answer.confidence} · ` +
      `abstained=${answer.abstained}`
  );
  const citations = answer.citations || [];
  if (citations.length > 0) {
    output.appendLine("citations:");
    for (const c of citations) {
      output.appendLine(`  - (${c.source}, ${c.score}) ${c.content}`);
    }
  }
  if (answer.abstained) {
    vscode.window.showWarningMessage(
      "zeroH abstained: memory does not support an answer."
    );
  } else {
    vscode.window.showInformationMessage(`zeroH (${label}): see output panel.`);
  }
}
