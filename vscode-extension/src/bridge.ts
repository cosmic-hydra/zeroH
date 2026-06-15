import * as path from "path";
import { spawn } from "child_process";
import * as vscode from "vscode";

/** A successful or failed response from the Python bridge. */
export interface BridgeResponse {
  ok: boolean;
  result?: any;
  error?: string;
  code?: string;
}

/** Resolved configuration read from the user's settings. */
interface ZerohConfig {
  pythonPath: string;
  databasePath: string;
  topK: number;
  llm: {
    provider: string;
    model: string;
    baseUrl: string;
    apiKey: string;
  };
}

function readConfig(): ZerohConfig {
  const cfg = vscode.workspace.getConfiguration("zeroh");
  return {
    pythonPath: cfg.get<string>("pythonPath", "python") || "python",
    databasePath: cfg.get<string>("databasePath", "") || "",
    topK: cfg.get<number>("topK", 5),
    llm: {
      provider: cfg.get<string>("llm.provider", "openai") || "openai",
      model: cfg.get<string>("llm.model", "gpt-4o-mini") || "gpt-4o-mini",
      baseUrl: cfg.get<string>("llm.baseUrl", "") || "",
      apiKey: cfg.get<string>("llm.apiKey", "") || "",
    },
  };
}

/**
 * Drives the bundled `python/zeroh_bridge.py` script. One process is spawned per
 * command; the request is written to stdin and a single JSON response is read
 * back from stdout, keeping the integration simple and stateless.
 */
export class ZerohBridge {
  private readonly scriptPath: string;

  constructor(private readonly context: vscode.ExtensionContext) {
    this.scriptPath = context.asAbsolutePath(
      path.join("python", "zeroh_bridge.py")
    );
  }

  /** Resolve the durable SQLite store path, defaulting to global storage. */
  private databasePath(cfg: ZerohConfig): string {
    if (cfg.databasePath) {
      return cfg.databasePath;
    }
    const dir = this.context.globalStorageUri.fsPath;
    return path.join(dir, "zeroh-memory.db");
  }

  private async ensureStorageDir(): Promise<void> {
    await vscode.workspace.fs.createDirectory(this.context.globalStorageUri);
  }

  /** Send a command to the bridge and resolve with its parsed response. */
  async call(
    command: string,
    payload: Record<string, unknown> = {},
    options: { withLlm?: boolean } = {}
  ): Promise<BridgeResponse> {
    await this.ensureStorageDir();
    const cfg = readConfig();

    const request: Record<string, unknown> = {
      command,
      db: this.databasePath(cfg),
      topK: cfg.topK,
      ...payload,
    };
    if (options.withLlm) {
      request.llm = cfg.llm;
    }

    return this.spawnBridge(cfg.pythonPath, JSON.stringify(request));
  }

  private spawnBridge(pythonPath: string, requestJson: string): Promise<BridgeResponse> {
    return new Promise((resolve) => {
      let child;
      try {
        child = spawn(pythonPath, [this.scriptPath], {
          stdio: ["pipe", "pipe", "pipe"],
        });
      } catch (err) {
        resolve({
          ok: false,
          error: `Failed to start Python ('${pythonPath}'): ${String(err)}`,
          code: "spawn-failed",
        });
        return;
      }

      let stdout = "";
      let stderr = "";

      child.on("error", (err) => {
        resolve({
          ok: false,
          error:
            `Failed to start Python ('${pythonPath}'). Set 'zeroh.pythonPath' ` +
            `in settings. Details: ${err.message}`,
          code: "spawn-failed",
        });
      });

      child.stdout.on("data", (d) => (stdout += d.toString()));
      child.stderr.on("data", (d) => (stderr += d.toString()));

      child.on("close", () => {
        const trimmed = stdout.trim();
        if (!trimmed) {
          resolve({
            ok: false,
            error: stderr.trim() || "The zeroH bridge returned no output.",
            code: "no-output",
          });
          return;
        }
        try {
          resolve(JSON.parse(trimmed) as BridgeResponse);
        } catch (err) {
          resolve({
            ok: false,
            error: `Could not parse bridge response: ${String(err)}\n${trimmed}`,
            code: "bad-json",
          });
        }
      });

      child.stdin.write(requestJson);
      child.stdin.end();
    });
  }
}
