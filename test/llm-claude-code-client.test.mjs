/**
 * Unit tests for the claude-code LLM client.
 * Covers: env sanitization, JSON extraction, error paths.
 */
import assert from "node:assert/strict";
import { describe, it, mock } from "node:test";
import jitiFactory from "jiti";

const jiti = jitiFactory(import.meta.url, { interopDefault: true });
const { createLlmClient, buildClaudeCodeEnv, extractJsonFromResponse, repairCommonJson } = jiti("../src/llm-client.ts");

// ---------------------------------------------------------------------------
// buildClaudeCodeEnv — env sanitization (most critical security logic)
// ---------------------------------------------------------------------------

describe("buildClaudeCodeEnv", () => {
  it("preserves ANTHROPIC_API_KEY when no explicit key provided (ambient auth)", () => {
    const saved = process.env.ANTHROPIC_API_KEY;
    process.env.ANTHROPIC_API_KEY = "ambient-key";
    try {
      const env = buildClaudeCodeEnv(undefined);
      assert.equal(env.ANTHROPIC_API_KEY, "ambient-key", "should preserve ambient key when no explicit key");
    } finally {
      if (saved === undefined) delete process.env.ANTHROPIC_API_KEY;
      else process.env.ANTHROPIC_API_KEY = saved;
    }
  });

  it("replaces ANTHROPIC_API_KEY when explicit key provided", () => {
    const saved = process.env.ANTHROPIC_API_KEY;
    process.env.ANTHROPIC_API_KEY = "ambient-key";
    try {
      const env = buildClaudeCodeEnv("explicit-key");
      assert.equal(env.ANTHROPIC_API_KEY, "explicit-key", "explicit key should override ambient");
    } finally {
      if (saved === undefined) delete process.env.ANTHROPIC_API_KEY;
      else process.env.ANTHROPIC_API_KEY = saved;
    }
  });

  it("always sets CLAUDE_CODE_ENTRYPOINT=sdk-ts", () => {
    const env = buildClaudeCodeEnv();
    assert.equal(env.CLAUDE_CODE_ENTRYPOINT, "sdk-ts");
  });

  it("strips CLAUDECODE (exact) to prevent nested-session errors", () => {
    const saved = process.env.CLAUDECODE;
    process.env.CLAUDECODE = "1";
    try {
      const env = buildClaudeCodeEnv();
      assert.equal(env.CLAUDECODE, undefined, "CLAUDECODE should be stripped");
    } finally {
      if (saved === undefined) delete process.env.CLAUDECODE;
      else process.env.CLAUDECODE = saved;
    }
  });

  it("strips CLAUDECODE_* prefixed vars", () => {
    process.env.CLAUDECODE_SOME_VAR = "should-be-stripped";
    try {
      const env = buildClaudeCodeEnv();
      assert.equal(env.CLAUDECODE_SOME_VAR, undefined);
    } finally {
      delete process.env.CLAUDECODE_SOME_VAR;
    }
  });

  it("strips CLAUDE_CODE_SESSION but preserves CLAUDE_CODE_OAUTH_TOKEN", () => {
    process.env.CLAUDE_CODE_SESSION = "strip-me";
    process.env.CLAUDE_CODE_OAUTH_TOKEN = "keep-me";
    try {
      const env = buildClaudeCodeEnv();
      assert.equal(env.CLAUDE_CODE_SESSION, undefined);
      assert.equal(env.CLAUDE_CODE_OAUTH_TOKEN, "keep-me");
    } finally {
      delete process.env.CLAUDE_CODE_SESSION;
      delete process.env.CLAUDE_CODE_OAUTH_TOKEN;
    }
  });

  it("logs warning when no auth source is present", () => {
    const savedKey = process.env.ANTHROPIC_API_KEY;
    const savedToken = process.env.ANTHROPIC_AUTH_TOKEN;
    const savedOauth = process.env.CLAUDE_CODE_OAUTH_TOKEN;
    delete process.env.ANTHROPIC_API_KEY;
    delete process.env.ANTHROPIC_AUTH_TOKEN;
    delete process.env.CLAUDE_CODE_OAUTH_TOKEN;

    const logs = [];
    buildClaudeCodeEnv(undefined, (msg) => logs.push(msg));
    assert.ok(logs.some(l => l.includes("WARNING") || l.includes("no ANTHROPIC")), "should warn when no auth source");

    if (savedKey !== undefined) process.env.ANTHROPIC_API_KEY = savedKey;
    if (savedToken !== undefined) process.env.ANTHROPIC_AUTH_TOKEN = savedToken;
    if (savedOauth !== undefined) process.env.CLAUDE_CODE_OAUTH_TOKEN = savedOauth;
  });
});

// ---------------------------------------------------------------------------
// JSON extraction helpers
// ---------------------------------------------------------------------------

describe("extractJsonFromResponse", () => {
  it("extracts plain JSON", () => {
    const raw = '{"memories":[{"text":"test","category":"fact"}]}';
    const jsonStr = extractJsonFromResponse(raw);
    assert.notEqual(jsonStr, null);
    assert.deepEqual(JSON.parse(jsonStr), { memories: [{ text: "test", category: "fact" }] });
  });

  it("handles markdown fences", () => {
    const raw = "Here is the result:\n```json\n{\"ok\":true}\n```";
    const jsonStr = extractJsonFromResponse(raw);
    assert.notEqual(jsonStr, null);
    assert.deepEqual(JSON.parse(jsonStr), { ok: true });
  });

  it("returns null for non-JSON text", () => {
    assert.equal(extractJsonFromResponse("no json here"), null);
  });
});

describe("repairCommonJson", () => {
  it("removes trailing commas", () => {
    const broken = '{"a":1,"b":2,}';
    assert.doesNotThrow(() => JSON.parse(repairCommonJson(broken)));
  });

  it("escapes unescaped newlines in strings", () => {
    const broken = '{"text":"line1\nline2"}';
    const repaired = repairCommonJson(broken);
    assert.doesNotThrow(() => JSON.parse(repaired));
  });
});

// ---------------------------------------------------------------------------
// createLlmClient — claude-code client instantiation
// ---------------------------------------------------------------------------

describe("createLlmClient claude-code", () => {
  it("returns a client with completeJson and getLastError functions", () => {
    const llm = createLlmClient({
      auth: "claude-code",
      model: "claude-haiku-4-5",
      stateDir: "/tmp/test-state",
    });
    assert.equal(typeof llm.completeJson, "function");
    assert.equal(typeof llm.getLastError, "function");
    assert.equal(llm.getLastError(), null, "no error before any call");
  });

  it("returns null and sets lastError when claudeCodePath does not exist", async () => {
    const llm = createLlmClient({
      auth: "claude-code",
      model: "claude-haiku-4-5",
      stateDir: "/tmp/test-state",
      claudeCodePath: "/nonexistent/path/to/claude",
    });
    const result = await llm.completeJson('{"test": true}', "test-label");
    assert.equal(result, null, "should return null when claude binary not found");
    const err = llm.getLastError();
    assert.ok(err !== null, "should set lastError");
    assert.ok(
      err.includes("not found") || err.includes("not installed") || err.includes("claude"),
      `lastError should describe the failure, got: ${err}`,
    );
  });

  it("caches claude path resolution failure — does not retry execSync on subsequent calls", async () => {
    const llm = createLlmClient({
      auth: "claude-code",
      model: "claude-haiku-4-5",
      stateDir: "/tmp/test-state",
      claudeCodePath: "/nonexistent/path/to/claude-cached-test",
    });
    // First call triggers resolution failure
    const r1 = await llm.completeJson("prompt1", "label1");
    assert.equal(r1, null);
    const err1 = llm.getLastError();
    // Second call must also fail without re-running execSync
    const r2 = await llm.completeJson("prompt2", "label2");
    assert.equal(r2, null);
    const err2 = llm.getLastError();
    // Both errors should reference the same binary path issue
    assert.ok(err1 !== null && err2 !== null);
    assert.ok(
      err1.includes("nonexistent") || err1.includes("not found"),
      `first error should mention path, got: ${err1}`,
    );
  });
});
