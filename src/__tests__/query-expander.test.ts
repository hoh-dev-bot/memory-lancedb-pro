import { describe, it, expect } from "bun:test";
import { expandQuery } from "../query-expander.js";

describe("expandQuery", () => {
  it("expands colloquial Chinese to technical terms", () => {
    const result = expandQuery("服务突然挂了");
    expect(result).toContain("崩溃");
    expect(result).toContain("crash");
    expect(result).toContain("挂了");
  });

  it("expands debug-related queries", () => {
    const result = expandQuery("程序报错了");
    expect(result).toContain("error");
    expect(result).toContain("exception");
    expect(result).toContain("bug");
  });

  it("preserves original query terms", () => {
    const result = expandQuery("配置文件");
    expect(result).toContain("配置");
    expect(result).toContain("config");
  });

  it("returns original for already-precise queries", () => {
    const result = expandQuery("JINA_API_KEY");
    expect(result).toBe("JINA_API_KEY");
  });

  it("handles empty/short queries", () => {
    expect(expandQuery("")).toBe("");
    expect(expandQuery("hi")).toBe("hi");
  });
});
