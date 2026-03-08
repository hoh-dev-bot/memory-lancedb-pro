/**
 * Lightweight Chinese query expansion via static synonym dictionary.
 * Expands colloquial/fuzzy terms into technical equivalents for BM25 boost.
 * No API calls — pure local dictionary lookup.
 *
 * Why: Chinese users often search with colloquial terms ("挂了", "炸了")
 * while stored content uses technical terms ("crash", "崩溃"). Vector search
 * handles semantic similarity, but BM25 needs exact term matches — this
 * bridges that gap.
 */

// Each entry: [trigger patterns, expansion terms]
// Trigger: if any pattern matches (substring), add all expansion terms to query
const SYNONYM_MAP: Array<[string[], string[]]> = [
  // --- Status / Failure ---
  [["挂了", "挂掉", "宕机", "down"], ["崩溃", "crash", "error", "报错", "挂了", "宕机", "失败"]],
  [["卡住", "卡死", "没反应"], ["hang", "timeout", "超时", "卡住", "无响应", "stuck"]],
  [["炸了", "爆了"], ["崩溃", "crash", "OOM", "内存溢出", "error"]],

  // --- Config / Deploy ---
  [["配置", "设置", "config"], ["配置", "config", "configuration", "settings", "设置"]],
  [["部署", "上线"], ["deploy", "部署", "上线", "发布", "release"]],
  [["容器", "docker"], ["Docker", "容器", "container", "docker-compose"]],

  // --- Code / Debug ---
  [["报错", "出错", "错误"], ["error", "报错", "exception", "错误", "失败", "bug"]],
  [["修复", "修了", "修好"], ["fix", "修复", "patch", "修了", "解决"]],
  [["踩坑", "坑"], ["踩坑", "bug", "问题", "教训", "排查", "troubleshoot"]],

  // --- Search / Memory ---
  [["记忆", "memory"], ["记忆", "memory", "记忆系统", "LanceDB", "索引"]],
  [["搜索", "查找", "找"], ["搜索", "search", "retrieval", "检索", "查找"]],

  // --- Infrastructure ---
  [["推送", "push"], ["push", "推送", "git push", "commit"]],
  [["日志", "log"], ["日志", "log", "logging", "输出", "打印"]],
  [["权限", "permission"], ["权限", "permission", "access", "授权", "认证"]],
];

/**
 * Expand a query by appending synonym terms from the dictionary.
 * Returns the original query with additional terms appended.
 * Idempotent — already-precise queries pass through unchanged.
 */
export function expandQuery(query: string): string {
  if (!query || query.trim().length < 2) return query;

  const lower = query.toLowerCase();
  const additions = new Set<string>();

  for (const [triggers, expansions] of SYNONYM_MAP) {
    if (triggers.some(t => lower.includes(t.toLowerCase()))) {
      for (const exp of expansions) {
        // Don't add terms already in the query
        if (!lower.includes(exp.toLowerCase())) {
          additions.add(exp);
        }
      }
    }
  }

  if (additions.size === 0) return query;
  return `${query} ${[...additions].join(" ")}`;
}
