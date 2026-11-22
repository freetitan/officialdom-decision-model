// Regression test for the demo page's client script.
// Catches exactly the class of bug that killed v4 in the browser:
// the served <script> must parse AND run (template-literal backslash eating
// breaks it silently while `node --check worker.js` stays green).
// Run: node test_client.js [path/to/worker.js]
const fs = require("fs");
const vm = require("vm");

const WORKER_PATH = process.argv[2] ||
  "/home/hatch/workspace/goals/officialdom-specialist-model-fine-tuning/files/demo-worker/worker.js";

function makeEl(id) {
  const el = {
    id: id || "",
    children: [],
    listeners: {},
    style: {},
    className: "",
    value: "",
    textContent: "",
    innerHTML: "",
    disabled: false,
    hidden: false,
    scrollTop: 0,
    scrollHeight: 100,
    appendChild(child) { this.children.push(child); return child; },
    removeChild(child) {
      const i = this.children.indexOf(child);
      if (i >= 0) this.children.splice(i, 1);
      return child;
    },
    addEventListener(type, fn) {
      (this.listeners[type] = this.listeners[type] || []).push(fn);
    },
    focus() {},
    click() {
      (this.listeners.click || []).forEach((fn) => fn.call(this, { preventDefault() {} }));
    },
  };
  return el;
}

async function main() {
  const tmp = "/tmp/worker_under_test.mjs";
  fs.copyFileSync(WORKER_PATH, tmp);
  const mod = await import(tmp + "?t=" + Date.now());
  const resp = await mod.default.fetch(new Request("https://public-decision.us.ci/"), {}, {});
  const html = await resp.text();
  const m = html.match(/<script>([\s\S]*?)<\/script>/);
  if (!m) throw new Error("no <script> block in served HTML");
  const code = m[1];
  if (!code.includes("sendBtn.addEventListener")) throw new Error("send listener missing from served script");

  const els = {};
  ["log", "logwrap", "input", "send", "stop", "newchat", "welcome"].forEach((id) => {
    els[id] = makeEl(id);
  });
  const exButtons = [
    "某县遭遇水灾，作为县长，赈灾工作应如何组织？",
    "一名官员在哪些情形下需要回避？具体程序是什么？",
    "2026 年赤字率比 2025 年变化了多少？",
  ].map((q) => {
    const b = makeEl("");
    b.className = "ex";
    b.textContent = q;
    return b;
  });

  const sse = 'data: {"response": "4%左右，与2025年相同，没有变化。"}\n\ndata: [DONE]\n\n';
  const sandbox = {
    console,
    TextDecoder,
    AbortController,
    setTimeout,
    clearTimeout,
    Response,
    navigator: {},
    atob: (s) => Buffer.from(s, "base64").toString("binary"),
    fetch: async () =>
      new Response(sse, {
        headers: {
          "content-type": "text/event-stream; charset=utf-8",
          "x-rag-sources": "",
        },
      }),
    document: {
      getElementById: (id) => els[id] || makeEl(id),
      createElement: () => makeEl(""),
      querySelectorAll: (sel) => (sel === ".ex" ? exButtons : []),
    },
  };
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;
  vm.createContext(sandbox);
  const probe = "\n;globalThis.__t = { renderMd: renderMd, getMsgs: function(){ return msgs; } };";
  vm.runInNewContext(code + probe, sandbox, { filename: "client.js" });

  const assert = (cond, msg) => {
    if (!cond) throw new Error("ASSERT FAILED: " + msg);
    console.log("ok - " + msg);
  };

  assert((els.send.listeners.click || []).length === 1, "send button has click listener");
  assert(exButtons.every((b) => (b.listeners.click || []).length === 1), "example cards have click listeners");
  assert((els.newchat.listeners.click || []).length === 1, "new-chat button has click listener");

  const md = sandbox.__t.renderMd("**加粗**与`代码`\n- 一\n- 二\n普通行");
  assert(md.includes("<strong>加粗</strong>"), "renderMd bold");
  assert(md.includes("<code>代码</code>"), "renderMd inline code");
  assert(md.includes("<ul>") && md.includes("<li>一</li>"), "renderMd list");

  // Simulate clicking the third example card (deficit question).
  exButtons[2].click();
  const deadline = Date.now() + 5000;
  let done = false;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 50));
    const msgs = sandbox.__t.getMsgs();
    if (msgs.length === 2 && msgs[1].content.includes("4%左右")) { done = true; break; }
  }
  assert(done, "example click produces a streamed assistant reply");
  assert(els.welcome.style.display === "none", "welcome hidden after first question");
  const assistantRow = els.log.children[els.log.children.length - 1];
  const assistantContent = assistantRow.children[1].children[0];
  assert(
    assistantContent.innerHTML.includes("4%左右"),
    "assistant bubble shows streamed text, got: " + assistantContent.innerHTML.slice(0, 80)
  );

  // New chat resets.
  els.newchat.click();
  assert(sandbox.__t.getMsgs().length === 0, "new chat clears history");
  console.log("ALL CLIENT TESTS PASSED");
}

main().then(
  () => process.exit(0),
  (e) => {
    console.error(e && e.stack ? e.stack : e);
    process.exit(1);
  }
);
