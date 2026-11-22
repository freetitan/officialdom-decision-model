// 演示站 Worker v4：ChatGPT 式界面 + Workers AI（Llama 3.3 70B）流式输出 + 训练资料 RAG。
// 本文件是模板：__RAG_CORPUS_JSON__ 由 build.py 替换为 corpus.json 内容，生成 worker.js 再部署。
// 部署：python3 build.py（生成 + 校验大小）→ /tmp/deploy_worker.py（multipart，metadata 在前）。
// 绑定域名：public-decision.us.ci（Workers Custom Domain）。这不是微调模型 v15；v15 在 decision.ccwu.cc。
// 教训：内联 classic script 顶层禁用 var history/name/status/location 等与 window 内置属性撞名的变量。

const SYSTEM_PROMPT =
  "你是一位熟悉中国官场规则、制度与决策程序的助手，帮助使用者分析具体情境、" +
  "识别约束与风险、给出可执行的处理建议。回答用中文，条理清晰，引用制度依据时注明出处。";
const MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";
const MAX_HISTORY = 20;
const MAX_TOKENS = 1024;
const DIAG = [];

// 训练资料 RAG 语料（build.py 注入）：{id, t 标题, s 来源, x 正文, h 历史标记}
const RAG_CORPUS = __RAG_CORPUS_JSON__;

// ---- 关键词检索（无向量库、无额外费用）：中文双字重叠 + 停用双字表 ----
const STOP = new Set(
  "这是 这个 那个 一个 是在 有了 可以 应该 怎么 如何 什么 为什么 多少 哪些 是否 能否 请问 如果 但是 因为 所以 然后 这样 那样 这里 那里 一些 一点 一下 没有 不是 还是 就是 而且 或者 并且 以及 之所以 来说 方面 情况 时候 地方 事情 工作 人员 干部 领导 问题 进行 要求 规定 根据 按照 对于 关于 有关 需要 必须 应当 能够 可能 已经 正在".split(
    " "
  )
);
const HIST_Q = /清代|清朝|古代|明代|宋代|元代|大清律|封建|旧时|古时|历史|传统/;
const MIN_SCORE = 5;
const TOP_K = 3;

function bigramsOf(s) {
  const t = s.replace(/\s+/g, "");
  const out = new Set();
  for (let i = 0; i < t.length - 1; i++) {
    const g = t.slice(i, i + 2);
    if (!STOP.has(g)) out.add(g);
  }
  return out;
}
const RAG_INDEX = RAG_CORPUS.map((c) => ({ c, bg: bigramsOf(c.t + " " + c.x) }));

function retrieve(q) {
  const qb = bigramsOf(q);
  const allowHist = HIST_Q.test(q);
  const scored = [];
  for (const e of RAG_INDEX) {
    if (e.c.h && !allowHist) continue;
    let shared = 0;
    qb.forEach((g) => {
      if (e.bg.has(g)) shared++;
    });
    if (shared >= MIN_SCORE) scored.push({ c: e.c, score: shared });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, TOP_K);
}

function buildContext(hits) {
  const lines = [
    "【参考资料】以下内容来自本站绑定的训练资料（DPO 核验回答与法规片段），回答时优先依据这些内容；如与你的一般知识冲突，以参考资料为准并注明出处。",
  ];
  hits.forEach((h, i) => {
    lines.push(
      "[" +
        (i + 1) +
        "] " +
        (h.c.h ? "【史料背景，仅作参考，不得作为现行决策依据】" : "") +
        h.c.t +
        "（" +
        h.c.s +
        "）：" +
        h.c.x
    );
  });
  return lines.join("\n");
}

function b64encodeUnicode(str) {
  const bytes = new TextEncoder().encode(str);
  let bin = "";
  bytes.forEach((b) => {
    bin += String.fromCharCode(b);
  });
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

const HTML = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>公共管理决策大模型（演示）</title>
<style>
*{box-sizing:border-box}
[hidden]{display:none !important}
html,body{height:100%}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;background:#fff;color:#1f2328;display:flex;flex-direction:column;height:100dvh}
.top{border-bottom:1px solid #e5e7eb;background:#fff}
.top-inner{max-width:820px;margin:0 auto;padding:12px 16px 4px;display:flex;align-items:center;gap:10px}
.brand{display:flex;align-items:center;gap:9px;min-width:0}
.avatar{width:30px;height:30px;border-radius:50%;background:#111827;color:#fff;display:flex;align-items:center;justify-content:center;font-size:15px;flex:none}
.title{font-size:16px;font-weight:650;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ghost-btn{margin-left:auto;border:1px solid #d1d5db;background:#fff;border-radius:999px;padding:6px 14px;font-size:13px;cursor:pointer;flex:none}
.ghost-btn:hover{background:#f9fafb}
#logwrap{flex:1;overflow-y:auto}
#log{max-width:820px;margin:0 auto;padding:8px 16px 16px}
#welcome{padding:36px 8px 8px;text-align:center}
#welcome h2{font-size:26px;margin:0 0 8px;font-weight:650}
#welcome p{color:#6b7280;font-size:14px;line-height:1.7;margin:0 0 22px}
.examples{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;text-align:left}
.ex{border:1px solid #e5e7eb;background:#fff;border-radius:12px;padding:12px 14px;font-size:13.5px;line-height:1.6;cursor:pointer;text-align:left;font-family:inherit}
.ex:hover{background:#f9fafb;border-color:#d1d5db}
.row{display:flex;gap:12px;padding:14px 0}
.row.user{justify-content:flex-end}
.row.user .bubble{background:#f4f4f5;border-radius:18px;padding:10px 14px;max-width:85%;font-size:15px;line-height:1.75;white-space:pre-wrap;word-break:break-word}
.row.assistant .main{flex:1;min-width:0}
.row.assistant .avatar{margin-top:2px}
.content{font-size:15px;line-height:1.8;word-break:break-word}
.content p{margin:0 0 10px}
.content p:last-child{margin-bottom:0}
.content ul{margin:0 0 10px;padding-left:20px}
.content li{margin:4px 0}
.content code{background:#f3f4f6;border-radius:6px;padding:1px 5px;font-size:13.5px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.meta{display:flex;gap:12px;align-items:center;margin-top:6px;flex-wrap:wrap}
.sources{font-size:12px;color:#6b7280}
.copy-btn{border:0;background:none;color:#6b7280;font-size:12px;cursor:pointer;padding:2px 4px}
.copy-btn:hover{color:#111827}
.stopped{color:#6b7280;font-size:13px}
.dots span{animation:blink 1.2s infinite;display:inline-block}
.dots span:nth-child(2){animation-delay:.2s}
.dots span:nth-child(3){animation-delay:.4s}
@keyframes blink{0%,80%,100%{opacity:.25}40%{opacity:1}}
.composer{border-top:1px solid #e5e7eb;background:#fff}
.composer-inner{max-width:820px;margin:0 auto;padding:12px 16px 14px}
.box{display:flex;align-items:flex-end;gap:8px;border:1px solid #d1d5db;border-radius:24px;padding:8px 8px 8px 16px;box-shadow:0 1px 4px rgba(0,0,0,.06)}
.box:focus-within{border-color:#9ca3af;box-shadow:0 1px 6px rgba(0,0,0,.08)}
#input{flex:1;border:0;outline:none;resize:none;font-size:15px;line-height:1.6;font-family:inherit;max-height:180px;padding:6px 0;background:transparent}
#send,#stop{width:34px;height:34px;border-radius:50%;border:0;background:#111827;color:#fff;font-size:17px;cursor:pointer;flex:none;display:flex;align-items:center;justify-content:center}
#send:disabled{opacity:.35;cursor:default}
.foot{font-size:12px;color:#9ca3af;text-align:center;margin:10px 0 0;line-height:1.6}
.credit{font-size:12px;color:#6b7280;text-align:center;margin:6px 0 0;line-height:1.6}
.credit a{color:inherit}
@media (max-width:640px){#welcome h2{font-size:22px}}
</style>
</head>
<body>
<header class="top">
<div class="top-inner">
<div class="brand"><span class="avatar">官</span><span class="title">公共管理决策大模型（演示）</span></div>
<button id="newchat" class="ghost-btn">新对话</button>
</div>
</header>
<main id="logwrap"><div id="log">
<div id="welcome">
<h2>有什么可以帮你分析的？</h2>
<p>输入一个具体情境，模型会结合制度依据给出处理建议；相关问题会自动参考训练资料。</p>
<div class="examples">
<button class="ex">某县遭遇水灾，作为县长，赈灾工作应如何组织？</button>
<button class="ex">一名官员在哪些情形下需要回避？具体程序是什么？</button>
<button class="ex">2026 年赤字率比 2025 年变化了多少？</button>
</div>
</div>
</div></main>
<footer class="composer">
<div class="composer-inner">
<div class="box">
<textarea id="input" rows="1" placeholder="输入一个具体情境，例如：县里要推一项改革，局长不支持，该怎么办？"></textarea>
<button id="send" title="发送">↑</button>
<button id="stop" title="停止" hidden>■</button>
</div>
<p class="foot">回答仅供研究参考，不构成决策依据。引用制度依据时请自行核对原文。</p>
<p class="credit">西南交通大学数字治理研究中心；西南交通大学智能政策实验室<br>联系方式：<a href="mailto:crip@swjtu.edu.cn">crip@swjtu.edu.cn</a></p>
</div>
</footer>
<script>
var log = document.getElementById("log");
var logwrap = document.getElementById("logwrap");
var input = document.getElementById("input");
var sendBtn = document.getElementById("send");
var stopBtn = document.getElementById("stop");
var newchatBtn = document.getElementById("newchat");
var welcome = document.getElementById("welcome");
var msgs = [];
var busy = false;
var ctrl = null;
var stoppedByUser = false;
var lastRender = 0;
function esc(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");}
function inlineFmt(s){var BT=String.fromCharCode(96);var parts=s.split(BT);var out="";for(var i=0;i<parts.length;i++){out+=(i%2===1)?"<code>"+parts[i]+"</code>":parts[i];}out=out.replace(/\\*\\*([^*]+)\\*\\*/g,"<strong>$1</strong>");return out;}
function renderMd(src){
  var lines = esc(src).split("\\n");
  var html = ""; var inList = false; var para = [];
  function flushPara(){if(para.length){html += "<p>" + para.join("<br>") + "</p>";para=[];}}
  function closeList(){if(inList){html += "</ul>";inList=false;}}
  for(var i=0;i<lines.length;i++){
    var ln = lines[i];
    var m = ln.match(/^\\s*[-*]\\s+(.*)$/);
    var m2 = ln.match(/^\\s*\\d+[.、]\\s+(.*)$/);
    if(m || m2){
      flushPara();
      if(!inList){html += "<ul>";inList=true;}
      html += "<li>" + inlineFmt(m ? m[1] : m2[1]) + "</li>";
    }else if(/^\\s*$/.test(ln)){
      closeList();flushPara();
    }else if(/^#{1,3}\\s+/.test(ln)){
      closeList();flushPara();
      html += "<p><strong>" + inlineFmt(ln.replace(/^#{1,3}\\s+/,"")) + "</strong></p>";
    }else{
      closeList();para.push(inlineFmt(ln));
    }
  }
  closeList();flushPara();
  return html;
}
function scrollBottom(){logwrap.scrollTop = logwrap.scrollHeight;}
function streamText(obj){
  if(!obj) return "";
  if(typeof obj.response==="string") return obj.response;
  try{
    var d=obj.choices[0].delta;
    if(d&&typeof d.content==="string") return d.content;
  }catch(e){}
  return "";
}
function addRow(role){
  var row = document.createElement("div");
  row.className = "row " + role;
  if(role === "assistant"){
    var av = document.createElement("div");av.className="avatar";av.textContent="官";row.appendChild(av);
    var main = document.createElement("div");main.className="main";
    var content = document.createElement("div");content.className="content";main.appendChild(content);
    var meta = document.createElement("div");meta.className="meta";main.appendChild(meta);
    row.appendChild(main);log.appendChild(row);scrollBottom();
    return {row:row,content:content,meta:meta};
  }
  var b = document.createElement("div");b.className="bubble";row.appendChild(b);log.appendChild(row);scrollBottom();
  return {row:row,content:b,meta:null};
}
function setBusy(v){
  busy=v;sendBtn.hidden=v;stopBtn.hidden=!v;sendBtn.disabled=v;
}
function autoGrow(){input.style.height="auto";input.style.height=Math.min(input.scrollHeight,180)+"px";}
function decodeSources(h){
  try{
    var s=h.replace(/-/g,"+").replace(/_/g,"/");
    while(s.length%4) s+="=";
    var bin=atob(s);var bytes=new Uint8Array(bin.length);
    for(var i=0;i<bin.length;i++) bytes[i]=bin.charCodeAt(i);
    return JSON.parse(new TextDecoder().decode(bytes));
  }catch(e){return [];}
}
function renderSources(meta,sources){
  if(!meta||!sources||!sources.length) return;
  var t=sources.map(function(x){return x.t;}).join("；");
  var d=document.createElement("div");d.className="sources";d.textContent="参考训练资料："+t;meta.appendChild(d);
}
function addCopyBtn(meta,getText){
  var b=document.createElement("button");b.className="copy-btn";b.textContent="复制";
  b.addEventListener("click",function(){
    var txt=getText();
    if(navigator.clipboard&&navigator.clipboard.writeText){
      navigator.clipboard.writeText(txt).then(function(){b.textContent="已复制";setTimeout(function(){b.textContent="复制";},1200);});
    }
  });
  meta.appendChild(b);
}
async function ask(){
  var q=input.value.trim();
  if(!q||busy) return;
  if(welcome) welcome.style.display="none";
  input.value="";autoGrow();
  var u=addRow("user");u.content.textContent=q;
  var a=null;var acc="";
  stoppedByUser=false;
  setBusy(true);
  ctrl=new AbortController();
  var timer=setTimeout(function(){try{ctrl.abort();}catch(e){}},90000);
  try{
    msgs.push({role:"user",content:q});
    a=addRow("assistant");
    a.content.innerHTML='<span class="dots"><span>.</span><span>.</span><span>.</span></span>';
    var r=await fetch("/api/chat",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({messages:msgs,stream:true}),signal:ctrl.signal});
    var ct=r.headers.get("content-type")||"";
    var sources=decodeSources(r.headers.get("x-rag-sources")||"");
    if(!r.ok){
      var ej=null;try{ej=await r.json();}catch(e){}
      throw new Error((ej&&ej.error)||("HTTP "+r.status));
    }
    if(ct.indexOf("text/event-stream")!==-1&&r.body&&r.body.getReader){
      var reader=r.body.getReader();var dec=new TextDecoder();var buf="";var first=true;
      for(;;){
        var part=await reader.read();
        if(part.done) break;
        buf+=dec.decode(part.value,{stream:true});
        var chunks=buf.split("\\n\\n");buf=chunks.pop();
        for(var i=0;i<chunks.length;i++){
          var line=chunks[i].trim();if(!line) continue;
          if(line.indexOf("data:")===0) line=line.slice(5).trim();
          if(!line||line==="[DONE]") continue;
          var obj=null;try{obj=JSON.parse(line);}catch(e){continue;}
          var delta=streamText(obj);
          if(delta){
            acc+=delta;first=false;
            var now=Date.now();
            if(now-lastRender>80){lastRender=now;a.content.innerHTML=renderMd(acc);scrollBottom();}
          }
        }
      }
      if(buf.trim()){
        var tail=buf.trim();
        if(tail.indexOf("data:")===0) tail=tail.slice(5).trim();
        if(tail&&tail!=="[DONE]"){try{var o2=JSON.parse(tail);acc+=streamText(o2);}catch(e){}}
      }
      if(!acc) throw new Error("模型没有返回内容");
    }else{
      var j=await r.json();
      acc=j.reply||"";
      if(j.sources) sources=j.sources;
      if(!acc) throw new Error("模型没有返回内容");
    }
    clearTimeout(timer);
    a.content.innerHTML=renderMd(acc);
    renderSources(a.meta,sources);
    addCopyBtn(a.meta,function(){return acc;});
    msgs.push({role:"assistant",content:acc});
  }catch(e){
    clearTimeout(timer);
    var reason=stoppedByUser?"已停止":((e&&e.name==="AbortError")?"超时（90 秒）":((e&&e.message)?e.message:"请求失败"));
    if(acc){
      if(a){
        a.content.innerHTML=renderMd(acc)+'<p class="stopped">（输出中断：'+esc(reason)+'）</p>';
        addCopyBtn(a.meta,function(){return acc;});
      }
      msgs.push({role:"assistant",content:acc});
    }else if(a){
      a.content.innerHTML=renderMd("请求"+reason+"，没拿到回答。问题已放回输入框，再点一次发送即可。");
      msgs.pop();
      input.value=q;autoGrow();
    }else{
      msgs.pop();
      input.value=q;autoGrow();
    }
  }
  setBusy(false);ctrl=null;scrollBottom();input.focus();
}
sendBtn.addEventListener("click",ask);
stopBtn.addEventListener("click",function(){stoppedByUser=true;if(ctrl){try{ctrl.abort();}catch(e){}}});
input.addEventListener("keydown",function(e){if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();ask();}});
input.addEventListener("input",autoGrow);
newchatBtn.addEventListener("click",function(){
  if(busy) return;
  msgs=[];log.innerHTML="";
  if(welcome){welcome.style.display="";log.appendChild(welcome);}
  input.value="";autoGrow();input.focus();
});
var exs=document.querySelectorAll(".ex");
for(var i=0;i<exs.length;i++){
  exs[i].addEventListener("click",function(){input.value=this.textContent.trim();autoGrow();ask();});
}
input.focus();
</script>
</body>
</html>`;

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/api/diag") {
      return Response.json({
        aiBinding: !!env.AI,
        ragChunks: RAG_CORPUS.length,
        log: DIAG.slice(-20),
      });
    }
    if (url.pathname === "/api/chat") {
      if (request.method !== "POST") {
        return new Response("Method Not Allowed", { status: 405 });
      }
      let body;
      try {
        body = await request.json();
      } catch {
        return Response.json({ error: "请求不是合法 JSON" }, { status: 400 });
      }
      const incoming = Array.isArray(body.messages) ? body.messages : [];
      const history = incoming
        .filter(
          (m) =>
            m &&
            (m.role === "user" || m.role === "assistant") &&
            typeof m.content === "string"
        )
        .map((m) => ({ role: m.role, content: m.content.slice(0, 4000) }))
        .slice(-MAX_HISTORY);
      if (!history.length || history[history.length - 1].role !== "user") {
        return Response.json({ error: "请先输入问题" }, { status: 400 });
      }
      const hits = retrieve(history[history.length - 1].content);
      const modelMessages = [{ role: "system", content: SYSTEM_PROMPT }];
      if (hits.length) {
        modelMessages.push({ role: "user", content: buildContext(hits) });
      }
      modelMessages.push(...history);
      const sources = hits.map((h) => ({ t: h.c.t, s: h.c.s }));
      const started = Date.now();
      const ragInfo = hits.map((h) => ({ id: h.c.id, score: h.score }));
      try {
        if (body.stream === true) {
          const aiStream = await env.AI.run(MODEL, {
            messages: modelMessages,
            max_tokens: MAX_TOKENS,
            stream: true,
          });
          DIAG.push({
            t: new Date().toISOString(),
            ok: true,
            stream: true,
            rag: ragInfo,
            ms: Date.now() - started,
          });
          if (DIAG.length > 20) DIAG.shift();
          return new Response(aiStream, {
            headers: {
              "content-type": "text/event-stream; charset=utf-8",
              "cache-control": "no-store",
              "x-rag-count": String(hits.length),
              "x-rag-sources": b64encodeUnicode(JSON.stringify(sources)),
            },
          });
        }
        const out = await env.AI.run(MODEL, {
          messages: modelMessages,
          max_tokens: MAX_TOKENS,
        });
        DIAG.push({
          t: new Date().toISOString(),
          ok: true,
          stream: false,
          rag: ragInfo,
          ms: Date.now() - started,
        });
        if (DIAG.length > 20) DIAG.shift();
        return Response.json({ reply: out.response || "", sources });
      } catch (e) {
        const msg = String((e && e.message) || e);
        DIAG.push({
          t: new Date().toISOString(),
          ok: false,
          stream: body.stream === true,
          rag: ragInfo,
          ms: Date.now() - started,
          err: msg.slice(0, 200),
        });
        if (DIAG.length > 20) DIAG.shift();
        const quota = /neuron|quota|429|limit/i.test(msg);
        return Response.json(
          {
            error: quota
              ? "今日免费额度已用完，UTC 零点后恢复。"
              : "模型调用失败，请稍后重试。",
          },
          { status: 502 }
        );
      }
    }
    return new Response(HTML, {
      headers: {
        "content-type": "text/html; charset=utf-8",
        "cache-control": "no-store",
      },
    });
  },
};
