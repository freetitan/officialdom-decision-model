/**
 * Cloudflare Worker 反代：把自定义域名指向 Hugging Face Space，
 * 让国内不用 VPN 也能打开聊天页。
 *
 * Worker 用自定义域名提供访问入口，请求转发到 Space 后端。
 *
 * 部署：Cloudflare Dashboard → Workers & Pages → Create → 贴入本文件
 * → Deploy → 在 Worker 的 Settings / Domains 里绑定你的自定义域名。
 * 只改下面这一行：把 BACKEND_HOST 换成你的 Space 地址（不带 https://）。
 */

// ===== 只改这一行 =====
const BACKEND_HOST = "unimaster-officialdom-v15.hf.space";

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // 目标地址：路径与 query 原样转发到 Space
    const target = new URL(url.pathname + url.search, `https://${BACKEND_HOST}`);

    // HEAD 转 GET：HF Space 根路径对 HEAD 返回 405，不转的话监控会报站挂了
    const method = request.method === "HEAD" ? "GET" : request.method;

    const headers = new Headers(request.headers);
    headers.set("Host", BACKEND_HOST);
    headers.set("X-Forwarded-Host", url.hostname);
    headers.set("X-Forwarded-Proto", "https");

    const newRequest = new Request(target, {
      method,
      headers,
      body: method === "GET" || method === "HEAD" ? null : request.body,
      redirect: "manual", // 自己处理重定向，避免 Location 泄露回 hf.space
    });

    let response;
    try {
      response = await fetch(newRequest);
    } catch (err) {
      return new Response(`上游连接失败: ${err.message}`, { status: 502 });
    }

    // WebSocket（Gradio 流式输出走 websocket）：直接把 101 回去
    if (response.status === 101) {
      return response;
    }

    const newHeaders = new Headers(response.headers);

    // 重定向回写到自定义域名，别让浏览器跳去 hf.space
    const location = newHeaders.get("Location");
    if (location) {
      try {
        const loc = new URL(location, `https://${BACKEND_HOST}`);
        if (loc.hostname === BACKEND_HOST) {
          loc.hostname = url.hostname;
          newHeaders.set("Location", loc.toString());
        }
      } catch (_) {
        // 相对路径重定向不用动
      }
    }

    // Cookie 的 Domain 若指向 hf.space，浏览器在自定义域名下会丢掉它；去掉
    const setCookie = newHeaders.get("Set-Cookie");
    if (setCookie && setCookie.includes("hf.space")) {
      newHeaders.set(
        "Set-Cookie",
        setCookie.replace(/;\s*Domain=[^;]+/gi, "")
      );
    }

    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: newHeaders,
    });
  },
};
