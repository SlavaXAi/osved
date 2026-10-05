/* Isolated-world DOM observation only. No cookies, fetch, scroll or page actions. */
(() => {
  "use strict";
  const FOR_YOU_LABELS = new Set(["for you", "для вас", "推荐", "為你推薦", "为你推荐"]);

  function feedState() {
    if (location.origin !== "https://x.com" || location.pathname !== "/home") {
      return {ok: false, error: "wrong_feed"};
    }
    if (document.visibilityState !== "visible") {
      return {ok: false, error: "no_visible_tab"};
    }
    if (document.querySelector('iframe[src*="arkoselabs"], iframe[src*="captcha"]')) {
      return {ok: false, error: "challenge"};
    }
    const selected = document.querySelector('[role="tab"][aria-selected="true"]');
    const label = selected?.textContent?.trim().toLowerCase().replace(/\s+/g, " ");
    if (!label || !FOR_YOU_LABELS.has(label)) return {ok: false, error: "wrong_feed"};
    return {ok: true};
  }

  function snapshot(maxItems) {
    const state = feedState();
    if (!state.ok) return state;
    const limit = Math.max(1, Math.min(80, Number(maxItems) || 40));
    const posts = [];
    const seen = new Set();
    for (const article of document.querySelectorAll('article[data-testid="tweet"]')) {
      const rect = article.getBoundingClientRect();
      // An open/background tab or virtualized offscreen items are not user-visible observations.
      if (rect.bottom <= 0 || rect.top >= window.innerHeight || rect.width <= 0) continue;
      const timestamp = article.querySelector("time[datetime]");
      const permalink = timestamp?.closest("a")?.getAttribute("href");
      const match = permalink?.match(/^\/([A-Za-z0-9_]{1,15})\/status\/(\d{5,30})(?:[/?#]|$)/);
      if (!match || seen.has(match[2])) continue;
      const text = article.querySelector('[data-testid="tweetText"]')?.textContent?.trim() || "";
      if (!text) continue; // Media-only content is outside this experiment's claim.
      const links = [];
      for (const anchor of article.querySelectorAll('a[href^="https://"]')) {
        try {
          const url = new URL(anchor.getAttribute("href"));
          if (["x.com", "twitter.com"].includes(url.hostname)) continue;
          for (const key of [...url.searchParams.keys()]) {
            if (/cookie|token|authorization|password|secret|api.?key|ct0|session|^utm_/i.test(key)
                || ["fbclid", "gclid", "ref"].includes(key.toLowerCase())) {
              url.searchParams.delete(key);
            }
          }
          url.hash = "";
          if (!url.username && !url.password && !links.includes(url.href)) links.push(url.href);
        } catch (_) { /* An unusable link does not invalidate the post. */ }
        if (links.length >= 30) break;
      }
      seen.add(match[2]);
      posts.push({
        id: match[2], author: match[1], text: text.slice(0, 8000),
        published_at: timestamp.getAttribute("datetime"),
        url: `https://x.com/${match[1]}/status/${match[2]}`, links,
      });
      if (posts.length >= limit) break;
    }
    if (!posts.length) return {ok: false, error: "empty_feed"};
    return {ok: true, result: {feed_mode: "for_you", posts}};
  }

  chrome.runtime.onMessage.addListener((message, sender, respond) => {
    if (sender.id !== chrome.runtime.id || !message || typeof message !== "object") return false;
    if (message.type === "OSVED_PROBE") respond(feedState());
    else if (message.type === "OSVED_VISIBLE_SNAPSHOT") respond(snapshot(message.maxItems));
    return false;
  });
})();
