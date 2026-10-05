"use strict";
const BASE_URL = "http://127.0.0.1:8769";
const ALARM = "pars-visible-snapshot";
let busy = false;

async function setStatus(message) {
  await chrome.storage.session.set({lastStatus: message, lastCheck: new Date().toISOString()});
}

async function request(path, body, token) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch(`${BASE_URL}${path}`, {
      method: body === undefined ? "GET" : "POST", mode: "cors", credentials: "omit",
      cache: "no-store", redirect: "error", signal: controller.signal,
      headers: {Authorization: `Bearer ${token}`, "Content-Type": "application/json"},
      ...(body === undefined ? {} : {body: JSON.stringify(body)}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    return data;
  } finally { clearTimeout(timer); }
}

async function poll() {
  if (busy) return;
  busy = true;
  try {
    const settings = await chrome.storage.session.get(["pairingToken", "enabled"]);
    if (!settings.enabled || !settings.pairingToken) return;
    const tabs = await chrome.tabs.query({active: true, lastFocusedWindow: true, url: "https://x.com/*"});
    const tab = tabs.find(item => new URL(item.url).pathname === "/home");
    if (!tab?.id) { await setStatus("Откройте X → Для вас в активной вкладке."); return; }
    const probe = await chrome.tabs.sendMessage(tab.id, {type: "OSVED_PROBE"});
    if (!probe?.ok) {
      await setStatus(`Снимок недоступен: ${probe?.error || "page_unavailable"}`);
      if (probe?.error === "challenge") await chrome.storage.session.set({enabled: false});
      return;
    }
    const {task} = await request("/v1/tasks/claim", {}, settings.pairingToken);
    if (!task) { await setStatus("Связь есть. В очереди пока нет новых задач."); return; }
    if (task.platform !== "x" || task.kind !== "visible_feed_snapshot" || task.payload?.mode !== "for_you") {
      await setStatus("Неизвестная задача остановлена.");
      await chrome.storage.session.set({enabled: false});
      return;
    }
    const captured = await chrome.tabs.sendMessage(tab.id, {
      type: "OSVED_VISIBLE_SNAPSHOT", maxItems: task.payload.max_items,
    });
    if (!captured?.ok) {
      const code = captured?.error || "extension_error";
      await request("/v1/tasks/fail", {
        task_id: task.id, claim_token: task.claim_token, error_code: code,
      }, settings.pairingToken);
      if (code === "challenge" || code === "login_required") {
        await chrome.storage.session.set({enabled: false});
      }
      await setStatus(`Снимок остановлен: ${code}`);
      return;
    }
    // Persist only sanitized observations before sending. If the service worker
    // stops during transport, the same task/result is retried on the next alarm.
    const staged = {task_id: task.id, claim_token: task.claim_token, result: captured.result};
    await chrome.storage.session.set({pendingResult: staged});
    await request("/v1/tasks/result", staged, settings.pairingToken);
    await chrome.storage.session.remove("pendingResult");
    await setStatus(`Передано ${captured.result.posts.length} видимых публикаций. Cookies остались в браузере.`);
  } catch (error) {
    await setStatus(`Связь или вкладка недоступна: ${error.message}`);
  } finally { busy = false; }
}

async function run() {
  const settings = await chrome.storage.session.get(["pairingToken", "enabled", "pendingResult"]);
  if (settings.enabled && settings.pairingToken && settings.pendingResult) {
    try {
      await request("/v1/tasks/result", settings.pendingResult, settings.pairingToken);
      await chrome.storage.session.remove("pendingResult");
      await setStatus("Сохранённый снимок успешно доставлен.");
    } catch (error) {
      if (["stale_claim", "invalid_claim", "result_conflict"].includes(error.message)) {
        await chrome.storage.session.remove("pendingResult");
      }
      await setStatus(`Ожидание повторной доставки: ${error.message}`);
      return;
    }
  }
  await poll();
}

chrome.runtime.onInstalled.addListener(() => {
  chrome.alarms.create(ALARM, {periodInMinutes: 1});
});
chrome.runtime.onStartup.addListener(() => {
  chrome.alarms.create(ALARM, {periodInMinutes: 1});
});
chrome.alarms.onAlarm.addListener(alarm => { if (alarm.name === ALARM) void run(); });
chrome.runtime.onMessage.addListener((message, sender, respond) => {
  if (sender.id !== chrome.runtime.id) return false;
  if (message?.type === "OSVED_POLL") {
    run().then(() => respond({ok: true})).catch(() => respond({ok: false}));
    return true;
  }
  if (message?.type === "OSVED_TEST") {
    chrome.storage.session.get("pairingToken").then(settings =>
      request("/v1/health", undefined, settings.pairingToken || "")
    ).then(() => respond({ok: true})).catch(error => respond({ok: false, error: error.message}));
    return true;
  }
  return false;
});
