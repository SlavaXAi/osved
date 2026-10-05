"use strict";
const byId = id => document.getElementById(id);
byId("extension-id").value = chrome.runtime.id;

async function refresh() {
  const settings = await chrome.storage.session.get(["enabled", "lastStatus"]);
  byId("enabled").checked = Boolean(settings.enabled);
  byId("status").textContent = settings.lastStatus || "Радар выключен.";
}
byId("save").addEventListener("click", async () => {
  const token = byId("token").value.trim();
  if (!/^[A-Za-z0-9_-]{32,128}$/.test(token)) {
    byId("status").textContent = "Введите код подключения из локального OSVED (минимум 32 символа).";
    return;
  }
  await chrome.storage.session.set({pairingToken: token, enabled: byId("enabled").checked});
  byId("token").value = "";
  const result = await chrome.runtime.sendMessage({type: "OSVED_TEST"});
  byId("status").textContent = result.ok ? "Подключение готово. Откройте X → Для вас." : `Не удалось подключиться: ${result.error}`;
});
byId("enabled").addEventListener("change", async () => {
  await chrome.storage.session.set({enabled: byId("enabled").checked});
  byId("status").textContent = byId("enabled").checked ? "Радар разрешён для видимой вкладки." : "Радар выключен.";
});
byId("poll").addEventListener("click", async () => {
  byId("poll").disabled = true;
  byId("status").textContent = "Проверяю…";
  try { await chrome.runtime.sendMessage({type: "OSVED_POLL"}); await refresh(); }
  finally { byId("poll").disabled = false; }
});
void refresh();
