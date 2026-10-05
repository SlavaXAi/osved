const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const paths = {
  grid: "M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z",
  sparkles:
    "m12 3 2.2 6.8L21 12l-6.8 2.2L12 21l-2.2-6.8L3 12l6.8-2.2L12 3 M20 3v4 M18 5h4",
  rocket:
    "M14 5c3-3 7-2 7-2s1 4-2 7l-6 6-5-5 6-6Z M8 11H3l3-5h6 M13 16v5l5-3v-6 M5 15c-2 1-2 5-2 5s4 0 5-2 M15 8h.01",
  globe:
    "M21 12a9 9 0 1 0-18 0 9 9 0 0 0 18 0 M3 12h18 M12 3c5 5 5 13 0 18-5-5-5-13 0-18",
  orbit:
    "M8 8a5.7 5.7 0 0 0 8 8 M16 8a5.7 5.7 0 0 0-8 8 M19.3 4.7c2.3 2.3.8 7.5-3.4 11.7S6.5 22.1 4.2 19.8 3.4 12.3 7.6 8.1 17 2.4 19.3 4.7 M12 12h.01",
  people:
    "M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2 M13 7a4 4 0 1 0-8 0 4 4 0 0 0 8 0 M22 21v-2a4 4 0 0 0-3-3.9 M16 3.1a4 4 0 0 1 0 7.8",
  idea: "M9 18h6 M9 21h6 M8 14a6 6 0 1 1 8 0c-1 1-1 2-1 2H9s0-1-1-2Z",
  bookmark: "M6 3h12v18l-6-4-6 4V3Z",
  video: "M3 5h14v14H3z M17 9l5-3v12l-5-3",
  menu: "M4 6h16 M4 12h16 M4 18h16",
  close: "M6 6l12 12 M6 18 18 6",
  refresh: "M20 8a8 8 0 1 0 0 8 M20 3v5h-5",
  search: "M17 10a7 7 0 1 0-14 0 7 7 0 0 0 14 0 M15 15l6 6",
  compass: "M21 12a9 9 0 1 0-18 0 9 9 0 0 0 18 0 M16 8l-3 5-5 3 3-5 5-3Z",
  pin: "M19 10c0 5-7 11-7 11S5 15 5 10a7 7 0 0 1 14 0Z M14 10a2 2 0 1 0-4 0 2 2 0 0 0 4 0",
  down: "M7 14h3v5c0 2 3 2 3 0l3-7V3H6L3 11c-.5 2 0 3 2 3h2Z M17 3h4v10h-4",
  up: "M7 10h3V5c0-2 3-2 3 0l3 7v9H6l-3-8c-.5-2 0-3 2-3h2Z M17 11h4v10h-4",
  no: "M21 12a9 9 0 1 0-18 0 9 9 0 0 0 18 0 M8 12h8",
  flame:
    "M13 2s1 5-3 8c0-3-2-4-2-4s-4 4-4 8a8 8 0 0 0 16 0c0-5-7-12-7-12Z M10 21c-4-4 2-7 2-7s3 3 2 6",
  arrow: "M7 17 17 7 M7 7h10v10",
  chevron: "m9 5 7 7-7 7",
  file: "M14 2H5v20h14V7l-5-5Z M14 2v6h5 M8 13h8 M8 17h6",
  check: "m5 12 4 4L19 6",
  news: "M5 4h15v16H5z M2 8v12h3 M8 8h9 M8 12h9 M8 16h5",
  inbox: "M4 3h16l2 11v7H2v-7L4 3Z M2 14h6l2 3h4l2-3h6",
};
function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS(svg.namespaceURI, "path");
  path.setAttribute("d", paths[name] || paths.news);
  svg.append(path);
  return svg;
}
function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}
function button(cls, label, iconName) {
  const b = el("button", cls);
  b.type = "button";
  if (iconName) b.append(icon(iconName));
  if (label) b.append(el("span", "", label));
  return b;
}
$$("[data-icon]").forEach((node) =>
  node.replaceChildren(icon(node.dataset.icon)),
);
const folders = {
  all: {
    name: "Все материалы",
    icon: "grid",
    description: "Общая картина из всех твоих интересов.",
  },
  ai_software: {
    name: "AI и software",
    icon: "sparkles",
    description: "Модели, агенты и инструменты, которые меняют работу.",
  },
  startups: {
    name: "Стартапы",
    icon: "rocket",
    description: "Новые продукты, рынки и способы создавать ценность.",
  },
  political_economy: {
    name: "Политэкономика",
    icon: "globe",
    description: "Государства, мировая торговля и экономические связи.",
  },
  space: {
    name: "Космос",
    icon: "orbit",
    description: "Компании и технологии за пределами земной орбиты.",
  },
  people_institutions: {
    name: "Люди и институты",
    icon: "people",
    description: "Что говорят и делают люди, за которыми ты следишь.",
  },
  human_civilization: {
    name: "Философия",
    icon: "idea",
    description: "Человек, сознание и возможное будущее цивилизации.",
  },
  unclassified: {
    name: "Другие находки",
    icon: "inbox",
    description: "Материалы, пока не распределённые по папкам.",
  },
  saved: {
    name: "Сохранённое",
    icon: "bookmark",
    description: "Материалы, к которым хочется вернуться.",
  },
};
const regions = {
  moscow: "Москва",
  beijing: "Пекин",
  gba: "Greater Bay Area",
  rostov: "Ростов-на-Дону",
  tehran: "Тегеран",
};
const state = {
  folder: "all",
  view: "feed",
  region: null,
  query: "",
  csrf: "",
  bootstrap: null,
  items: [],
  itemMap: new Map(),
  cursor: null,
  hasMore: false,
  total: 0,
  loading: false,
  controller: null,
  epoch: 0,
  expansions: new Map(),
  busy: new Map(),
  errors: new Map(),
  jobs: new Map(),
};
try {
  const prior = JSON.parse(localStorage.getItem("pars.ui") || "{}");
  if (folders[prior.folder]) state.folder = prior.folder;
} catch {}
function persist() {
  try {
    localStorage.setItem("pars.ui", JSON.stringify({ folder: state.folder }));
  } catch {}
}
function safeURL(value, local = false) {
  if (typeof value !== "string" || !value) return null;
  try {
    const url = new URL(value, location.origin);
    if (
      !["http:", "https:"].includes(url.protocol) ||
      url.username ||
      url.password
    )
      return null;
    if (local && url.origin !== location.origin) return null;
    return url.href;
  } catch {
    return null;
  }
}
function link(value, text, cls) {
  const href = safeURL(value);
  if (!href) return el("span", cls, text);
  const a = el("a", cls, text);
  a.href = href;
  a.target = "_blank";
  a.rel = "noopener noreferrer";
  return a;
}
function richText(value) {
  const fragment = document.createDocumentFragment();
  if (!value) return fragment;
  if (!/<[a-z][\s\S]*>/i.test(String(value))) {
    String(value)
      .split(/\n\s*\n/)
      .forEach((text) => {
        const p = el("p");
        text.split("\n").forEach((line, i) => {
          if (i) p.append(document.createElement("br"));
          p.append(document.createTextNode(line));
        });
        fragment.append(p);
      });
    return fragment;
  }
  const parsed = new DOMParser().parseFromString(String(value), "text/html");
  const allowed = new Set([
    "P",
    "BR",
    "B",
    "STRONG",
    "I",
    "EM",
    "U",
    "S",
    "BLOCKQUOTE",
    "CODE",
    "PRE",
    "UL",
    "OL",
    "LI",
    "H3",
    "HR",
  ]);
  const forbidden = new Set([
    "SCRIPT",
    "STYLE",
    "IFRAME",
    "OBJECT",
    "EMBED",
    "FORM",
    "INPUT",
    "TEXTAREA",
    "SVG",
    "MATH",
    "IMG",
    "VIDEO",
    "AUDIO",
    "LINK",
    "META",
  ]);
  function copy(node) {
    if (node.nodeType === Node.TEXT_NODE)
      return document.createTextNode(node.textContent);
    if (node.nodeType !== Node.ELEMENT_NODE)
      return document.createDocumentFragment();
    if (forbidden.has(node.tagName)) return document.createDocumentFragment();
    let target;
    if (node.tagName === "A") {
      const href = safeURL(node.getAttribute("href"));
      target = el(href ? "a" : "span");
      if (href) {
        target.href = href;
        target.target = "_blank";
        target.rel = "noopener noreferrer";
      }
    } else
      target = allowed.has(node.tagName)
        ? el(node.tagName.toLowerCase())
        : document.createDocumentFragment();
    node.childNodes.forEach((child) => target.append(copy(child)));
    return target;
  }
  parsed.body.childNodes.forEach((node) => fragment.append(copy(node)));
  return fragment;
}
function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => {
    node.hidden = true;
  }, 4200);
}
function removeRepeatedTitle(fragment, title) {
  const normalize = (value) =>
    String(value || "")
      .replace(/\s+/g, " ")
      .trim();
  let first = fragment.firstChild;
  while (
    first &&
    first.nodeType === Node.TEXT_NODE &&
    !normalize(first.textContent)
  )
    first = first.nextSibling;
  if (first && normalize(first.textContent) === normalize(title)) {
    first.remove();
    while (
      fragment.firstChild &&
      (fragment.firstChild.nodeName === "BR" ||
        (fragment.firstChild.nodeType === Node.TEXT_NODE &&
          !normalize(fragment.firstChild.textContent)))
    )
      fragment.firstChild.remove();
  }
  // Telegram messages use literal newlines around headings; after lifting the
  // heading into the card chrome those must not become a large empty paragraph.
  for (const edge of ["firstChild", "lastChild"]) {
    while (fragment[edge]) {
      const node = fragment[edge];
      if (node.nodeName === "BR" || !normalize(node.textContent)) {
        node.remove();
        continue;
      }
      if (node.nodeType === Node.TEXT_NODE) {
        node.textContent =
          edge === "firstChild"
            ? node.textContent.trimStart()
            : node.textContent.trimEnd();
      }
      break;
    }
  }
  return fragment;
}
function displayTitle(item) {
  if (item.kind !== "video" || !item.body_html) return item.title || "Материал";
  const fragment = richText(item.body_html);
  const first = [...fragment.childNodes].find(
    (node) => node.nodeName !== "BR" && node.textContent.trim(),
  );
  if (!first || !["B", "STRONG"].includes(first.nodeName))
    return item.title || "Материал";
  const title = first.textContent.replace(/\s+/g, " ").trim();
  // Only promote the existing Russian editorial headline. Do not translate or
  // manufacture a new claim in the browser, and don't mistake a section label.
  if (
    title.length >= 15 &&
    title.length <= 240 &&
    /[а-яё]{3}/i.test(title) &&
    !/^(кратк|саммари|summary|tldr|что внутри|почему смотреть|о чём видео|содержание)/i.test(
      title,
    )
  )
    return title;
  return item.title || "Материал";
}
function stageLabel(value) {
  const stages = {
    queued: "В очереди на подготовку…",
    preparing: "Собираем материал…",
    facts: "Проверяем факты и контекст…",
    searching: "Ищем дополнительные источники…",
    research: "Изучаем источники…",
    writing: "Готовим связный текст…",
    generating: "Готовим материал…",
    transcript: "Извлекаем содержание видео…",
    summarizing: "Собираем главное…",
    pdf: "Оформляем PDF-отчёт…",
    rendering: "Оформляем PDF-отчёт…",
    running: "Материал в работе…",
  };
  return (
    stages[value] ||
    (/^[a-z_]+$/.test(value || "") ? "Материал в работе…" : value) ||
    "Материал в работе…"
  );
}
function jobError(value) {
  const message = typeof value === "string" ? value : value?.message;
  const known = {
    local_budget_blocked:
      "Не хватает свободного бюджета на новый запрос. Часть средств может быть зарезервирована. Сохранённые материалы доступны.",
    local_budget_exhausted:
      "Бюджет локальных тестов исчерпан. Сохранённые материалы остаются доступны.",
    local_paid_request_in_progress:
      "Уже выполняется платный запрос. Дождись его завершения и попробуй снова.",
    operation_already_attempted:
      "Этот запрос уже запускался. Проверь сохранённый результат; повторное списание не выполнялось.",
    invalid_reservation:
      "Не удалось оценить стоимость запроса. Генерация не запущена.",
    generation_disabled:
      "Новая генерация выключена. Сохранённые сводки и статьи доступны.",
  };
  if (known[message]) return known[message];
  return !message || /^[a-z_]+$/.test(message)
    ? "Не удалось подготовить материал. Попробуй ещё раз позже."
    : message;
}
function parseStoredDate(value) {
  if (value === null || value === undefined || value === "")
    return new Date(NaN);
  // SQLite datetime('now') is UTC even when the stored text has no offset.
  // Preserve explicit provider timezones; only normalize naive date-times.
  if (
    typeof value === "string" &&
    /^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?$/.test(value.trim())
  ) {
    return new Date(value.trim().replace(" ", "T") + "Z");
  }
  return new Date(value);
}
function dateLabel(value) {
  const date = parseStoredDate(value);
  return Number.isNaN(date.valueOf())
    ? "Дата не указана"
    : new Intl.DateTimeFormat("ru", {
        day: "numeric",
        month: "short",
        year: "numeric",
      }).format(date);
}
function dateTime(value) {
  const date = parseStoredDate(value);
  return Number.isNaN(date.valueOf())
    ? ""
    : new Intl.DateTimeFormat("ru", {
        day: "numeric",
        month: "short",
        hour: "2-digit",
        minute: "2-digit",
      }).format(date);
}
class APIError extends Error {
  constructor(message, status, code) {
    super(message);
    this.status = status;
    this.code = code;
  }
}
async function api(path, options = {}) {
  const { signal, ...rest } = options;
  const request = new AbortController();
  const abort = () => request.abort();
  let timedOut = false;
  const timeout = setTimeout(() => {
    timedOut = true;
    request.abort();
  }, 30000);
  if (signal?.aborted) request.abort();
  else signal?.addEventListener("abort", abort, { once: true });
  let response, payload;
  try {
    response = await fetch(path, {
      credentials: "same-origin",
      cache: "no-store",
      ...rest,
      signal: request.signal,
      headers: {
        Accept: "application/json",
        ...(rest.body
          ? { "Content-Type": "application/json", "X-CSRF-Token": state.csrf }
          : {}),
        ...rest.headers,
      },
    });
    payload = await response.json();
  } catch (error) {
    if (timedOut)
      throw new APIError(
        "Локальное приложение отвечает дольше обычного. Попробуй повторить запрос.",
        0,
        "timeout",
      );
    if (error.name === "AbortError") throw error;
    throw new APIError(
      "Не удалось связаться с локальным приложением. Проверь, запущен ли сервер.",
      0,
      "network",
    );
  } finally {
    clearTimeout(timeout);
    signal?.removeEventListener("abort", abort);
  }
  if (!response.ok) {
    if (response.status === 401) showAuth();
    throw new APIError(
      payload.error?.message ||
        `Не удалось выполнить запрос (${response.status}).`,
      response.status,
      payload.error?.code,
    );
  }
  return payload;
}
function showAuth() {
  state.controller?.abort();
  $("#app").hidden = true;
  $("#auth-screen").hidden = false;
  window.dispatchEvent(
    new CustomEvent("pars-globe-visibility", { detail: { visible: false } }),
  );
}
function showApp() {
  $("#auth-screen").hidden = true;
  $("#app").hidden = false;
}
function emptyState(title, description, iconName = "inbox", retry = false) {
  const node = el("div", "empty-state");
  const image = el("div", "empty-icon");
  image.append(icon(iconName));
  node.append(image, el("h2", "", title), el("p", "", description));
  if (retry) {
    const b = button("secondary-button", "Попробовать ещё раз");
    b.addEventListener("click", () =>
      state.bootstrap ? loadView() : initialize(),
    );
    node.append(b);
  }
  return node;
}
function skeletons() {
  const f = document.createDocumentFragment();
  for (let i = 0; i < 3; i++) {
    const card = el("div", "skeleton-card");
    card.setAttribute("aria-hidden", "true");
    for (let j = 0; j < 6; j++) card.append(el("div", "skeleton-line"));
    f.append(card);
  }
  return f;
}
function renderFolders() {
  const nav = $("#folder-nav");
  nav.replaceChildren();
  const counts = new Map(
    (state.bootstrap.folders || []).map((f) => [f.id, f.count]),
  );
  Object.entries(folders)
    .filter(
      ([id]) => id !== "saved" && (id !== "unclassified" || counts.get(id) > 0),
    )
    .forEach(([id, folder]) => {
      const b = button("folder-button", "", folder.icon);
      b.dataset.folder = id;
      b.append(el("span", "folder-label", folder.name));
      if (counts.has(id))
        b.append(
          el(
            "span",
            "folder-count",
            new Intl.NumberFormat("ru", { notation: "compact" }).format(
              counts.get(id),
            ),
          ),
        );
      nav.append(b);
    });
}
function renderHeader() {
  const folder = folders[state.folder] || folders.all;
  $("#folder-title").textContent = folder.name;
  $("#breadcrumb-folder").textContent = folder.name;
  $("#folder-description").textContent = folder.description;
  $$("[data-folder]").forEach((b) => {
    if (b.dataset.folder === state.folder)
      b.setAttribute("aria-current", "page");
    else b.removeAttribute("aria-current");
  });
  $$("[data-view]").forEach((b) => {
    const active = b.dataset.view === state.view;
    b.setAttribute("aria-selected", String(active));
    b.tabIndex = active ? 0 : -1;
  });
  $("#feed-panel").hidden = state.view !== "feed";
  $("#sources-panel").hidden = state.view !== "sources";
  $(".search-field").hidden = state.view !== "feed";
  $("#sources-tab").hidden = state.folder === "saved";
  const world = state.folder === "political_economy";
  $("#world-panel").hidden = !world;
  $("#reading-layout").classList.toggle("with-world", world);
  window.dispatchEvent(
    new CustomEvent("pars-globe-visibility", { detail: { visible: world } }),
  );
  $("#region-filter").hidden = !state.region;
  $("#region-filter-label").textContent = state.region
    ? `${regions[state.region]} · по упоминаниям в архиве`
    : "";
  $$("[data-region]").forEach((b) =>
    b.setAttribute("aria-pressed", String(b.dataset.region === state.region)),
  );
  $("#feed-count").textContent = state.total
    ? new Intl.NumberFormat("ru").format(state.total)
    : "";
}
function updateRegion(region) {
  clearTimeout(searchTimer);
  state.query = $("#feed-search").value;
  state.region = regions[region] ? region : null;
  state.view = "feed";
  window.dispatchEvent(
    new CustomEvent("pars-region-set", { detail: { region: state.region } }),
  );
  renderHeader();
  loadFeed();
}
function selectFolder(folder) {
  if (!folders[folder]) return;
  clearTimeout(searchTimer);
  state.folder = folder;
  state.query = "";
  state.region = null;
  state.view = "feed";
  $("#feed-search").value = "";
  window.dispatchEvent(
    new CustomEvent("pars-region-set", { detail: { region: null } }),
  );
  closeNav();
  persist();
  renderHeader();
  loadView();
  window.scrollTo({ top: 0, behavior: "instant" });
}
function itemKey(item) {
  return `${item.kind}/${item.id}`;
}
function normalize(item) {
  return {
    ...item,
    key: itemKey(item),
    sources: Array.isArray(item.sources) ? item.sources : [],
    levels: item.levels || {},
  };
}
function currentItems() {
  return state.items;
}
function bodyFor(item, level) {
  if (level === "brief") return item.body_html || item.summary || "";
  return item.content?.[level] || item[`${level}_html`] || "";
}
function renderSources(item) {
  const node = el("div", "story-sources");
  const factual = item.sources.filter(
    (s) => !["signal", "discovery", "radar", "trigger"].includes(s.role),
  );
  const signals = item.sources.filter((s) =>
    ["signal", "discovery", "radar", "trigger"].includes(s.role),
  );
  const sources = [...factual, ...signals];
  if (!sources.length && item.url)
    sources.push({
      url: item.url,
      title: item.source_name || "Оригинал",
      role: item.kind === "video" ? "primary" : "discovery",
    });
  if (!sources.length) return node;
  node.append(el("span", "sources-prefix", "ИСТОЧНИКИ"));
  sources.slice(0, 3).forEach((source) => {
    const isSignal = ["signal", "discovery", "radar", "trigger"].includes(
      source.role,
    );
    let domain = "Ссылка";
    try {
      domain = new URL(source.url).hostname.replace(/^www\./, "");
    } catch {}
    const name = source.title || source.name || domain;
    const a = link(
      source.url,
      name,
      "source-chip" + (isSignal ? " signal" : ""),
    );
    a.title = `${isSignal ? "Сигнал радара" : "Источник материала"} · ${name}`;
    if (isSignal) a.prepend(document.createTextNode("Радар · "));
    a.append(icon("arrow"));
    node.append(a);
  });
  if (sources.length > 3) {
    const more = button("sources-extra", `+${sources.length - 3}`);
    more.title = "Показать все источники";
    more.addEventListener("click", () => {
      more.remove();
      sources.slice(3).forEach((source) => {
        const radar = ["signal", "discovery", "radar", "trigger"].includes(
          source.role,
        );
        const name = source.title || source.name || "Источник";
        const chip = link(
          source.url,
          `${radar ? "Радар · " : ""}${name}`,
          `source-chip${radar ? " signal" : ""}`,
        );
        chip.title = `${radar ? "Сигнал радара" : "Источник материала"} · ${name}`;
        node.append(chip);
      });
    });
    node.append(more);
  }
  return node;
}
function statusNode(text) {
  const node = el("div", "generation-status");
  node.setAttribute("role", "status");
  const loader = el("span", "signal-loader");
  loader.setAttribute("aria-hidden", "true");
  for (let i = 0; i < 5; i++) loader.append(el("span"));
  node.append(loader, el("span", "", text));
  return node;
}
function card(item) {
  const key = itemKey(item),
    node = el("article", "story");
  const title = displayTitle(item);
  node.dataset.key = key;
  node.dataset.kind = item.kind;
  node.setAttribute("aria-label", title);
  const meta = el("div", "story-meta"),
    type = el("span", "story-type");
  type.append(
    icon(item.kind === "video" ? "video" : "news"),
    document.createTextNode(item.kind === "video" ? "Видео" : "Событие"),
  );
  meta.append(
    type,
    el("span", "meta-dot", "·"),
    el(
      "span",
      "",
      item.folder_title || folders[item.folder]?.name || "Материал",
    ),
    el("time", "story-date", dateLabel(item.published_at)),
  );
  if (item.published_at) $("time", meta).dateTime = item.published_at;
  node.append(meta, el("h2", "", title));
  const level = state.expansions.get(key) || "brief";
  if (level !== "brief") {
    const tabs = el("div", "level-tabs");
    tabs.setAttribute("aria-label", "Глубина материала");
    const choices =
      item.kind === "video"
        ? [
            ["brief", "Описание"],
            ["short", "Краткая сводка"],
            ["long", "Полный отчёт"],
          ]
        : [
            ["brief", "Событие"],
            ["context", "Контекст"],
            ["article", "Статья"],
          ];
    choices.forEach(([id, label]) => {
      const b = button("", label);
      b.dataset.action = "level";
      b.dataset.level = id;
      b.setAttribute("aria-pressed", String(level === id));
      b.disabled = state.busy.has(key);
      tabs.append(b);
    });
    node.append(tabs);
  }
  const content = el("div", "story-content");
  content.append(
    removeRepeatedTitle(
      richText(
        bodyFor(item, level) ||
          item.body_html ||
          item.summary ||
          "В архиве нет текста этого материала.",
      ),
      title,
    ),
  );
  node.append(content);
  if (level === "article" && item.article_created_at)
    node.append(
      el(
        "p",
        "article-date",
        `Статья подготовлена ${dateLabel(item.article_created_at)}`,
      ),
    );
  const pdf = safeURL(item.pdf_url, true);
  if (pdf && level === "long") {
    const a = el("a", "pdf-button");
    a.href = pdf;
    a.target = "_blank";
    a.rel = "noopener";
    a.append(icon("file"), document.createTextNode("Открыть PDF-отчёт"));
    node.append(a);
  }
  if (state.busy.has(key)) node.append(statusNode(state.busy.get(key)));
  if (state.errors.has(key))
    node.append(el("div", "story-error", state.errors.get(key)));
  node.append(renderSources(item));
  const actions = el("div", "story-actions");
  actions.setAttribute("aria-label", "Оценка и действия");
  [
    ["super_dislike", "no", "Супердизлайк"],
    ["dislike", "down", "Дизлайк"],
    ["like", "up", "Лайк"],
    ["super_like", "flame", "Суперлайк"],
  ].forEach(([value, glyph, label]) => {
    const b = button(
      "reaction-button" +
        (["dislike", "super_dislike"].includes(value) ? " negative" : ""),
      "",
      glyph,
    );
    b.dataset.action = "reaction";
    b.dataset.reaction = value;
    b.setAttribute("aria-label", label);
    b.title = label;
    b.setAttribute("aria-pressed", String(item.reaction === value));
    actions.append(b);
  });
  actions.append(el("span", "action-divider"));
  const save = button("reaction-button save-button", "", "bookmark");
  save.dataset.action = "save";
  save.title = item.saved ? "Убрать из сохранённого" : "Сохранить";
  save.setAttribute("aria-label", save.title);
  save.setAttribute("aria-pressed", String(item.saved));
  actions.append(save);
  const more = button(
    "details-button",
    state.busy.has(key)
      ? "Подготовка…"
      : item.kind === "video"
        ? level === "brief"
          ? "Краткая сводка"
          : level === "short"
            ? "Полный отчёт"
            : "Свернуть"
        : level === "brief"
          ? "Подробнее"
          : level === "context"
            ? "Читать статью"
            : "Свернуть",
    level === "article" || level === "long" ? "close" : "chevron",
  );
  more.dataset.action = "expand";
  more.setAttribute("aria-expanded", String(level !== "brief"));
  more.dataset.level =
    item.kind === "video"
      ? level === "brief"
        ? "short"
        : level === "short"
          ? "long"
          : "brief"
      : level === "brief"
        ? "context"
        : level === "context"
          ? "article"
          : "brief";
  more.disabled = state.busy.has(key);
  actions.append(more);
  node.append(actions);
  return node;
}
function renderFeed() {
  if (state.loading && !state.items.length) return;
  const container = $("#feed"),
    items = currentItems();
  container.replaceChildren();
  if (!items.length) {
    const title = state.region
      ? "Пока нет совпадений"
      : state.folder === "saved"
        ? "На потом — здесь"
        : state.query
          ? "Ничего не найдено"
          : "Пока нет материалов";
    const description = state.region
      ? "В архиве этой папки не найдено упоминаний региона. Можно убрать региональный фильтр или изменить поисковый запрос."
      : state.folder === "saved"
        ? "Отмечай материалы закладкой. Реакции и раскрытие публикации остаются отдельными действиями."
        : state.query
          ? "Попробуй другой запрос или другую папку."
          : "В этой папке нет сохранённых публикаций. Здесь показана локальная копия базы.";
    container.append(
      emptyState(
        title,
        description,
        state.folder === "saved" ? "bookmark" : "inbox",
      ),
    );
  } else items.forEach((item) => container.append(card(item)));
  $("#load-more").hidden = !state.hasMore;
  $("#load-more").disabled = state.loading;
  $("#feed-end").textContent = state.loading
    ? "Загружаем материалы…"
    : state.hasMore
      ? `Показано ${state.items.length} из ${state.total || "архива"}`
      : items.length
        ? "Ты дошёл до конца подборки. Можно вернуться к своим делам."
        : "";
  container.setAttribute("aria-busy", String(state.loading));
  renderHeader();
}
function replaceCard(key) {
  const node = $$(".story").find((node) => node.dataset.key === key);
  const item = state.itemMap.get(key);
  if (node && item) {
    const focused = node.contains(document.activeElement)
      ? document.activeElement
      : null;
    const selector = focused?.dataset.action;
    const reaction = focused?.dataset.reaction;
    const replacement = card(item);
    node.replaceWith(replacement);
    if (selector) {
      const candidate = $$("[data-action]", replacement).find(
        (b) =>
          b.dataset.action === selector &&
          (!reaction || b.dataset.reaction === reaction),
      );
      candidate?.focus({ preventScroll: true });
    }
  }
}
function updateItem(item) {
  const normalized = normalize(item),
    key = normalized.key;
  const old = state.itemMap.get(key) || {};
  const merged = {
    ...old,
    ...normalized,
    content: { ...old.content, ...normalized.content },
  };
  state.itemMap.set(key, merged);
  state.items = state.items.map((i) => (itemKey(i) === key ? merged : i));
  return merged;
}
async function loadFeed(append = false) {
  state.controller?.abort();
  const controller = new AbortController();
  state.controller = controller;
  const epoch = ++state.epoch;
  state.loading = true;
  if (!append) {
    state.items = [];
    state.cursor = null;
    state.hasMore = false;
    $("#feed").replaceChildren(skeletons());
    $("#feed-pagination").hidden = true;
  } else $("#load-more").disabled = true;
  $("#feed").setAttribute("aria-busy", "true");
  $("#refresh-feed").setAttribute("aria-busy", "true");
  const query = new URLSearchParams({
    folder: state.folder === "saved" ? "all" : state.folder,
    limit: "20",
  });
  if (state.folder === "saved") query.set("saved", "1");
  if (state.query) query.set("q", state.query);
  if (state.region) query.set("region", state.region);
  if (append && state.cursor) query.set("cursor", state.cursor);
  try {
    const result = await api(`/api/feed?${query}`, {
      signal: controller.signal,
    });
    if (epoch !== state.epoch) return;
    const incoming = (result.items || []).map(normalize);
    const seen = new Set(state.items.map(itemKey));
    incoming.forEach((item) => {
      if (!seen.has(item.key)) {
        const old = state.itemMap.get(item.key);
        const merged = {
          ...old,
          ...item,
          content: { ...old?.content, ...item.content },
        };
        state.items.push(merged);
        state.itemMap.set(item.key, merged);
        seen.add(item.key);
      }
    });
    state.cursor = result.cursor;
    state.hasMore = Boolean(result.has_more);
    state.total = result.total || 0;
    state.loading = false;
    $("#feed-pagination").hidden = false;
    renderFeed();
  } catch (error) {
    if (error.name === "AbortError" || epoch !== state.epoch) return;
    state.loading = false;
    if (append) {
      toast(error.message);
      $("#load-more").disabled = false;
    } else
      $("#feed").replaceChildren(
        emptyState("Лента не загрузилась", error.message, "inbox", true),
      );
  } finally {
    if (epoch === state.epoch) {
      state.loading = false;
      $("#refresh-feed").setAttribute("aria-busy", "false");
      $("#feed").setAttribute("aria-busy", "false");
    }
  }
}
async function loadSources() {
  state.controller?.abort();
  const controller = new AbortController();
  state.controller = controller;
  const epoch = ++state.epoch;
  $("#sources-list").replaceChildren(
    statusNode("Открываем список источников…"),
  );
  try {
    const result = await api(
      `/api/sources?folder=${encodeURIComponent(state.folder === "saved" ? "all" : state.folder)}`,
      { signal: controller.signal },
    );
    if (epoch !== state.epoch) return;
    const list = $("#sources-list");
    list.replaceChildren();
    if (!result.sources?.length) {
      list.append(
        emptyState(
          "Источники не привязаны",
          "В локальной конфигурации пока нет источников, закреплённых за этой папкой.",
        ),
      );
      return;
    }
    result.sources.forEach((source) => {
      const node = el("article", "source-card");
      node.append(
        el(
          "div",
          "source-avatar",
          (source.name || "?").slice(0, 1).toUpperCase(),
        ),
      );
      const details = el("div", "source-details"),
        heading = el("h3");
      heading.append(link(source.url, source.name || source.url || "Источник"));
      details.append(
        heading,
        el(
          "p",
          "",
          `${source.kind || "Источник"}${source.last_success_at ? " · собран " + dateTime(source.last_success_at) : ""}`,
        ),
      );
      if (source.topics?.length) {
        const tags = el("div", "source-topics");
        source.topics
          .slice(0, 4)
          .forEach((topic) =>
            tags.append(
              el("span", "", typeof topic === "string" ? topic : String(topic)),
            ),
          );
        details.append(tags);
      }
      node.append(
        details,
        el(
          "span",
          "source-status" + (!source.enabled ? " inactive" : ""),
          source.enabled ? "В радаре" : "Отключён",
        ),
      );
      list.append(node);
    });
  } catch (error) {
    if (error.name !== "AbortError" && epoch === state.epoch)
      $("#sources-list").replaceChildren(
        emptyState(
          "Не удалось открыть источники",
          error.message,
          "inbox",
          true,
        ),
      );
  }
}
function loadView() {
  return state.view === "sources" ? loadSources() : loadFeed();
}
const mutationLocks = new Set();
async function mutateItem(key, action, payload) {
  if (mutationLocks.has(key)) return;
  mutationLocks.add(key);
  const old = state.itemMap.get(key);
  try {
    const result = await api(`/api/items/${key}/${action}`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    if (result.item) updateItem(result.item);
    else updateItem({ ...old, ...payload });
    if (state.folder === "saved" && action === "saved" && !payload.saved) {
      state.items = state.items.filter((i) => itemKey(i) !== key);
      renderFeed();
    } else replaceCard(key);
    if (action === "saved")
      toast(
        payload.saved
          ? "Сохранено в локальной подборке"
          : "Убрано из сохранённого",
      );
  } catch (error) {
    toast(error.message);
  } finally {
    mutationLocks.delete(key);
  }
}
async function expandItem(key, level) {
  if (state.busy.has(key)) return;
  if (level === "brief") {
    state.expansions.set(key, "brief");
    state.errors.delete(key);
    replaceCard(key);
    return;
  }
  const item = state.itemMap.get(key);
  if (bodyFor(item, level) || (level === "long" && item.pdf_url)) {
    state.expansions.set(key, level);
    state.errors.delete(key);
    replaceCard(key);
    return;
  }
  state.errors.delete(key);
  state.busy.set(key, "Открываем сохранённый материал…");
  replaceCard(key);
  try {
    const result = await api(`/api/items/${key}/expand`, {
      method: "POST",
      body: JSON.stringify({ level }),
    });
    if (["failed", "error", "cancelled"].includes(result.status))
      throw new Error(jobError(result.error));
    if (result.job_id) {
      state.busy.set(key, stageLabel(result.stage));
      state.jobs.set(String(result.job_id), {
        key,
        level,
        started: Date.now(),
        attempts: 0,
      });
      replaceCard(key);
      refreshBudget();
      pollJobs();
    } else await finishExpansion(key, level, result);
  } catch (error) {
    state.busy.delete(key);
    state.errors.set(key, error.message);
    replaceCard(key);
    refreshBudget();
  }
}
async function finishExpansion(key, level, result) {
  let item = state.itemMap.get(key);
  try {
    const detail = await api(`/api/items/${key}`);
    item = updateItem(detail.item || detail);
  } catch {}
  if (item) {
    item = {
      ...item,
      content: {
        ...item.content,
        [level]: result.content_html || item.content?.[level] || "",
      },
      pdf_url: result.pdf_url || item.pdf_url,
    };
    updateItem(item);
  }
  state.expansions.set(key, level);
  state.busy.delete(key);
  state.errors.delete(key);
  replaceCard(key);
  refreshBudget();
}
let polling = false,
  pollTimer;
async function pollJobs() {
  if (polling || document.hidden || !state.jobs.size) return;
  clearTimeout(pollTimer);
  polling = true;
  try {
    for (const [id, job] of state.jobs) {
      if (Date.now() - job.started > 15 * 60 * 1000) {
        state.jobs.delete(id);
        state.busy.delete(job.key);
        state.errors.set(
          job.key,
          "Ожидание заняло слишком много времени. Задача может ещё выполняться; попробуй открыть результат позже.",
        );
        replaceCard(job.key);
        continue;
      }
      try {
        const result = await api(`/api/jobs/${encodeURIComponent(id)}`);
        job.attempts = 0;
        if (
          ["ready", "done", "completed", "succeeded"].includes(result.status)
        ) {
          state.jobs.delete(id);
          await finishExpansion(job.key, job.level, result);
        } else if (["failed", "error", "cancelled"].includes(result.status)) {
          state.jobs.delete(id);
          state.busy.delete(job.key);
          state.errors.set(job.key, jobError(result.error));
          replaceCard(job.key);
          refreshBudget();
        } else {
          state.busy.set(job.key, stageLabel(result.stage));
          replaceCard(job.key);
        }
      } catch (error) {
        if (++job.attempts >= 3) {
          state.jobs.delete(id);
          state.busy.delete(job.key);
          state.errors.set(job.key, error.message);
          replaceCard(job.key);
        }
      }
    }
  } finally {
    polling = false;
    if (state.jobs.size && !document.hidden)
      pollTimer = setTimeout(pollJobs, 1800);
  }
}
async function loadChannels() {
  const node = $("#channels-list");
  node.replaceChildren(statusNode("Открываем каналы…"));
  try {
    const result = await api("/api/channels");
    node.replaceChildren();
    if (!result.channels?.length) {
      node.append(
        el(
          "p",
          "dialog-description",
          "Здесь появятся каналы, за которыми ты хочешь следить.",
        ),
      );
      return;
    }
    result.channels.forEach((channel) => {
      const row = el("div", "channel-row");
      row.append(
        icon("video"),
        link(channel.url, channel.name || channel.url || "Канал"),
        el("span", "", channel.enabled ? "В радаре" : "Отключён"),
      );
      node.append(row);
    });
  } catch (error) {
    node.replaceChildren(el("p", "error-text", error.message));
  }
}
function closeNav() {
  $("#app").classList.remove("nav-open");
  $("#nav-scrim").hidden = true;
  $("#mobile-menu").setAttribute("aria-expanded", "false");
}
$("#mobile-menu").addEventListener("click", () => {
  $("#app").classList.add("nav-open");
  $("#nav-scrim").hidden = false;
  $("#mobile-menu").setAttribute("aria-expanded", "true");
});
$("#nav-scrim").addEventListener("click", closeNav);
$("#sidebar").addEventListener("click", (event) => {
  const b = event.target.closest("[data-folder]");
  if (b) selectFolder(b.dataset.folder);
});
$$("[data-view]").forEach((b) =>
  b.addEventListener("click", () => {
    clearTimeout(searchTimer);
    state.query = $("#feed-search").value;
    state.view = b.dataset.view;
    renderHeader();
    loadView();
  }),
);
$(".view-tabs").addEventListener("keydown", (event) => {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
  event.preventDefault();
  const tabs = $$("[data-view]").filter((b) => !b.hidden);
  const index = tabs.indexOf(document.activeElement);
  const next =
    event.key === "Home"
      ? 0
      : event.key === "End"
        ? tabs.length - 1
        : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) %
          tabs.length;
  tabs[next].focus();
  tabs[next].click();
});
$("#refresh-feed").addEventListener("click", () => loadView());
$("#load-more").addEventListener("click", () => loadFeed(true));
$("#clear-region").addEventListener("click", () => updateRegion(null));
$("#region-buttons").addEventListener("click", (event) => {
  const b = event.target.closest("[data-region]");
  if (b)
    updateRegion(state.region === b.dataset.region ? null : b.dataset.region);
});
window.addEventListener("pars-region-select", (event) => {
  if (state.folder === "political_economy") updateRegion(event.detail?.region);
});
let searchTimer;
$("#feed-search").addEventListener("input", (event) => {
  clearTimeout(searchTimer);
  const query = event.target.value;
  searchTimer = setTimeout(() => {
    state.query = query;
    loadFeed();
  }, 300);
});
$("#feed").addEventListener("click", (event) => {
  const b = event.target.closest("[data-action]"),
    node = b?.closest("[data-key]");
  if (!b || !node) return;
  const key = node.dataset.key,
    item = state.itemMap.get(key);
  if (!item) return;
  if (b.dataset.action === "reaction")
    mutateItem(key, "reaction", {
      reaction:
        item.reaction === b.dataset.reaction ? null : b.dataset.reaction,
    });
  if (b.dataset.action === "save")
    mutateItem(key, "saved", { saved: !item.saved });
  if (["expand", "level"].includes(b.dataset.action))
    expandItem(key, b.dataset.level);
});
$("#open-channels").addEventListener("click", () => {
  closeNav();
  $("#channels-dialog").showModal();
  loadChannels();
});
$("#close-channels").addEventListener("click", () =>
  $("#channels-dialog").close(),
);
$("#channels-dialog").addEventListener("click", (event) => {
  if (event.target === $("#channels-dialog")) {
    const rect = event.target.getBoundingClientRect();
    if (
      event.clientX < rect.left ||
      event.clientX > rect.right ||
      event.clientY < rect.top ||
      event.clientY > rect.bottom
    )
      event.target.close();
  }
});
$("#channel-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const submit = $("button[type=submit]", event.target);
  submit.disabled = true;
  $("#channel-form-error").textContent = "";
  try {
    await api("/api/channels", {
      method: "POST",
      body: JSON.stringify({ url: $("#channel-url").value.trim() }),
    });
    $("#channel-url").value = "";
    toast("Канал добавлен в локальный радар. Серверный бот не изменён.");
    await loadChannels();
  } catch (error) {
    $("#channel-form-error").textContent = error.message;
  } finally {
    submit.disabled = false;
    refreshBudget();
  }
});
$("#auth-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const submit = $("button[type=submit]", event.target);
  submit.disabled = true;
  $("#auth-error").textContent = "";
  try {
    const response = await api("/api/session", {
      method: "POST",
      body: JSON.stringify({ token: $("#access-token").value.trim() }),
    });
    state.csrf = response.csrf || "";
    $("#access-token").value = "";
    await initialize();
  } catch (error) {
    $("#auth-error").textContent = error.message;
  } finally {
    submit.disabled = false;
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeNav();
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) pollJobs();
  else clearTimeout(pollTimer);
});
window.addEventListener("pagehide", () => {
  state.controller?.abort();
  clearTimeout(pollTimer);
  clearTimeout(searchTimer);
});
function updateTime() {
  $("#local-time").textContent = new Intl.DateTimeFormat("ru", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date());
}
updateTime();
setInterval(updateTime, 30000);
function renderBudget(budget) {
  const node = $("#budget-status");
  node.hidden = !budget;
  if (!budget) return;
  const usd = (value) => `$${Math.max(0, Number(value) || 0).toFixed(2)}`;
  node.textContent = budget.enabled
    ? `Тесты: ${usd(budget.spent_usd)} / ${usd(budget.limit_usd)}`
    : "Новые запросы выключены";
  node.title = budget.enabled
    ? `Только локальные тесты. Потрачено $${Math.max(0, Number(budget.spent_usd) || 0).toFixed(4)}. Зарезервировано ${usd(budget.reserved_usd)}. Доступно ${usd(budget.remaining_usd)}.`
    : "Новая платная генерация выключена. Сохранённые материалы доступны. Это не баланс серверного аккаунта.";
}
let budgetRefresh = false;
async function refreshBudget() {
  if (budgetRefresh || !state.bootstrap) return;
  budgetRefresh = true;
  try {
    const data = await api("/api/bootstrap");
    state.bootstrap.paid_budget = data.paid_budget;
    if (data.csrf) state.csrf = data.csrf;
    renderBudget(data.paid_budget);
  } catch {
    // Cost refresh must never hide a successfully opened article or report.
  } finally {
    budgetRefresh = false;
  }
}
async function initialize() {
  try {
    const data = await api("/api/bootstrap");
    state.bootstrap = data;
    state.csrf = data.csrf || state.csrf;
    renderFolders();
    renderBudget(data.paid_budget);
    const status = data.status || {};
    $("#snapshot-label").textContent = status.snapshot_at
      ? `Копия · ${dateTime(status.snapshot_at)}`
      : status.latest_at
        ? `Архив · ${dateLabel(status.latest_at)}`
        : "Локальная копия базы";
    $("#snapshot-label").title = status.snapshot_at
      ? "Снимок базы. Локальные изменения не передаются серверному боту."
      : "Дата последнего материала, а не время обновления сервера";
    showApp();
    renderHeader();
    await loadFeed();
  } catch (error) {
    if (error.status === 401) {
      showAuth();
      return;
    }
    showApp();
    $("#feed").replaceChildren(
      emptyState("Не удалось открыть OSVED", error.message, "inbox", true),
    );
  }
}
async function start() {
  const hash = new URLSearchParams(location.hash.slice(1));
  const token = hash.get("token");
  if (token) {
    history.replaceState(null, "", location.pathname + location.search);
    try {
      const result = await api("/api/session", {
        method: "POST",
        body: JSON.stringify({ token }),
      });
      state.csrf = result.csrf || "";
    } catch (error) {
      showAuth();
      $("#auth-error").textContent = error.message;
      return;
    }
  }
  await initialize();
}
start();
