"use strict";

// One visit per browser tab: clicks survive a reload (sessionStorage) and end with the
// tab or with "Start new session". Nothing is kept across tabs or visits.
const VISIT_KEY = "clew-visit";
const CATEGORY_LIMIT = 500;   // the API's maximum
const PAGE_SIZE = 20;
const state = { clicks: [], started: Date.now(), sessionId: "", generation: 0, categories: [], ready: false,
                logging: false };
const el = id => document.getElementById(id);
const node = (tag, value, cls) => {
  const n = document.createElement(tag);
  if (value !== undefined) n.textContent = value;
  if (cls) n.className = cls;
  return n;
};
const link = (label, href, cls) => {
  const a = node("a", label, cls);
  a.href = href;
  return a;
};
function newSessionId() {
  try { return crypto.randomUUID(); } catch (e) { return Date.now().toString(36) + Math.random().toString(36).slice(2); }
}
function saveVisit() {
  try {
    sessionStorage.setItem(VISIT_KEY, JSON.stringify({ clicks: state.clicks, started: state.started,
                                                       sessionId: state.sessionId }));
  } catch (e) { /* storage blocked: the visit simply ends on reload */ }
}
function startVisit() {
  state.clicks = []; state.started = Date.now(); state.sessionId = newSessionId();
  saveVisit();
}
function loadVisit() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(VISIT_KEY) || "null");
    if (saved && Array.isArray(saved.clicks) && Number.isFinite(saved.started) && typeof saved.sessionId === "string") {
      state.clicks = saved.clicks.filter(c => c && Number.isInteger(c.item_id) && c.item_id >= 0
                                              && Number.isInteger(c.time_ms) && c.time_ms >= 0).slice(-10000);
      state.started = saved.started;
      state.sessionId = saved.sessionId.slice(0, 64);
      return;
    }
  } catch (e) { /* unreadable or blocked storage: start a new visit */ }
  startVisit();
}
const elapsed = () => Math.max(0, Math.floor(Date.now() - state.started));
const viewed = id => state.clicks.some(c => c.item_id === id);

async function request(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  let data = {};
  try { data = await response.json(); } catch (e) { /* no JSON body */ }
  if (!response.ok) {
    const failure = new Error(data.error || "Please try again");
    failure.status = response.status;
    throw failure;
  }
  return data;
}
// Interaction logging (only when the server was started with --log; HANDOFF.md section 4).
function log(event) {
  if (!state.logging) return;
  const body = { session_id: state.sessionId, time_ms: elapsed(), ...event };
  fetch("/log", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
                  keepalive: true }).catch(() => {});
}
function impression(list) {
  log({ event: "impression", page: list.page, shown: list.shown, list_source: list.source,
        category_id: list.categoryId, context_item_id: list.contextItemId });
}

const productName = p => p.name || "Product " + p.item_id;
const price = p => p.price === undefined ? "Price unavailable" : "SGD " + Number(p.price).toFixed(2);
const routeFor = (page, id) => "#/" + page + "/" + id;
const categoryInfo = id => state.categories.find(c => c.category_id === id);
const categoryName = id => categoryInfo(id)?.name || "Category " + id;
function image(p) {
  const image = node("span", undefined, p.image === undefined ? "product-image unavailable" : "product-image image-" + p.image);
  image.setAttribute("role", "img");
  image.setAttribute("aria-label", p.alt || "Image unavailable");
  if (p.image === undefined) image.textContent = "Image unavailable";
  return image;
}
// `list` describes where the card is shown (page, source, discovery places) for labels and logs;
// `position` is the card's 1-based place in that list.
function card(p, list, position) {
  const a = link("", routeFor("item", p.item_id), "card");
  // The picture carries the product's category (top left) and visit labels (bottom left).
  const media = node("span", undefined, "media");
  media.append(image(p));
  if (p.category_id !== null && p.category_id !== undefined) {
    const top = node("span", undefined, "tags top");
    top.append(node("span", categoryName(p.category_id), "tag category"));
    media.append(top);
  }
  const bottom = node("span", undefined, "tags bottom");
  if (list?.discovery?.has(position)) bottom.append(node("span", "Something different", "tag discovery"));
  if (viewed(p.item_id)) bottom.append(node("span", "Viewed", "tag viewed"));
  if (bottom.childElementCount) media.append(bottom);
  a.append(media, node("span", productName(p), "product-name"), node("span", price(p), "price"));
  a.addEventListener("click", event => {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    if (state.clicks.length >= 10000) {
      event.preventDefault();
      el("notice").textContent = "This visit is full. Start a new session to continue.";
      return;
    }
    state.clicks.push({ item_id: p.item_id, time_ms: elapsed() });
    saveVisit();
    if (list) log({ event: "click", page: list.page, item_id: p.item_id, position, list_source: list.source,
                    category_id: list.categoryId, context_item_id: list.contextItemId });
    // A second activation of the current route still refreshes recommendations.
    if (location.hash === a.hash) { event.preventDefault(); render(); }
  });
  return a;
}
async function metadata(ids) {
  if (!ids.length) return [];
  const result = await request("/demo/products?ids=" + [...new Set(ids)].join(","));
  const byId = new Map(result.products.map(p => [p.item_id, p]));
  return ids.map(id => byId.get(id) || { item_id: id, category_id: null });
}
function grid(products, list, offset = 0) {
  const result = node("div", undefined, "grid");
  products.forEach((p, i) => result.append(card(p, list, offset + i + 1)));
  return result;
}
function error(container, message, retry) {
  const button = node("button", "Try again");
  button.type = "button";
  button.addEventListener("click", retry);
  container.replaceChildren(node("p", message), button);
}
function navigation() {
  const nav = el("categories");
  nav.replaceChildren();
  for (const c of state.categories.filter(c => c.featured)) nav.append(link(c.name, routeFor("category", c.category_id)));
}
function categoryTiles(container) {
  const section = node("section");
  section.append(node("h1", "Shop by category"));
  const tiles = node("div", undefined, "category-grid");
  section.append(tiles); container.append(section);
  const categories = state.categories.filter(c => c.featured);
  metadata(categories.map(c => c.cover)).then(products => {
    if (!tiles.isConnected) return;
    categories.forEach((c, i) => {
      const a = link("", routeFor("category", c.category_id), "tile");
      a.append(image(products[i]), node("span", c.name));
      tiles.append(a);
    });
  }).catch(() => { if (tiles.isConnected) tiles.textContent = "Categories are unavailable"; });
}
function sessionBody() {
  const last = state.clicks.at(-1);
  return { clicks: state.clicks, now_ms: Math.max(elapsed(), last ? last.time_ms + 1 : 0), limit: CATEGORY_LIMIT };
}
async function showHome(view, generation) {
  categoryTiles(view);
  const section = node("section");
  const heading = node("h2", "Popular now");
  const results = node("div", "Loading products…");
  section.append(heading, results); view.append(section);
  const load = async () => {
    results.textContent = "Loading products…";
    try {
      const result = await request("/recommend/home", { clicks: state.clicks });
      const products = await metadata(result.items);
      if (generation !== state.generation) return;
      heading.textContent = result.source.startsWith("popularity") ? "Popular now" : "For you";
      const list = { page: "home", source: result.source, shown: result.items,
                     discovery: new Set(result.diverse_positions || []) };
      results.replaceChildren();
      if (!products.length) { results.textContent = "No products available"; return; }
      if (list.discovery.size) {
        results.append(node("p", `The last ${list.discovery.size} are picked to be different from the first `
                                 + `${products.length - list.discovery.size}, so there is something new to discover.`,
                            "caption"));
      }
      results.append(grid(products, list));
      impression(list);
    } catch (e) { if (generation === state.generation) error(results, "Products could not be loaded", load); }
  };
  await load();
}
async function showCategory(route, view, generation) {
  const section = node("section");
  const heading = node("h2", categoryName(route.id));
  const results = node("div", "Loading products…");
  section.append(heading, results); view.append(section);
  const load = async () => {
    results.textContent = "Loading products…";
    try {
      const result = await request("/recommend/category", { ...sessionBody(), category_id: route.id });
      if (generation !== state.generation) return;
      const ids = result.items;
      results.replaceChildren();
      if (!ids.length) { results.textContent = "This category has no products."; return; }
      const total = Math.max(categoryInfo(route.id)?.count || ids.length, ids.length);
      const list = { page: "category", source: result.source, categoryId: route.id, shown: [] };
      const display = node("div", undefined, "grid");
      const count = node("p", "", "caption");
      const more = node("button", "Show more");
      more.type = "button";
      results.append(display, count);
      const reveal = async () => {
        more.disabled = true;
        const batch = ids.slice(list.shown.length, list.shown.length + PAGE_SIZE);
        const products = await metadata(batch);
        if (generation !== state.generation) return;
        products.forEach((p, i) => display.append(card(p, list, list.shown.length + i + 1)));
        list.shown = ids.slice(0, list.shown.length + batch.length);
        impression(list);
        const shownText = `Showing ${list.shown.length.toLocaleString()} of ${total.toLocaleString()} products`;
        count.textContent = list.shown.length === ids.length && total > ids.length
          ? `${shownText} (the top ${ids.length.toLocaleString()} for this visit)` : shownText;
        more.disabled = false;
        if (list.shown.length < ids.length) results.append(more); else more.remove();
      };
      more.addEventListener("click", () => reveal().catch(() => {
        more.disabled = false;
        count.textContent = "More products could not be loaded. Please try again.";
      }));
      await reveal();
    } catch (e) { if (generation === state.generation) error(results, "Products could not be loaded", load); }
  };
  await load();
}
function notFound(container) {
  container.className = "";
  container.replaceChildren(node("h1", "Product not found"),
                            node("p", "This product is not in the catalog."), link("Back to the home page", "#/"));
}
async function showItem(route, view, generation) {
  const detail = node("section", "Loading product…");
  view.append(detail);
  try {
    const [product] = await metadata([route.id]);
    if (generation !== state.generation) return;
    detail.replaceChildren();
    detail.className = "detail";
    const info = node("div");
    info.append(node("h1", productName(product)), node("p", price(product), "detail-price"));
    if (product.description) info.append(node("p", product.description));
    if (product.category_id !== null) info.append(link(categoryName(product.category_id), routeFor("category", product.category_id)));
    detail.append(image(product), info);
    const section = node("section");
    const results = node("div", "Loading related products…");
    section.append(node("h2", "You may also like"), results);
    view.append(section);
    const loadRelated = async () => {
      results.textContent = "Loading related products…";
      try {
        const result = await request("/recommend/item/" + route.id);
        const products = await metadata(result.items);
        if (generation !== state.generation) return;
        const list = { page: "item", source: result.source, contextItemId: route.id, shown: result.items };
        results.replaceChildren(products.length ? grid(products, list) : node("p", "No related products available"));
        if (products.length) impression(list);
      } catch (e) { if (generation === state.generation) error(results, "Related products could not be loaded", loadRelated); }
    };
    await loadRelated();
  } catch (e) {
    if (generation !== state.generation) return;
    if (e.status === 404) notFound(detail);
    else error(detail, "Product could not be loaded", () => render());
  }
}
function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/");
  if (["item", "category"].includes(parts[0]) && /^\d+$/.test(parts[1])) return { page: parts[0], id: Number(parts[1]) };
  return { page: "home" };
}
async function render() {
  if (!state.ready) return;
  const generation = ++state.generation;
  const view = el("view");
  view.replaceChildren(); el("notice").replaceChildren();
  el("content").focus({ preventScroll: true });
  window.scrollTo(0, 0);
  const current = route();
  if (current.page === "item") await showItem(current, view, generation);
  else if (current.page === "category") await showCategory(current, view, generation);
  else await showHome(view, generation);
}
el("reset").addEventListener("click", () => {
  startVisit(); state.generation++;
  if (location.hash !== "#/") location.hash = "#/"; else render();
});
window.addEventListener("hashchange", render);
async function boot() {
  loadVisit();
  try {
    const status = await request("/demo/status");
    if (!status.ready) {
      el("view").replaceChildren();
      el("notice").textContent = "The demo is not ready: " + status.issues.join(", ");
      return;
    }
    state.logging = status.logging === true;
    el("privacy").hidden = !state.logging;
    state.categories = (await request("/demo/categories")).categories;
    state.ready = true; navigation(); render();
  } catch (e) { el("view").replaceChildren(); error(el("notice"), "Clew could not be loaded", boot); }
}
boot();
