"use strict";

const state = { clicks: [], started: performance.now(), generation: 0, categories: [], ready: false };
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
async function request(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Please try again");
  return data;
}
const productName = p => p.name || "Product " + p.item_id;
const price = p => p.price === undefined ? "Price unavailable" : "SGD " + Number(p.price).toFixed(2);
const routeFor = (page, id) => "#/" + page + "/" + id;
const categoryName = id => state.categories.find(c => c.category_id === id)?.name || "Category " + id;
function image(p) {
  const image = node("span", undefined, p.image === undefined ? "product-image unavailable" : "product-image image-" + p.image);
  image.setAttribute("role", "img");
  image.setAttribute("aria-label", p.alt || "Image unavailable");
  if (p.image === undefined) image.textContent = "Image unavailable";
  return image;
}
function card(p) {
  const a = link("", routeFor("item", p.item_id), "card");
  a.append(image(p), node("span", productName(p), "product-name"), node("span", price(p), "price"));
  a.addEventListener("click", event => {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    if (state.clicks.length >= 10000) {
      event.preventDefault();
      el("notice").textContent = "This visit is full. Start a new session to continue.";
      return;
    }
    state.clicks.push({ item_id: p.item_id, time_ms: Math.max(0, Math.floor(performance.now() - state.started)) });
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
function grid(products) {
  const result = node("div", undefined, "grid");
  products.forEach(p => result.append(card(p)));
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
  return { clicks: state.clicks, now_ms: Math.max(Math.floor(performance.now() - state.started), last ? last.time_ms + 1 : 0), limit: 40 };
}
async function showList(route, view, generation) {
  if (route.page === "home") categoryTiles(view);
  const section = node("section");
  const heading = node("h2", route.page === "category" ? categoryName(route.id) : "Popular now");
  const results = node("div", "Loading products…");
  section.append(heading, results); view.append(section);
  const load = async () => {
    results.textContent = "Loading products…";
    try {
      const result = route.page === "home"
        ? await request("/recommend/home", { clicks: state.clicks })
        : await request("/recommend/category", { ...sessionBody(), category_id: route.id });
      const products = await metadata(result.items);
      if (generation !== state.generation) return;
      if (route.page === "home") heading.textContent = result.source.startsWith("popularity") ? "Popular now" : "For you";
      results.replaceChildren();
      if (!products.length) { results.textContent = "No products available"; return; }
      const first = route.page === "category" ? 20 : products.length;
      const display = grid(products.slice(0, first));
      results.append(display);
      if (products.length > first) {
        const more = node("button", "Show more");
        more.type = "button";
        more.addEventListener("click", () => { products.slice(first).forEach(p => display.append(card(p))); more.remove(); });
        results.append(more);
      }
    } catch (e) { if (generation === state.generation) error(results, "Products could not be loaded", load); }
  };
  await load();
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
        results.replaceChildren(products.length ? grid(products) : node("p", "No related products available"));
      } catch (e) { if (generation === state.generation) error(results, "Related products could not be loaded", loadRelated); }
    };
    await loadRelated();
  } catch (e) { if (generation === state.generation) error(detail, "Product could not be loaded", () => render()); }
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
  const current = route();
  if (current.page === "item") await showItem(current, view, generation);
  else await showList(current, view, generation);
}
el("reset").addEventListener("click", () => {
  state.clicks = []; state.started = performance.now(); state.generation++;
  if (location.hash !== "#/") location.hash = "#/"; else render();
});
window.addEventListener("hashchange", render);
async function boot() {
  try {
    const status = await request("/demo/status");
    if (!status.ready) { el("notice").textContent = "The demo is not ready: " + status.issues.join(", "); return; }
    state.categories = (await request("/demo/categories")).categories;
    state.ready = true; navigation(); render();
  } catch (e) { error(el("notice"), "Clew could not be loaded", boot); }
}
boot();
