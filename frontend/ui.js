/* ============================================================
   MicroTutor shared UI behaviour: session gate + floating header
   ------------------------------------------------------------
   DEMO GATE ONLY. This stores a name and a role in sessionStorage.
   It is not authentication: both values are chosen in the browser
   and are trivially editable. The server already treats the client
   as untrusted for anything that matters, and real PSU sign in
   replaces this later.
   ============================================================ */
const API = "http://localhost:8000";

/* Theme: "dark" (default) or "light", per browser. Applied to <html
   data-theme> so the light token overrides in ui.css take effect. Each page's
   <head> sets this synchronously to avoid a flash on load; this is the setter
   the Settings dialog calls, plus a fallback apply for pages that miss the
   head snippet. */
const Theme = {
  key: "mt.theme",
  get(){
    try { return localStorage.getItem(this.key) === "light" ? "light" : "dark"; }
    catch { return "dark"; }
  },
  set(t){
    const v = t === "light" ? "light" : "dark";
    try { localStorage.setItem(this.key, v); } catch {}
    document.documentElement.dataset.theme = v;
  },
};
Theme.set(Theme.get());

const Session = {
  key: "microtutor.session",
  get(){
    try { return JSON.parse(sessionStorage.getItem(this.key) || "null"); }
    catch { return null; }
  },
  set(name, role){
    sessionStorage.setItem(this.key, JSON.stringify({name, role}));
  },
  clear(){ sessionStorage.removeItem(this.key); },
};

/* Send anyone without a session back to the gate. `role` optionally
   pins a page to one side, so a student cannot land on the instructor
   upload screen by typing the URL. */
function requireSession(role){
  const s = Session.get();
  if (!s || !s.name || (role && s.role !== role)) {
    location.replace("home.html");
    return null;
  }
  return s;
}

const initials = n => (n || "?").trim().split(/\s+/).slice(0, 2)
  .map(w => w[0]).join("").toUpperCase();

const esc = s => String(s == null ? "" : s)
  .replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

/* Floating glass header: navigation left, wordmark centre, account right.
   `active` names the current page so its nav button can be marked. */
function mountHeader({active = "", wide = false} = {}){
  const s = Session.get();
  // Tab title: plain "MicroTutor" for everyone except an instructor, whose
  // pages read "MicroTutor Portal". Set here so every page that mounts the
  // header inherits it without its own <title> logic.
  document.title = (s && s.role === "teacher") ? "MicroTutor Portal" : "MicroTutor";
  const roleHome = s && s.role === "teacher" ? "teacher.html" : "student.html";

  // SVG, never an emoji or a dingbat glyph like the old &#8962;: those render
  // as a different picture on every OS and ignore the surrounding text colour.
  const homeIcon = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" stroke-width="2" stroke-linecap="round"
    stroke-linejoin="round" aria-hidden="true">
    <path d="M3 10.2 12 3l9 7.2V20a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z"/></svg>`;
  const listIcon = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" stroke-width="2" stroke-linecap="round"
    stroke-linejoin="round" aria-hidden="true">
    <path d="M8 6h13M8 12h13M8 18h13M3.5 6h.01M3.5 12h.01M3.5 18h.01"/></svg>`;

  // "Practice" is gone. It was a single-item nav pointing at the page the
  // student was already on, so it navigated nowhere and cost the centre of the
  // bar. Home covers it. An instructor keeps one extra destination, which sits
  // on the LEFT beside Home rather than in the middle - the middle is the mark.
  const nav = [`<a class="hbtn ${active === "Home" || !active ? "on" : ""}"
      id="homeBtn" href="${roleHome}">${homeIcon}<span class="htext">Home</span></a>`];
  if (s && s.role === "teacher"){
    nav.push(`<a class="hbtn ${active === "Assignments" ? "on" : ""}"
      href="teacher.html#list">${listIcon}<span class="htext">Assignments</span></a>`);
  }

  const hdr = document.createElement("header");
  hdr.className = "hdr" + (wide ? " wide" : "");
  hdr.innerHTML = `
    <nav class="hnav" aria-label="Main">${nav.join("")}</nav>
    <a class="wordmark" href="${roleHome}" aria-label="MicroTutor home">
      <span class="dot" aria-hidden="true"></span>MicroTutor</a>
    <div class="account hacct" id="acct">
      <button class="who" id="whoBtn" aria-haspopup="true" aria-expanded="false">
        <span class="avatar">${esc(initials(s && s.name))}</span>
        <span class="nm">${esc(s ? s.name : "guest")}</span>
        <span class="caret" aria-hidden="true">&#9662;</span>
      </button>
      <div class="menu" id="whoMenu" role="menu" hidden>
        <button class="mi" id="miSettings" role="menuitem">Settings</button>
        <button class="mi mi-danger" id="miLogout" role="menuitem">Log out</button>
      </div>
    </div>`;
  document.body.prepend(hdr);

  // ---- profile menu -------------------------------------------------------
  const acct = hdr.querySelector("#acct");
  const whoBtn = hdr.querySelector("#whoBtn");
  const whoMenu = hdr.querySelector("#whoMenu");
  const closeMenu = () => {
    whoMenu.hidden = true;
    whoBtn.setAttribute("aria-expanded", "false");
  };
  whoBtn.addEventListener("click", e => {
    e.stopPropagation();
    const open = whoMenu.hidden;
    whoMenu.hidden = !open;
    whoBtn.setAttribute("aria-expanded", String(open));
  });
  document.addEventListener("click", e => {
    if (!acct.contains(e.target)) closeMenu();
  });
  addEventListener("keydown", e => { if (e.key === "Escape") closeMenu(); });

  hdr.querySelector("#miLogout").onclick = () => {
    Session.clear();
    location.href = "home.html";
  };
  hdr.querySelector("#miSettings").onclick = () => {
    closeMenu();
    openSettings(s);
  };

  // The sliding "glider" highlight that used to track the active pill is gone
  // with the pills themselves. It measured offsets on every resize and hover to
  // animate a bar under a nav that, for a student, had exactly one item.

  // Condense on scroll.
  let ticking = false;
  addEventListener("scroll", () => {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => {
      hdr.classList.toggle("condensed", scrollY > 24);
      ticking = false;
    });
  }, {passive: true});

  return hdr;
}

/* Settings dialog opened from the profile menu. Lazily built and reused.
   Real preferences land here once PSU sign-in exists; for now it shows the
   account the demo gate recorded. */
function openSettings(s){
  let ov = document.getElementById("mtSettings");
  if (!ov){
    ov = document.createElement("div");
    ov.id = "mtSettings";
    ov.className = "modal";
    ov.hidden = true;
    ov.innerHTML = `
      <div class="modalCard" role="dialog" aria-modal="true" aria-labelledby="mtsTitle">
        <div class="modalHead">
          <h2 id="mtsTitle">Settings</h2>
          <button class="modalX" id="mtsX" aria-label="Close">&times;</button>
        </div>
        <div class="modalBody" id="mtsBody"></div>
      </div>`;
    document.body.appendChild(ov);
    ov.addEventListener("click", e => { if (e.target === ov) ov.hidden = true; });
    ov.querySelector("#mtsX").onclick = () => { ov.hidden = true; };
    addEventListener("keydown", e => { if (e.key === "Escape") ov.hidden = true; });
  }
  ov.querySelector("#mtsBody").innerHTML = `
    <div class="setRow"><span>Name</span><b>${esc(s ? s.name : "guest")}</b></div>
    <div class="setRow"><span>Role</span><b>${esc(s ? s.role : "—")}</b></div>
    <div class="setRow"><span>Theme</span>
      <span class="seg" id="mtsTheme">
        <button type="button" data-t="dark">Dark</button>
        <button type="button" data-t="light">Light</button>
      </span>
    </div>
    <p class="setNote">Account preferences arrive with Penn State sign-in. For
      now, use <b>Log out</b> to switch name or role.</p>`;

  const seg = ov.querySelector("#mtsTheme");
  const paintSeg = () => seg.querySelectorAll("button").forEach(b =>
    b.classList.toggle("on", b.dataset.t === Theme.get()));
  seg.querySelectorAll("button").forEach(b =>
    b.onclick = () => { Theme.set(b.dataset.t); paintSeg(); });
  paintSeg();

  ov.hidden = false;
}
