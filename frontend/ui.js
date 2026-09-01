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

/* Floating glass header. `active` names the current nav item so the
   glider can sit under it. */
function mountHeader({active = "", wide = false} = {}){
  const s = Session.get();
  // Tab title: plain "MicroTutor" for everyone except an instructor, whose
  // pages read "MicroTutor Portal". Set here so every page that mounts the
  // header inherits it without its own <title> logic.
  document.title = (s && s.role === "teacher") ? "MicroTutor Portal" : "MicroTutor";
  const roleHome = s && s.role === "teacher" ? "teacher.html" : "student.html";
  const links = s && s.role === "teacher"
    ? [["teacher.html", "Upload"], ["teacher.html#list", "Assignments"]]
    : [["student.html", "Practice"]];

  const hdr = document.createElement("header");
  hdr.className = "hdr" + (wide ? " wide" : "");
  hdr.innerHTML = `
    <a class="brand" href="${roleHome}"><span class="dot"></span>MicroTutor</a>
    <a class="hbtn" id="homeBtn" href="${roleHome}"><span aria-hidden="true">&#8962;</span><span class="htext">Home</span></a>
    <span class="spacer"></span>
    <nav class="navpills" id="np">
      <span class="glider" id="glider"></span>
      ${links.map(([h, t]) =>
        `<a href="${h}" class="${t === active ? "on" : ""}">${t}</a>`).join("")}
    </nav>
    <div class="account" id="acct">
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

  // Slide the highlight under the active pill, and let it follow the
  // cursor on hover so the header responds rather than sitting still.
  const nav = hdr.querySelector("#np"), glider = hdr.querySelector("#glider");
  const items = [...nav.querySelectorAll("a")];
  const moveTo = el => {
    if (!el) { glider.style.width = "0px"; return; }
    glider.style.width = el.offsetWidth + "px";
    // .navpills is position:relative, so it is the offsetParent for BOTH the
    // glider and the links: el.offsetLeft is already measured from it. The old
    // `- nav.offsetLeft` also subtracted the nav's distance from the header's
    // left edge (~1400px on a wide header), throwing the glider across the bar
    // and parking it on top of the brand as a stray indigo blob.
    glider.style.transform = `translateX(${el.offsetLeft}px)`;
  };
  const home = () => moveTo(items.find(a => a.classList.contains("on")) || items[0]);
  items.forEach(a => a.addEventListener("mouseenter", () => moveTo(a)));
  nav.addEventListener("mouseleave", home);
  requestAnimationFrame(home);
  addEventListener("resize", home);

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
