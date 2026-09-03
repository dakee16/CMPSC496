/* ============================================================
   MicroTutor shared UI behaviour: session + floating header
   ------------------------------------------------------------
   Sign-in is username (PSU email) + password, and the thing that
   proves it is an HttpOnly cookie this file cannot read. What
   sessionStorage holds below is a COPY for drawing the header -
   a name and a role. Editing it changes what the avatar says and
   nothing else: every server route re-reads the cookie, so a
   forged role here buys a 403, not an upload screen.
   ============================================================ */

/* Same origin as the pages, because api_server.py serves them (see the
   StaticFiles mount at the bottom of that file). An empty string makes every
   fetch a relative URL, so the cookie rides along with no CORS involved. The
   localhost fallback is only for opening these files straight off disk, where
   there is no origin to be the same as. */
const API = location.protocol.startsWith("http") ? "" : "http://localhost:8000";

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
  set(me){
    sessionStorage.setItem(this.key, JSON.stringify(
      {name: me.name, role: me.role, id: me.student_id}));
    return me;
  },
  clear(){ try { sessionStorage.removeItem(this.key); } catch {} },

  /* Ask the SERVER who we are. Returns the account or null. This is the only
     honest answer: the cookie can expire mid-lab, and a page that trusted
     sessionStorage would keep drawing a signed-in header while every save
     silently 401'd. */
  async check(){
    try {
      const r = await fetch(`${API}/auth/me`, {credentials: "include"});
      if (!r.ok) { this.clear(); return null; }
      return this.set(await r.json());
    } catch {
      return null;              // server unreachable is not "signed out"
    }
  },

  /* `mode` is "login" or "register". Throws an Error whose message is meant
     for the student - the server writes it, so there is one wording for a bad
     password and not one per page. */
  async signIn(mode, username, password){
    let r;
    try {
      r = await fetch(`${API}/${mode === "register" ? "register" : "login"}`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        credentials: "include",
        body: JSON.stringify({username, password}),
      });
    } catch {
      throw new Error("Could not reach the server. Are you on the VPN?");
    }
    const body = await r.json().catch(() => ({}));
    if (!r.ok){
      const d = body.detail;
      throw new Error((d && d.message) || (typeof d === "string" && d)
                      || "Sign-in failed. Try again.");
    }
    return this.set(body);
  },

  async signOut(){
    try { await fetch(`${API}/logout`, {method: "POST", credentials: "include"}); }
    catch {}                    // the cookie expires on its own regardless
    this.clear();
  },
};

/* Send anyone without a session to sign-in. `role` optionally pins a page to
   one side, so a student cannot land on the instructor upload screen by
   typing the URL.

   Synchronous on purpose: it returns the cached account so callers can keep
   `const S = requireSession("student")` at the top of a plain script. That
   cache is only a HINT - it is revalidated against /auth/me a moment later,
   and it decides nothing on the server. */
function requireSession(role){
  const s = Session.get();
  if (!s || !s.name || (role && s.role !== role)) {
    location.replace("login.html");
    return null;
  }
  // Confirm with the server without blocking the page. A cookie that expired
  // while the tab sat open lands back on sign-in instead of failing later, on
  // a save the student thought had gone through.
  Session.check().then(me => {
    if (!me || (role && me.role !== role)) location.replace("login.html");
  });
  return s;
}

/* Any route may answer 401 once the cookie expires. Handling that in one place
   beats threading a check through every fetch on every page - and a page that
   ignores it shows stale work as though it were still being saved. */
const _fetch = window.fetch.bind(window);
window.fetch = async (input, init = {}) => {
  // Send the cookie by default. Same-origin would do this anyway; saying it
  // explicitly is what makes a cross-origin dev server (vite on :5173 against
  // the API on :8000) behave the same as production instead of looking
  // signed-out for reasons no error message explains.
  const r = await _fetch(input, {credentials: "include", ...init});
  if (r.status === 401 && !location.pathname.endsWith("login.html")){
    Session.clear();
    location.replace("login.html");
  }
  return r;
};

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

  hdr.querySelector("#miLogout").onclick = async () => {
    // Await it: clearing only sessionStorage would leave the cookie valid, so
    // the next page load would sign straight back in.
    await Session.signOut();
    location.href = "login.html";
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
   Shows the signed-in account; preferences beyond the theme land here when
   there is a second one worth storing. */
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
    <div class="setRow"><span>Account</span><b>${esc(s ? s.name : "guest")}</b></div>
    <div class="setRow"><span>Role</span><b>${esc(s ? s.role : "—")}</b></div>
    <div class="setRow"><span>Theme</span>
      <span class="seg" id="mtsTheme">
        <button type="button" data-t="dark">Dark</button>
        <button type="button" data-t="light">Light</button>
      </span>
    </div>
    <p class="setNote">Your role is set by the course staff, not here. Ask your
      instructor if it looks wrong.</p>`;

  const seg = ov.querySelector("#mtsTheme");
  const paintSeg = () => seg.querySelectorAll("button").forEach(b =>
    b.classList.toggle("on", b.dataset.t === Theme.get()));
  seg.querySelectorAll("button").forEach(b =>
    b.onclick = () => { Theme.set(b.dataset.t); paintSeg(); });
  paintSeg();

  ov.hidden = false;
}
