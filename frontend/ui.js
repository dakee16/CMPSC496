/* ============================================================
   ACADIA shared UI behaviour: session + app header
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

/* Opened by double-clicking the .html instead of visiting the server.

   This CANNOT work, and it fails in the most confusing way available: a
   file:// page has an opaque origin, so the browser stores no SameSite=Lax
   cookie from it. Sign-in answers 200 with the account, the page cheers, and
   the very next request is signed out again - so the user is bounced back to
   the login form having typed the right password. Saying so up front costs one
   comparison and saves that entire loop. */
const OFF_DISK = location.protocol === "file:";
const OFF_DISK_WHY =
  "Open the app at http://localhost:8000 instead of from a file. A page "
  + "opened straight off disk cannot keep you signed in - the browser throws "
  + "the session cookie away, so sign-in succeeds and then immediately "
  + "forgets you.";

/* One banner, on whatever page loaded, rather than only on the sign-in button:
   every page is equally broken off disk, and the student page would otherwise
   just redirect-loop back to login with nothing to read. */
function offDiskBanner(){
  if (!OFF_DISK || document.getElementById("mtOffDisk")) return;
  const b = document.createElement("div");
  b.id = "mtOffDisk";
  b.className = "banner bad";
  b.setAttribute("role", "alert");
  b.style.cssText = "position:fixed; inset:auto 12px 12px; z-index:9999; margin:0;"
    + " max-width:560px; margin-inline:auto";
  b.textContent = OFF_DISK_WHY;
  (document.body || document.documentElement).appendChild(b);
}
if (OFF_DISK) addEventListener("DOMContentLoaded", offDiskBanner);

/* Theme: "light" (default) or an explicitly saved "dark", per browser. Applied to <html
   data-theme> so the palette in tokens.css takes effect. Each page's
   <head> sets this synchronously to avoid a flash on load; this is the setter
   the Settings dialog calls, plus a fallback apply for pages that miss the
   head snippet. */
const Theme = {
  key: "mt.theme",
  get(){
    try { return localStorage.getItem(this.key) === "dark" ? "dark" : "light"; }
    catch { return "light"; }
  },
  set(t){
    const v = t === "dark" ? "dark" : "light";
    try { localStorage.setItem(this.key, v); } catch {}
    document.documentElement.dataset.theme = v;
    syncThemeControls(v);
    // Keep the browser's own chrome (the mobile address bar) on the same ground
    // as the page. Read from the token so --bg stays the single source of the
    // page's colour; the literal in each page's <meta> is only the value used
    // before this file has run.
    const m = document.querySelector('meta[name="theme-color"]');
    const bg = getComputedStyle(document.documentElement)
      .getPropertyValue("--bg").trim();
    if (m && bg) m.setAttribute("content", bg);
  },
};
/* Theme controls only change this browser's visual preference. */
function syncThemeControls(theme){
  document.querySelectorAll('[data-theme-toggle]').forEach(button => {
    const label = `Switch to ${theme === "dark" ? "light" : "dark"} mode`;
    button.setAttribute("aria-label", label);
    button.title = label;
  });
  document.querySelectorAll('#mtsTheme [data-t]').forEach(button => {
    const selected = button.dataset.t === theme;
    button.classList.toggle("on", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
}
function themeToggle(){
  return `<button class="theme-toggle" type="button" data-theme-toggle
      aria-label="Switch color theme">
    <svg class="sun-icon" width="18" height="18" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" stroke-width="1.7" stroke-linecap="round" aria-hidden="true">
      <circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/>
    </svg>
    <svg class="moon-icon" width="18" height="18" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M20.7 13.2A9 9 0 0 1 10.8 3.3a9 9 0 1 0 9.9 9.9Z"/>
    </svg>
  </button>`;
}
document.querySelectorAll('[data-theme-control]').forEach(host => {
  host.innerHTML = themeToggle();
});
document.addEventListener("click", event => {
  if (event.target.closest('[data-theme-toggle]')) {
    Theme.set(Theme.get() === "dark" ? "light" : "dark");
  }
});
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
  async signIn(mode, username, password, extra = {}){
    // Refuse before the round trip. The request would SUCCEED and the session
    // would still be lost, which is the one failure the student cannot debug.
    if (OFF_DISK) throw new Error(OFF_DISK_WHY);
    let r;
    try {
      r = await fetch(`${API}/${mode === "register" ? "register" : "login"}`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        credentials: "include",
        body: JSON.stringify({username, password,
                              ...(mode === "register" ? extra : {})}),
      });
    } catch {
      // fetch() only throws on a TRANSPORT failure - the server is down, or the
      // browser refused the request before sending it (a page opened straight
      // off disk, blocked by CORS). It cannot tell those apart, so it must not
      // guess: this used to say "are you on the VPN?", naming the one cause it
      // had no evidence for and sending people to check a connection that was
      // never the problem. Sign-in has NO VPN gate - see main/auth.py.
      throw new Error("Could not reach the server. Check that it is running, "
                      + "then try again.");
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
/* May an account with `role` open a page pinned to `page`?

   A teacher may open the student pages - that IS the "View as student" switch
   in the header, and it is the same permission they already have: every
   student route asks only for a signed-in account, so a teacher working
   through a problem is an ordinary student session bound to their own id. A
   student on an instructor page is still turned away here, and again by
   require_teacher() on every instructor route, which is the check that
   actually decides anything. */
const allowedOn = (page, role) => !page || role === page || role === "teacher";

function requireSession(role){
  const s = Session.get();

  // Confirm with the server without blocking the page. A cookie that expired
  // while the tab sat open lands back on sign-in instead of failing later, on
  // a save the student thought had gone through.
  const confirm = Session.check().then(me => {
    if (!me || !allowedOn(role, me.role)) { location.replace("login.html"); return null; }
    return me;
  });

  if (s && s.name && allowedOn(role, s.role)) return s;

  // No CACHED session - which is not the same thing as being signed out, and
  // treating it as such is how "watch" appeared broken. Session lives in
  // sessionStorage, which is per-TAB: any link opened with target="_blank"
  // starts with an empty one while the cookie, the actual credential, is still
  // perfectly valid. Redirecting here sent the new tab to login.html, which saw
  // the good cookie and forwarded it to the role's home page - so clicking
  // "watch" on the upload page silently landed back on the upload page.
  //
  // The server is the authority. Let the check above decide, and redraw the
  // header once it answers so the account chip is not left blank.
  confirm.then(me => { if (me) remountHeader(); });
  return null;
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

/* Shared app header: brand and portal, navigation, appearance and account.

   ONE header for every authenticated page. `variant` ("student" | "instructor")
   decides which nav items render and defaults to the signed-in role, `active`
   names the current one, and `crumbs` adds a breadcrumb trail beside the nav.

   The page's own <title> is left alone. This used to overwrite it with
   "ACADIA" / "ACADIA Portal", which meant no page could ever carry a
   descriptive tab title of its own. */
// The options the page last mounted with, so the header can be redrawn once an
// async session check fills in an account the first paint did not have.
let _headerOpts = null;
let _headerEvents = null;

function remountHeader(){
  if (_headerOpts) mountHeader(_headerOpts);
}

function uiIcon(name, size = 20){
  const paths = {
    grid:'<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
    book:'<path d="M4 19.5V5a2 2 0 0 1 2-2h14v18H6a2 2 0 0 1 0-4h14M8 7h8M8 11h5"/>',
    chart:'<path d="M4 3v18h17M9 16v-5M14 16V7M19 16v-8"/>',
    lab:'<path d="M9 3h6M10 3v6L4.5 18.5A1.7 1.7 0 0 0 6 21h12a1.7 1.7 0 0 0 1.5-2.5L14 9V3M7 15h10"/>',
    switch:'<path d="M4 7h16m-4-4 4 4-4 4M20 17H4m4-4-4 4 4 4"/>',
    settings:'<path d="M4 7h16M4 17h16"/><circle cx="9" cy="7" r="3"/><circle cx="15" cy="17" r="3"/>',
    chevron:'<path d="m9 5 7 7-7 7"/>',
    menu:'<path d="M4 6h16M4 12h16M4 18h16"/>',
    close:'<path d="m6 6 12 12M6 18 18 6"/>',
    arrow:'<path d="M5 12h14m-6-6 6 6-6 6"/>',
    plus:'<path d="M12 5v14M5 12h14"/>'
  };
  return `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" stroke-width="1.7" stroke-linecap="round"
    stroke-linejoin="round" aria-hidden="true">${paths[name] || paths.grid}</svg>`;
}

function mountHeader({active = "", wide = false, variant = "", crumbs = null} = {}){
  _headerOpts = {active, wide, variant, crumbs};
  if (_headerEvents) _headerEvents.abort();
  _headerEvents = new AbortController();
  const {signal} = _headerEvents;
  document.querySelectorAll('[data-app-shell]').forEach(el => el.remove());
  const s = Session.get();
  const role = variant || (s && s.role === "teacher" ? "instructor" : "student");
  const roleHome = role === "instructor" ? "teacher.html" : "dashboard.html";
  const roleLabel = role === "instructor" ? "Instructor" : "Student";
  const navItem = (label, href, icon, key, id = "") => `<a
    class="hbtn ${active === key || (!active && key === "Home") ? "on" : ""}"
    href="${href}" data-nav="${key}" ${id ? `id="${id}"` : ""}
    title="${label}" aria-label="${label}"
    ${active === key || (!active && key === "Home") ? 'aria-current="page"' : ''}>
    ${uiIcon(icon)}<span class="htext">${label}</span></a>`;
  const nav = role === "instructor"
    ? navItem("Overview",roleHome,"grid","Home","homeBtn")
      + navItem("Assignments","teacher.html#list","book","Assignments")
      + navItem("Grades","grades.html","chart","Grades")
      + navItem("Playground","playground.html","lab","Playground")
    : navItem("Dashboard",roleHome,"grid","Dashboard","homeBtn")
      + navItem("My assignments","student.html","book","Assignments")
      + navItem("My grades","student-grades.html","chart","Grades");
  const switcher = s && s.role === "teacher"
    ? `<a class="viewas" href="${role === "instructor" ? "dashboard.html" : "teacher.html"}"
        title="View as ${role === "instructor" ? "student" : "instructor"}"
        aria-label="View as ${role === "instructor" ? "student" : "instructor"}">
        ${uiIcon("switch",18)}<span>View as ${role === "instructor" ? "student" : "instructor"}</span></a>` : "";
  const shell = document.createElement("div");
  shell.dataset.appShell = "";
  shell.innerHTML = `
    <button class="nav-scrim" id="navScrim" aria-label="Close navigation" hidden></button>
    <aside class="sidebar" id="appSidebar" aria-label="${roleLabel} navigation">
      <a class="wordmark" href="${roleHome}" aria-label="ACADIA home">
        <img class="brand-symbol" src="favicon.svg" width="40" height="40" alt="" aria-hidden="true"><span class="brand-name">ACADIA<span class="brand-caption">LEARNING STUDIO</span></span>
      </a>
      <div class="portal-label"><span class="portal-monogram">${roleLabel[0]}</span><span>${roleLabel} workspace</span></div>
      <div class="nav-label">WORKSPACE</div>
      <nav class="hnav" aria-label="Main">${nav}</nav>
      <div class="sidebar-bottom">
        ${switcher}
        <div class="account" id="acct">
          <button class="who" id="whoBtn" type="button" aria-label="Account menu"
            aria-haspopup="menu" aria-controls="whoMenu" aria-expanded="false">
            <span class="avatar">${esc(initials(s && s.name))}</span>
            <span class="account-copy"><span class="nm">${esc(s ? s.name : "Your account")}</span><span class="account-role">${roleLabel}</span></span>
            ${uiIcon("settings",17)}
          </button>
          <div class="menu" id="whoMenu" role="menu" hidden>
            <button class="mi" id="miSettings" role="menuitem">Appearance &amp; account</button>
            <button class="mi mi-danger" id="miLogout" role="menuitem">Sign out</button>
          </div>
        </div>
      </div>
    </aside>
    <header class="hdr">
      <button class="icon-button mobile-nav-toggle" id="navToggle" aria-label="Open navigation" aria-expanded="false" aria-controls="appSidebar">${uiIcon("menu")}</button>
      <div class="location-trail"><a class="portal-home" href="${roleHome}">${roleLabel} workspace</a><span class="crumbs" id="hcrumbs"></span></div>
      <div class="header-tools"><span class="workspace-label">${active === "Playground" ? "Pipeline tools" : role === "instructor" ? "Course management" : "Python practice"}</span>${themeToggle()}</div>
    </header>`;
  document.body.classList.add("has-shell");
  document.body.dataset.portal = role;
  document.body.prepend(shell);
  const main = document.querySelector("main");
  if (main && !document.querySelector(".skip-link")){
    if (!main.id) main.id = "mainContent";
    main.tabIndex = -1;
    const skip = document.createElement("a");
    skip.className = "skip-link"; skip.href = `#${main.id}`; skip.textContent = "Skip to content";
    document.body.prepend(skip);
  }
  const acct = shell.querySelector('#acct'), whoBtn = shell.querySelector('#whoBtn'), whoMenu = shell.querySelector('#whoMenu');
  const closeMenu = () => { whoMenu.hidden = true; whoBtn.setAttribute('aria-expanded','false'); };
  whoBtn.onclick = () => { const open = whoMenu.hidden; whoMenu.hidden = !open; whoBtn.setAttribute('aria-expanded',String(open)); };
  document.addEventListener('click',e => { if(!acct.contains(e.target))closeMenu(); },{signal});
  acct.addEventListener('focusout',() => queueMicrotask(() => {if(!acct.contains(document.activeElement))closeMenu();}),{signal});
  acct.addEventListener('keydown',e => {
    if(e.key !== 'ArrowDown' && e.key !== 'ArrowUp')return;
    e.preventDefault(); whoMenu.hidden=false; whoBtn.setAttribute('aria-expanded','true');
    const items=[...whoMenu.querySelectorAll('[role="menuitem"]')], current=items.indexOf(document.activeElement);
    items[e.key==='ArrowDown' ? (current+1)%items.length : (current<=0 ? items.length-1 : current-1)].focus();
  },{signal});
  shell.querySelector('#miSettings').onclick=() => {closeMenu(); openSettings(s);};
  shell.querySelector('#miLogout').onclick=async() => {await Session.signOut(); location.href='login.html';};
  const navToggle=shell.querySelector('#navToggle'), scrim=shell.querySelector('#navScrim'), sidebar=shell.querySelector('#appSidebar');
  const closeNav=() => {document.body.classList.remove('nav-open');scrim.hidden=true;navToggle.setAttribute('aria-expanded','false');};
  navToggle.onclick=() => {
    const open=!document.body.classList.contains('nav-open');
    document.body.classList.toggle('nav-open',open); scrim.hidden=!open; navToggle.setAttribute('aria-expanded',String(open));
    if(open) sidebar.querySelector('.hbtn').focus();
  };
  scrim.onclick=() => {closeNav();navToggle.focus();};
  sidebar.addEventListener('click',e => {if(e.target.closest('a'))closeNav();},{signal});
  addEventListener('keydown',e => {
    if(e.key==='Escape'){
      if(!whoMenu.hidden){closeMenu();whoBtn.focus();}
      if(document.body.classList.contains('nav-open')){closeNav();navToggle.focus();}
    }
    if(e.key==='Tab' && document.body.classList.contains('nav-open')){
      const controls=[...sidebar.querySelectorAll('a,button')].filter(el=>!el.closest('[hidden]'));
      const first=controls[0],last=controls[controls.length-1];
      if(e.shiftKey && document.activeElement===first){e.preventDefault();last.focus();}
      else if(!e.shiftKey && document.activeElement===last){e.preventDefault();first.focus();}
    }
  },{signal});
  const media=matchMedia('(max-width: 760px)');
  media.addEventListener('change',e => {if(!e.matches)closeNav();},{signal});
  const updateNav=() => {
    if(role !== 'instructor' || !location.pathname.endsWith('teacher.html'))return;
    const key=location.hash==='#list'?'Assignments':'Home';
    shell.querySelectorAll('[data-nav]').forEach(el=>{
      el.classList.toggle('on',el.dataset.nav===key);
      if(el.dataset.nav===key)el.setAttribute('aria-current','page');else el.removeAttribute('aria-current');
    });
  };
  addEventListener('hashchange',updateNav,{signal}); updateNav();
  syncThemeControls(Theme.get());
  if(crumbs)setCrumbs(crumbs);
  return shell.querySelector('.hdr');
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
          <button class="modalX" id="mtsX" aria-label="Close settings">&times;</button>
        </div>
        <div class="modalBody" id="mtsBody"></div>
      </div>`;
    document.body.appendChild(ov);
  }
  const returnFocus = document.getElementById("whoBtn") || document.activeElement;
  const previousOverflow = document.body.style.overflow;
  const close = () => {
    ov.hidden = true;
    document.body.style.overflow = previousOverflow;
    if (returnFocus && returnFocus.isConnected) returnFocus.focus();
  };
  ov.onclick = e => { if (e.target === ov) close(); };
  ov.querySelector("#mtsX").onclick = close;
  ov.onkeydown = e => {
    if (e.key === "Escape") { e.preventDefault(); close(); }
    if (e.key !== "Tab") return;
    const controls = [...ov.querySelectorAll('button:not(:disabled)')];
    const first = controls[0], last = controls[controls.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault(); last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault(); first.focus();
    }
  };
  ov.querySelector("#mtsBody").innerHTML = `
    <div class="setRow"><span>Account</span><b>${esc(s ? s.name : "guest")}</b></div>
    <div class="setRow"><span>Portal</span><b>${s && s.role === "teacher" ? "Instructor" : "Student"}</b></div>
    <div class="setRow"><span>Appearance</span>
      <span class="seg" id="mtsTheme" role="group" aria-label="Color theme">
        <button type="button" data-t="dark">Dark</button>
        <button type="button" data-t="light">Light</button>
      </span>
    </div>
    <p class="setNote">Your appearance preference is saved on this browser.
      Contact your instructor if your account role needs to change.</p>`;
  ov.querySelectorAll("[data-t]").forEach(b => {
    b.onclick = () => Theme.set(b.dataset.t);
  });
  syncThemeControls(Theme.get());
  ov.hidden = false;
  document.body.style.overflow = "hidden";
  ov.querySelector("#mtsX").focus();
}

/* Breadcrumb trail in the header, e.g. Home / Practice Set / Count Vowels.

   `items` is [{label, go}] - `go` is a function, because the student page is
   one document that swaps sections rather than three URLs, so there is nothing
   to href. The LAST item is the current page and is not clickable. Call with
   [] to clear. */
function setCrumbs(items){
  if (_headerOpts) _headerOpts.crumbs = items;
  const host = document.getElementById("hcrumbs");
  if (!host) return;
  host.innerHTML = "";
  const sepEl = () => {
    const s = document.createElement("span");
    s.className = "sep";
    s.setAttribute("aria-hidden", "true");
    s.textContent = "/";
    return s;
  };
  (items || []).forEach((it, i) => {
    // A leading separator, because the trail continues from the Home button
    // sitting immediately to its left. Repeating "Home" as the first crumb put
    // the word on screen twice, 8px apart.
    host.appendChild(sepEl());
    const last = i === items.length - 1;
    const b = document.createElement("button");
    b.type = "button";
    b.className = "crumb";
    b.textContent = it.label;
    b.title = it.label;
    if (last){
      b.setAttribute("aria-current", "page");
      b.disabled = true;
      // A disabled crumb is still the label of where you are, so it must not
      // read as a broken control: no not-allowed cursor, no dimming.
      b.style.cursor = "default";
      b.style.background = "none";
      b.style.color = "var(--text)";
    } else if (it.go){
      b.addEventListener("click", it.go);
    }
    host.appendChild(b);
  });
}

/* ============================================================
   Shared primitives
   ------------------------------------------------------------
   Every screen used to roll its own version of these four things, which is why
   no two of them behaved the same. One implementation each.
   ============================================================ */

/* --- toast -------------------------------------------------------------
   For things the user should notice but does not have to act on. Anything
   BLOCKING belongs in an inline banner next to the control that failed, not
   in a corner of the screen. */
function toast(message, kind = "info", ms = 5000){
  let host = document.getElementById("mtToasts");
  if (!host){
    host = document.createElement("div");
    host.id = "mtToasts";
    host.className = "toasts";
    // polite, not assertive: a toast never interrupts what is being read.
    host.setAttribute("role", "status");
    host.setAttribute("aria-live", "polite");
    document.body.appendChild(host);
  }
  const t = document.createElement("div");
  t.className = "toast " + (kind === "info" ? "" : kind);
  t.innerHTML = `<span class="tbar" aria-hidden="true"></span>
                 <span class="tmsg"></span>
                 <button class="x" type="button" aria-label="Dismiss">&times;</button>`;
  t.querySelector(".tmsg").textContent = message;
  const kill = () => {
    t.classList.add("out");
    setTimeout(() => t.remove(), 200);
  };
  t.querySelector(".x").onclick = kill;
  host.appendChild(t);
  if (ms) setTimeout(kill, ms);
  return t;
}

/* --- button loading / disabled states ----------------------------------
   `disable(btn, why)` is the only way a button in this app goes dead: it is
   impossible to call without saying why, and the reason lands in the tooltip
   the user sees when they hover the thing that will not click. */
function disable(btn, why){
  if (!btn) return;
  btn.disabled = true;
  btn.setAttribute("aria-disabled", "true");
  if (why) btn.title = why;
}
function enable(btn){
  if (!btn) return;
  btn.disabled = false;
  btn.removeAttribute("aria-disabled");
  btn.removeAttribute("title");
}
/* Busy is not the same as disabled: the control is unavailable because IT is
   working, so it says so and keeps its own label to come back to. */
function setBusy(btn, on, busyLabel){
  if (!btn) return;
  if (on){
    if (btn.dataset.label == null) btn.dataset.label = btn.innerHTML;
    btn.dataset.busy = "1";
    btn.disabled = true;
    btn.setAttribute("aria-busy", "true");
    btn.innerHTML = `<span class="spin" aria-hidden="true"></span>`
      + esc(busyLabel || "Working…");
  } else {
    btn.dataset.busy = "";
    btn.removeAttribute("aria-busy");
    if (btn.dataset.label != null){
      btn.innerHTML = btn.dataset.label;
      delete btn.dataset.label;
    }
    enable(btn);
  }
}

/* --- skeleton ----------------------------------------------------------
   Anything that fetches shows one of these. A blank panel and a broken panel
   look identical, which is why "loading..." was never good enough. */
function skeletonRows(n = 3, widths = [78, 62, 88, 54, 70]){
  let out = "";
  for (let i = 0; i < n; i++){
    out += `<div class="skelRow" aria-hidden="true">
      <span class="skel" style="width:${widths[i % widths.length]}%"></span>
      <span class="skel" style="width:34%;height:9px"></span></div>`;
  }
  return `<div role="status" aria-label="Loading">${out}</div>`;
}

/* Small, text-only renderer for assignment docstrings and tutor messages.
   Never interpret source HTML: even code containing tags stays literal text.
   Preserve fenced/indented Python and doctests; reflow only prose. */
function renderLearningText(container, value, examples){
  container.replaceChildren();
  let lines = String(value || "").replace(/\r\n?/g, "\n")
    .replace(/^\n+|\n+$/g, "").split("\n");
  const nonempty = lines.filter(line => line.trim());
  if (!nonempty.length) return;
  const indent = Math.min(...nonempty.map(line => (line.match(/^ */) || [""])[0].length));
  if (indent) lines = lines.map(line => line.slice(indent));
  const inline = (node, text) => {
    text.split(/(`[^`\n]+`)/g).forEach(part => {
      if (part.startsWith("`") && part.endsWith("`") && part.length > 2){
        const code = document.createElement("code");
        code.textContent = part.slice(1, -1);
        node.appendChild(code);
      } else node.appendChild(document.createTextNode(part));
    });
  };
  const codeBlock = (text, label) => {
    const pre = document.createElement("pre");
    pre.className = "code";
    pre.tabIndex = 0;
    pre.setAttribute("aria-label", label || "Code example");
    const code = document.createElement("code");
    code.textContent = text.replace(/\n+$/, "");
    pre.appendChild(code);
    container.appendChild(pre);
  };
  let paragraph = [];
  const flush = () => {
    if (!paragraph.length) return;
    const text = paragraph.join(" ").trim();
    const match = /^examples?\s*\d*\s*[:.]\s*([\s\S]+)$/i.exec(text);
    const example = examples && match && examples(match[1]);
    if (example) container.appendChild(example);
    else {
      const p = document.createElement("p");
      inline(p, text);
      container.appendChild(p);
    }
    paragraph = [];
  };
  const python = line => /^(?:(?:async\s+)?def\s+\w+\s*\(|class\s+\w+|(?:for|while|if|elif|else|try|except|with)\b.*:\s*$|return\s+\S|[A-Za-z_]\w*(?:\[[^\]]+\])?\s*=(?!=))/.test(line.trim());
  for (let i = 0; i < lines.length;){
    const line = lines[i];
    const fence = /^\s*(`{3,}|~{3,})[^`~]*$/.exec(line);
    if (fence){
      flush();
      const block = [];
      const close = new RegExp("^\\s*" + fence[1][0] + "{" + fence[1].length + ",}\\s*$");
      i++;
      while (i < lines.length && !close.test(lines[i])) block.push(lines[i++]);
      if (i < lines.length) i++;
      codeBlock(block.join("\n"));
      continue;
    }
    if (/^\s*>>>/.test(line) || python(line) || (/^( {4}|\t)\S/.test(line) && !paragraph.length)){
      flush();
      const block = [];
      const doctest = /^\s*>>>/.test(line);
      while (i < lines.length && lines[i].trim() && !/^\s*(`{3,}|~{3,})/.test(lines[i])) block.push(lines[i++]);
      codeBlock(block.join("\n"), doctest ? "Python example and output" : "Code example");
      continue;
    }
    const bullet = /^\s*(?:[-*]\s+|\d+[.)]\s+)(.+)/.exec(line);
    if (bullet){
      flush();
      const ordered = /^\s*\d/.test(line);
      const list = document.createElement(ordered ? "ol" : "ul");
      if (ordered) list.start = parseInt(line, 10);
      const pattern = ordered ? /^\s*\d+[.)]\s+(.+)/ : /^\s*[-*]\s+(.+)/;
      let item;
      while (i < lines.length && (item = pattern.exec(lines[i]))){
        const li = document.createElement("li");
        inline(li, item[1]); list.appendChild(li); i++;
      }
      container.appendChild(list);
      continue;
    }
    if (!line.trim()) flush();
    else paragraph.push(line.trim());
    i++;
  }
  flush();
}

/* --- formatting --------------------------------------------------------
   toLocaleString() renders "9/1/2026, 12:46:42 PM" - seconds nobody needs and
   a date order that means two different things depending on where the reader
   grew up. */
function fmtWhen(iso){
  const d = new Date(iso);
  if (isNaN(d)) return "-";
  const date = d.toLocaleDateString("en-US",
    {month: "short", day: "numeric", year: "numeric"});
  const time = d.toLocaleTimeString("en-US",
    {hour: "numeric", minute: "2-digit"});
  return `${date} · ${time}`;
}
function relTime(iso){
  const d = new Date(iso);
  if (isNaN(d)) return "";
  const secs = (Date.now() - d.getTime()) / 1000;
  if (secs < 60) return "just now";
  const units = [["minute", 60], ["hour", 3600], ["day", 86400],
                 ["month", 2592000], ["year", 31536000]];
  let label = "minute", size = 60;
  for (const [u, s] of units){ if (secs >= s){ label = u; size = s; } }
  const n = Math.floor(secs / size);
  return `${n} ${label}${n === 1 ? "" : "s"} ago`;
}
function fmtBytes(n){
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1048576).toFixed(1)} MB`;
}

/* --- tiny Python highlighter -------------------------------------------
   For the static snippets (the format example on the instructor page). Four
   token classes is enough to make a four-line sample readable, and it beats
   pulling CodeMirror into a page that has no editor.

   Order matters: comments and strings are matched FIRST and their content is
   never re-scanned, so a keyword inside a string stays a string. */
function hlPython(src){
  const out = [];
  const re = /(#[^\n]*)|('''[\s\S]*?'''|"""[\s\S]*?"""|'(?:[^'\\\n]|\\.)*'|"(?:[^"\\\n]|\\.)*")/g;
  let last = 0, m;
  const plain = s => esc(s)
    .replace(/\b(def|return|if|elif|else|for|while|in|not|and|or|import|from|None|True|False)\b/g,
             '<span class="t-kw">$1</span>')
    .replace(/\b([a-zA-Z_]\w*)(?=\()/g, '<span class="t-fn">$1</span>');
  while ((m = re.exec(src))){
    out.push(plain(src.slice(last, m.index)));
    out.push(`<span class="${m[1] ? "t-com" : "t-str"}">${esc(m[0])}</span>`);
    last = m.index + m[0].length;
  }
  out.push(plain(src.slice(last)));
  return out.join("");
}
