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
  const links = s && s.role === "teacher"
    ? [["teacher.html", "Upload"], ["teacher.html#list", "Assignments"]]
    : [["student.html", "Practice"]];

  const hdr = document.createElement("header");
  hdr.className = "hdr" + (wide ? " wide" : "");
  hdr.innerHTML = `
    <a class="brand" href="home.html"><span class="dot"></span>MicroTutor</a>
    <span class="spacer"></span>
    <nav class="navpills" id="np">
      <span class="glider" id="glider"></span>
      ${links.map(([h, t]) =>
        `<a href="${h}" class="${t === active ? "on" : ""}">${t}</a>`).join("")}
    </nav>
    <span class="who">
      <span class="avatar">${esc(initials(s && s.name))}</span>
      <span class="nm">${esc(s ? s.name : "guest")}</span>
    </span>
    <button class="ghost" id="signout" style="padding:7px 13px;font-size:13px">Sign out</button>`;
  document.body.prepend(hdr);

  hdr.querySelector("#signout").onclick = () => {
    Session.clear();
    location.href = "home.html";
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
