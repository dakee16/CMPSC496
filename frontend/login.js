/* Two modes on one form, because they differ by one field's meaning and one
   URL. A separate register page would duplicate the whole card to change a
   button label. */
let mode = "login";

const el = id => document.getElementById(id);
const serverErr = m => { el("err").textContent = m; };

/* One rule, stated once, in the same words the server uses. main/auth.py
   allows psu.edu only, so anything else is a round trip that can only fail. */
const PSU = /^[^\s@]+@([a-z0-9-]+\.)*psu\.edu$/i;

function fieldErr(id, msg){
  const input = el(id), out = el(id + "Err");
  out.textContent = msg || "";
  input.setAttribute("aria-invalid", msg ? "true" : "false");
  return !msg;
}

/* Says what is wrong AND how to fix it. "Invalid email" is neither. */
function checkEmail(){
  const v = el("u").value.trim();
  if (!v) return fieldErr("u", "Enter your Penn State email address.");
  return fieldErr("u", PSU.test(v)
    ? "" : "Use your Penn State address, like abc1234@psu.edu.");
}
function checkName(id, what){
  const v = el(id).value.trim();
  return fieldErr(id, v ? "" : `Enter your ${what} name.`);
}
function checkPassword(){
  const v = el("p").value;
  if (!v) return fieldErr("p", "Enter your password.");
  if (mode === "register" && v.length < 8)
    return fieldErr("p", "Use at least 8 characters.");
  return fieldErr("p", "");
}

/* Validate on the way OUT of a field, never on every keystroke: telling
   someone their address is wrong while they are still typing it is noise. */
el("u").addEventListener("blur", () => { if (el("u").value.trim()) checkEmail(); });
el("p").addEventListener("blur", () => { if (el("p").value) checkPassword(); });
// Clear a standing error as soon as they start fixing it.
["u","p","fn","ln"].forEach(id => el(id).addEventListener("input", () => {
  if (el(id + "Err").textContent) fieldErr(id, "");
  serverErr("");
}));

function paint(){
  const reg = mode === "register";
  document.title = reg ? "Create an account · ACADIA" : "Sign in · ACADIA";
  el("title").textContent = reg ? "Set up your account" : "Sign in";
  el("sub").textContent = reg
    ? "Use your Penn State email. Your work is saved against this account."
    : "Welcome back. Pick up where your thinking left off.";
  el("go").textContent = reg ? "Create account" : "Sign in";
  el("p").setAttribute("autocomplete", reg ? "new-password" : "current-password");
  el("pwHelp").hidden = !reg;
  el("nameRow").hidden = !reg;
  // Required only while they are shown, or sign-in would refuse to submit over
  // two fields nobody can see.
  ["fn","ln"].forEach(id => { el(id).required = reg; el(id).value = ""; });
  el("swapText").textContent = reg ? "Already have an account?" : "New here?";
  el("swapBtn").textContent = reg ? "Sign in" : "Create an account";
  enable(el("go"));
  serverErr("");
  fieldErr("u", ""); fieldErr("p", "");
  fieldErr("fn", ""); fieldErr("ln", "");
}

el("swapBtn").addEventListener("click", () => {
  mode = mode === "login" ? "register" : "login";
  paint();
  el(mode === "register" ? "fn" : "u").focus();
});

/* Already signed in? The cookie is the authority, so ask the server rather
   than trusting what sessionStorage remembers. */
Session.check().then(me => {
  if (me) location.replace(me.role === "teacher" ? "teacher.html" : "dashboard.html");
});

el("gate").addEventListener("submit", async e => {
  e.preventDefault();

  /* Client-side checks are a COURTESY - they save a round trip on an obvious
     typo. The server re-checks all of it; nothing here admits anyone. */
  const reg = mode === "register";
  const checks = [
    ...(reg ? [["fn", checkName("fn", "first")], ["ln", checkName("ln", "last")]] : []),
    ["u", checkEmail()], ["p", checkPassword()],
  ];
  const bad = checks.find(([, ok]) => !ok);
  if (bad){
    el(bad[0]).focus();
    return;
  }

  setBusy(el("go"), true, mode === "register" ? "Creating…" : "Signing in…");
  try {
    const me = await Session.signIn(mode, el("u").value.trim(), el("p").value,
      {first_name: el("fn").value.trim(), last_name: el("ln").value.trim()});
    location.href = me.role === "teacher" ? "teacher.html" : "dashboard.html";
  } catch (ex) {
    setBusy(el("go"), false);
    serverErr(ex.message);
    el("p").focus();
  }
});

paint();
