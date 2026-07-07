console.log("app.js loaded");

// ════════════════════════════════════════════════════════════
// CONSTANTS + STATE
// ════════════════════════════════════════════════════════════

const PROBLEM = {
  slug: "two-sum",
  title: "Two Sum",
  description: "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice."
};

const API_URL = "http://localhost:8000";
const MAX_ATTEMPTS = 3;


let chunks = [];
let currentChunkIndex = 0;
let header = "";
let acceptedAnswers = [];
let activeTab = "leetcode";
let attemptCount = 0;
let selectedProblem = null;
let message = "";
let lockLine = 0;
let activeMark = null;
let currentStudent = null;
let solvedSlugs = new Set();
let authMode = "login";


// ════════════════════════════════════════════════════════════
// DOM REFERENCES
// ════════════════════════════════════════════════════════════

const answerInput = document.getElementById("answer-input");
const submitBtn = document.getElementById("submit-btn");
const feedbackSection = document.getElementById("feedback-section");
const feedbackText = document.getElementById("feedback-text");
const startBtn = document.getElementById("start-btn");
const problemSection = document.getElementById("problem-section");
const partSection = document.getElementById("part-section");
const chunkNumber = document.getElementById("part-number");
const totalchunks = document.getElementById("total-parts");
const chunkPrompt = document.getElementById("part-prompt");
const tabButtons = document.querySelectorAll(".tab-btn");
const tabDiv = document.querySelectorAll(".tab-div");
const problemList = document.getElementById("problem-list");
const problemListContainer = document.getElementById("problem-list-container");
const uploadListContainer = document.getElementById("uploaded-problem-list-container");
const problemTitle = document.getElementById("problem-title");
const problemDifficulty = document.getElementById("problem-difficulty");
const problemDescription = document.getElementById("problem-description");
const backButtons = document.querySelectorAll(".back-btn");
const fileInput = document.getElementById("file-input");
const editCodeBtn = document.getElementById("edit-code-btn");
const uploadedProblemList = document.getElementById("uploaded-problem-list");
const nextBtn = document.getElementById("next-btn");
const uploadControls = document.getElementById("upload-controls");
const fileName = document.getElementById("file-name");
const uploadAnotherBtn = document.getElementById("upload-another-btn");
const solutionSection = document.getElementById("solution-section");
const finalSolution = document.getElementById("final-solution");
const authSection = document.getElementById("auth-section");
const mainApp = document.getElementById("main-app");
const authHeading = document.getElementById("auth-heading");
const authError = document.getElementById("auth-error");
const authUsername = document.getElementById("auth-username");
const authPassword = document.getElementById("auth-password");
const authSubmitBtn = document.getElementById("auth-submit-btn");
const authToggle = document.getElementById("auth-toggle");
const authTogglePrompt = document.getElementById("auth-toggle-prompt");
const logoutBtn = document.getElementById("logout-btn");
const storedSession = localStorage.getItem("microtutor_student");





// ════════════════════════════════════════════════════════════
// EVENT LISTENERS
// ════════════════════════════════════════════════════════════


//
// AUTH TOGGLE EVEN LISTENER 
//

authToggle.addEventListener("click", function (e) {
  e.preventDefault();
  authError.style.display = "none";

  if (authMode === "login") {
    authMode = "signup";
    authHeading.textContent = "Sign Up";
    authSubmitBtn.textContent = "Sign Up";
    authTogglePrompt.textContent = "Already have an account?";
    authToggle.textContent = "Log in";
  } else {
    authMode = "login";
    authHeading.textContent = "Log In";
    authSubmitBtn.textContent = "Log In";
    authTogglePrompt.textContent = "Don't have an account?";
    authToggle.textContent = "Sign up";
  }
});

//
// AUTH SUBMIT EVEN LISTENER 
//

authSubmitBtn.addEventListener("click", async function () {

  authError.style.display = "none";
  const userUsername = authUsername.value.trim();
  const userPassword = authPassword.value;

  if (userUsername.length === 0) {
    authError.textContent = "Username field cannot be empty";
    authError.style.display = "block";
    return;
  }
  if (userPassword.length === 0) {
    authError.textContent = "Password field cannot be empty";
    authError.style.display = "block";
    return;
  }

  const endpoint = authMode === "login" ? "/login" : "/register";

  try {
    const response = await fetch(`${API_URL}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username: userUsername,
        password: userPassword
      })
    });

    const data = await response.json();

    if (!response.ok) {
      authError.textContent = data.detail || "Something went wrong.";
      authError.style.display = "block";
      return;
    }


    currentStudent = { id: data.student_id, username: data.username };
    localStorage.setItem("microtutor_student", JSON.stringify(currentStudent));

    await loadSolvedSlugs();
    showMainApp();

  } catch (error) {
    console.error("Auth request failed:", error);
    authError.textContent = "Could not reach the server.";
    authError.style.display = "block";
  }
});

//
// LOGOUT BUTTON 
//


logoutBtn.addEventListener("click", function () {

  localStorage.removeItem("microtutor_student");
  currentStudent = null;
  solvedSlugs = new Set()
  showAuthPage()

});

//
// PRESS ENTER ON USERNAME FIELD
//
authUsername.addEventListener("keydown", function (e) {
  if (e.key === "Enter") {
    e.preventDefault();
    authPassword.focus();
  }
});

//
// PRESS ENTER ON PASSWORD FIELD
//

authPassword.addEventListener("keydown", function (e) {
  if (e.key === "Enter") {
    e.preventDefault();
    authSubmit.click();
  }
});

//
// START BUTTON EVEN LISTENER 
//

startBtn.addEventListener("click", async function () {

  console.log("Start clicked — calling backend...");
  startBtn.textContent = "Calling Backend...";
  startBtn.disabled = true;

  try {
    const response = await fetch(`${API_URL}/decompose_chunks`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        slug: selectedProblem.slug,
        description: selectedProblem.description
      })
    });

    const data = await response.json();
    console.log("Got response from backend:", data);

    header = data.header;
    chunks = data.chunks;
    acceptedAnswers = [];

    problemSection.style.display = "none";
    startBtn.style.display = "none";
    partSection.style.display = "block";
    document.getElementById("chunk-problem-list-btn").style.display = "block";
    codeEditor.refresh();
    chunkNumber.textContent = "1";
    totalchunks.textContent = chunks.length;
    chunkPrompt.textContent = chunks[0].prompt;
    seedEditorForProblem(header);



    console.log(`Loaded ${chunks.length} chunks. First chunk:`, chunks[0]);

  } catch (error) {
    console.error("Decompose error:", error);
    feedbackSection.style.display = "block";
    feedbackText.textContent = "⚠️ Could not reach the grader. Try again.";
    startBtn.disabled = false;
    startBtn.textContent = "Start Tutoring Session";
  }
});

//
// SUBMIT BUTTON EVEN LISTENER 
//

submitBtn.addEventListener("click", async function () {

  console.log("Submit clicked");
  attemptCount++;
  submitBtn.disabled = true;
  submitBtn.textContent = "Evaluating...";

  const studentCode = getEditableStudentCode();
  console.log("student_code being sent:", studentCode);
  try {
    const response = await fetch(`${API_URL}/grade_chunk`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        problem: {
          slug: selectedProblem.slug,
          title: selectedProblem.title,
          description: selectedProblem.description,
          solution: ""
        },
        chunks: chunks,
        index: currentChunkIndex,
        student_code: studentCode,
        accepted_prefix: acceptedAnswers
      })
    });
    const result = await response.json();

    if (currentStudent) {
      fetch(`${API_URL}/log_interaction`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          student_id: currentStudent.id,
          slug: selectedProblem.slug,
          chunk_index: currentChunkIndex,
          attempt_number: attemptCount,
          student_code: studentCode,
          verdict: result.correct,
          tier: result.tier,
          reason: result.reason
        })
      }).catch(function (error) {
        console.warn("Interaction logging failed:", error);
      });
    }

    feedbackSection.style.display = "block";

    if (result.correct == true) {
      message = "✅ Correct! " + result.reason;
      if (result.tier == "execution-adapted") {
        message = "✅ Correct — your approach works! " + result.reason;
      }
      feedbackText.textContent = message;
      acceptedAnswers.push(studentCode);
      nextBtn.disabled = false;
    }
    else {
      nextBtn.disabled = true;
      message = "❌ " + result.reason;
      if (result.failures && (result.failures.length > 0)) {
        message += "\n\nFailing cases:";
        for (const f of result.failures) {
          message += `\n  input ${JSON.stringify(f.input)} → expected ${JSON.stringify(f.expected)}, got ${JSON.stringify(f.got)}`;
        }
      }
      if (attemptCount >= MAX_ATTEMPTS && acceptedAnswers.length === currentChunkIndex) {
        const ref = chunks[currentChunkIndex].reference || "";
        message += "\n\n💡 Here's a reference answer for this part:\n" + ref;
        acceptedAnswers.push(ref);
        replaceEditableWithReference(ref);
        nextBtn.disabled = false;
      }
      feedbackText.textContent = message;
    }

    submitBtn.disabled = false;
    submitBtn.textContent = "Submit Answer";
  }
  catch (error) {
    feedbackText.textContent = "⚠️ Could not reach the grader. Try again.";
    submitBtn.disabled = false;
    submitBtn.textContent = "Submit Answer";
  }
});

//
// NEXT BUTTON EVEN LISTENER 
//

nextBtn.addEventListener("click", function () {

  console.log("Next clicked");

  currentChunkIndex++;

  if (currentChunkIndex >= chunks.length) {
    partSection.style.display = "none";
    feedbackSection.style.display = "none";
    const full = buildFinalSolution();
    finalSolution.textContent = full;
    solutionSection.style.display = "block";
    if (currentStudent && selectedProblem) {
      fetch(`${API_URL}/mark_solved`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          student_id: currentStudent.id,
          slug: selectedProblem.slug
        })
      })
        .then(function () {
          solvedSlugs.add(selectedProblem.slug);
          applySolvedHighlight();
        })
        .catch(function (error) {
          console.warn("Mark solved failed:", error);
        });
    }
  } else {
    chunkNumber.textContent = (currentChunkIndex + 1);
    chunkPrompt.textContent = chunks[currentChunkIndex].prompt;
    feedbackSection.style.display = "none";
    attemptCount = 0;
    lockCurrentChunkAndAdvance();
  }

});

//
// BACK BUTTON LISTENERS
//


backButtons.forEach(function (btn) {
  btn.addEventListener("click", function () {
    solutionSection.style.display = "none";
    problemSection.style.display = "none";
    partSection.style.display = "none";
    document.getElementById("chunk-problem-list-btn").style.display = "none";
    startBtn.style.display = "none";
    if (activeTab === "upload") {

      uploadListContainer.style.display = "block";

    } else {
      problemListContainer.style.display = "block";
    }
    feedbackSection.style.display = "none";

    chunks = [];
    currentChunkIndex = 0;
    header = "";
    acceptedAnswers = [];
    attemptCount = 0;
    selectedProblem = null;

    // Reset the editor + lock state
    if (activeMark) {
      activeMark.clear();
      activeMark = null;
    }
    lockLine = 0;
    codeEditor.setValue("");
    startBtn.textContent = "Start Tutoring Session"
    startBtn.disabled = false

  });
});

//
// FILE UPLOAD LISTENER
//

fileInput.addEventListener("change", function (event) {
  const file = event.target.files[0];
  if (!file) return;

  fileName.textContent = file.name;   // ← show chosen filename

  const reader = new FileReader();
  reader.onload = function () {
    const raw = reader.result;

    const problems = parseProblems(raw);
    console.log("Parsed problems:", problems);

    if (problems.length === 0) {
      alert("No problems found — check the file format (Problem N: / Description:).");
      return;
    }

    uploadedProblemList.innerHTML = "";

    problems.forEach(function (p) {
      const uploadProblemListElement = document.createElement("li");
      uploadProblemListElement.classList.add("problem-item");
      uploadProblemListElement.innerHTML = `<span class="problem-title">${p.title}</span>`;
      uploadProblemListElement.addEventListener("click", function () {
        showProblemDetail(p);
      });
      uploadedProblemList.appendChild(uploadProblemListElement);
    });

    uploadListContainer.style.display = "block";
    uploadControls.style.display = "none";
    uploadAnotherBtn.style.display = "block";
  };
  reader.readAsText(file);
});

// EDIT CODE BUTTON — placeholder.
editCodeBtn.addEventListener("click", function () {
  alert("Edit Code coming soon.");
});

//
// UPLOAD ANOTHER FILE LISTENER
//

uploadAnotherBtn.addEventListener("click", function () {
  uploadControls.style.display = "flex";
  uploadAnotherBtn.style.display = "none";
  uploadListContainer.style.display = "none";
  fileInput.value = "";
  fileName.textContent = "No file chosen";
});

// ════════════════════════════════════════════════════════════
// FUNCTIONS
// ════════════════════════════════════════════════════════════




async function loadSolvedSlugs() {
  if (!currentStudent) return;
  try {
    const response = await fetch(`${API_URL}/solved/${currentStudent.id}`);
    const data = await response.json();
    solvedSlugs = new Set(data.slugs);
    applySolvedHighlight()
  } catch (error) {
    console.log("error while loading solved problems for user")

  }
};

function applySolvedHighlight() {
  document.querySelectorAll("#problem-list .problem-item").forEach(function (currentProblem) {
    const slug = currentProblem.dataset.slug;
    if (solvedSlugs.has(slug)) {
      currentProblem.classList.add("solved");
    } else {
      currentProblem.classList.remove("solved");
    }
  });
}


function refreshLockedRegion() {
  if (activeMark) {
    activeMark.clear();
    activeMark = null;
  }
  if (lockLine <= 0) return;
  activeMark = codeEditor.markText(
    { line: 0, ch: 0 },
    { line: lockLine, ch: 0 },
    { readOnly: true, inclusiveLeft: true, inclusiveRight: false, className: "cm-locked" }
  );
}


function seedEditorForProblem(headerText) {
  codeEditor.setValue(headerText + "\n    ");
  lockLine = 1;
  refreshLockedRegion();
  codeEditor.setCursor({ line: 1, ch: 4 });
  codeEditor.focus();
}

function getEditableStudentCode() {
  const lines = codeEditor.getValue().split("\n");
  const editable = lines.slice(lockLine);
  const dedented = editable.map(function (line) {
    return line.startsWith("    ") ? line.slice(4) : line;
  });
  while (dedented.length > 0 && dedented[dedented.length - 1].trim() === "") {
    dedented.pop();
  }
  return dedented.join("\n");
}


function lockCurrentChunkAndAdvance() {
  const currentText = codeEditor.getValue();
  const trimmed = currentText.replace(/\s+$/, "");
  const newText = trimmed + "\n    ";
  codeEditor.setValue(newText);
  const lastLineIndex = codeEditor.lineCount() - 1;
  lockLine = lastLineIndex;
  refreshLockedRegion();
  codeEditor.setCursor({ line: lastLineIndex, ch: 4 });
  codeEditor.focus();
}

function replaceEditableWithReference(refText) {
  const lines = codeEditor.getValue().split("\n");
  const locked = lines.slice(0, lockLine);
  const indentedRef = refText.split("\n").map(function (l) { return "    " + l; });
  const newValue = locked.concat(indentedRef).join("\n");
  codeEditor.setValue(newValue);
  refreshLockedRegion();
}

function showAuthPage() {
  authSection.style.display = "flex";
  mainApp.style.display = "none";
  logoutBtn.style.display = "none";
}
function showMainApp() {
  authSection.style.display = "none";
  mainApp.style.display = "block";
  logoutBtn.style.display = "inline-block";

}
function buildFinalSolution() {

  const indentedAnswers = acceptedAnswers.map(function (answer) {
    return answer
      .split("\n")
      .map(function (line) { return "    " + line; })
      .join("\n");
  });
  return header + "\n" + indentedAnswers.join("\n");
}

// 
// FUNCTION FOR THE TAB SELECT BUTTONS 
//


tabButtons.forEach(function (clicked_button) {
  clicked_button.addEventListener("click", function () {

    const tabName = clicked_button.dataset.tab;
    tabButtons.forEach(function (btn) { btn.classList.remove("active") });
    clicked_button.classList.add("active");
    tabDiv.forEach(function (div) { div.style.display = "none" });
    const tabToDisplay = "tab-" + tabName;
    document.getElementById(tabToDisplay).style.display = "block";
    activeTab = tabName;

    // ── reset any open problem/chunk view when switching tabs ──
    problemSection.style.display = "none";
    partSection.style.display = "none";
    document.getElementById("chunk-problem-list-btn").style.display = "none";
    startBtn.style.display = "none";
    feedbackSection.style.display = "none";
    selectedProblem = null;

    // ── restore the leetcode list (it gets hidden when you open a problem) ──
    if (tabName === "leetcode") {
      problemListContainer.style.display = "block";
    } else {
      // upload tab: show the list if files were parsed, else show the controls
      if (uploadedProblemList.children.length > 0) {
        uploadListContainer.style.display = "block";
        uploadControls.style.display = "none";
        uploadAnotherBtn.style.display = "block";
      } else {
        uploadControls.style.display = "flex";
        uploadListContainer.style.display = "none";
      }
    }
  });
});

// 
// LOAD PROBLEMS ON PAGE LOAD
// 

async function loadProblems() {
  try {
    const response = await fetch(`${API_URL}/problems`);
    const data = await response.json();
    console.log("Loaded problems:", data);

    data.problems.forEach(function (problem) {
      const problemsListElement = document.createElement("li");
      problemsListElement.dataset.slug = problem.slug;
      if (solvedSlugs.has(problem.slug)) {
        problemsListElement.classList.add("solved");
      }

      problemsListElement.classList.add("problem-item");
      problemsListElement.innerHTML = `
      <span class="problem-title">${problem.title}</span>
      <span class="difficulty difficulty-${problem.difficulty.toLowerCase()}">${problem.difficulty}</span>`;
      problemList.appendChild(problemsListElement);
      //
      // LIST ELEMENT CLICK LISTENER
      //
      problemsListElement.addEventListener("click", async function () {

        const response = await fetch(`${API_URL}/problems/${problem.slug}`);
        const data = await response.json();
        selectedProblem = data;
        showProblemDetail(data);
      })

    });

  } catch (error) {
    console.error("Failed to load problems:", error);
  }
}

// 
// HELPER FUNCTION TO FORMAT THE PROBLEM DESCRIPTION
//

function formatDescription(text) {
  const constraintsIndex = text.indexOf("Constraints:");
  if (constraintsIndex !== -1) {
    text = text.substring(0, constraintsIndex);
  }

  return `<div class="desc-header">Description:</div>` + text
    .replace(/\s*Example \d+:\s*/g, function (match) {
      return `<div class="desc-header">${match.trim()}</div>`;
    })
    .replace(/\s*Input:\s*/g, `<em class="io-label">Input: </em>`)
    .replace(/\s*Output:\s*/g, `<em class="io-label">Output: </em>`)
    .replace(/\s*Explanation:\s*/g, `<em class="io-label">Explanation: </em>`);
}

//
// HELPER FUNCTION FOR SLUGIFY 
//

function slugify(title) {
  return title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
}

//
// HELPER FUNCTION FOR PARSING PROBLEMS
//

function parseProblems(rawText) {
  return rawText
    .split(/^Problem\s+\d+\s*:/m)
    .slice(1)
    .map(chunk => {
      const idx = chunk.search(/Description\s*:/i);
      if (idx === -1) return null;
      let title = chunk.slice(0, idx);
      const diffMatch = chunk.match(/Difficulty\s*:\s*(Easy|Medium|Hard)/i);
      let difficulty = null;
      if (diffMatch) {
        difficulty = diffMatch[1];
        difficulty = difficulty[0].toUpperCase() + difficulty.slice(1).toLowerCase();
        title = title.replace(/Difficulty\s*:\s*(Easy|Medium|Hard)/i, "");
      }

      title = title.trim().replace(/_/g, " ");

      const description = chunk.slice(idx)
        .replace(/Description\s*:/i, "")
        .trim()
        .replace(/\s+/g, " ");

      const problem = { slug: slugify(title), title, description };
      if (difficulty) problem.difficulty = difficulty;
      return problem;
    })
    .filter(p => p && p.title && p.description);
}


//
// HELPER FOR SHOWING PROBLEM DETAILS
//

function showProblemDetail(problem) {

  selectedProblem = problem;
  problemTitle.textContent = problem.title;
  problemDescription.innerHTML = formatDescription(problem.description);

  if (problem.difficulty) {
    problemDifficulty.style.display = "";
    problemDifficulty.textContent = problem.difficulty;
    problemDifficulty.className = `difficulty difficulty-${problem.difficulty.toLowerCase()}`;
  }
  else {
    problemDifficulty.style.display = "none";

  }
  problemListContainer.style.display = "none";
  uploadListContainer.style.display = "none"
  problemSection.style.display = "block";
  startBtn.style.display = "block";


}


// ════════════════════════════════════════════════════════════
// INITIALIZATION
// ════════════════════════════════════════════════════════════

//
// INITIALIZE CODEMIRROR
//

const codeEditor = CodeMirror.fromTextArea(answerInput, {
  mode: "python",
  theme: "dracula",
  lineNumbers: true,
  indentUnit: 4,
  tabSize: 4,
  indentWithTabs: false,
  lineWrapping: true,
});



if (storedSession) {
  currentStudent = JSON.parse(storedSession);

  loadProblems().then(loadSolvedSlugs);

  showMainApp();
} else {
  loadProblems();

  showAuthPage();
}