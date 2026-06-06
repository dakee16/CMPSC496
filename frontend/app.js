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


let steps = [];
let currentStepIndex = 0;
let stepCode = [];
let activeTab = "leetcode";

let attemptCount = 0;
let selectedProblem = null;
let lockedCode = "";
let lockedLineCount = 0;
const MAX_ATTEMPTS = 3;

// ════════════════════════════════════════════════════════════
// DOM REFERENCES
// ════════════════════════════════════════════════════════════

const answerInput = document.getElementById("answer-input");
const submitBtn = document.getElementById("submit-btn");
const feedbackSection = document.getElementById("feedback-section");
const feedbackText = document.getElementById("feedback-text");
const startBtn = document.getElementById("start-btn");
const problemSection = document.getElementById("problem-section");
const stepSection = document.getElementById("step-section");
const stepNumber = document.getElementById("step-number");
const totalSteps = document.getElementById("total-steps");
const stepPrompt = document.getElementById("step-prompt");
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
const prevBtn = document.getElementById("prev-btn");
const uploadedProblemList = document.getElementById("uploaded-problem-list");
const nextBtn = document.getElementById("next-btn");
const uploadControls = document.getElementById("upload-controls");
const fileName = document.getElementById("file-name");
const uploadAnotherBtn = document.getElementById("upload-another-btn");


// ════════════════════════════════════════════════════════════
// EVENT LISTENERS
// ════════════════════════════════════════════════════════════

//
// START BUTTON EVEN LISTENER 
//

startBtn.addEventListener("click", async function () {
  console.log("Start clicked — calling backend...");
  startBtn.textContent = "Calling Backend...";
  startBtn.disabled = true;

  try {
    const response = await fetch(`${API_URL}/decompose`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        slug: selectedProblem.slug,
        title: selectedProblem.title,
        description: selectedProblem.description
      })
    });

    const data = await response.json();
    console.log("Got response from backend:", data);

    steps = data.steps;
    problemSection.style.display = "none";
    startBtn.style.display = "none";
    stepSection.style.display = "block";
    document.getElementById("step-problem-list-btn").style.display = "block";
    codeEditor.refresh();
    stepNumber.textContent = "1";
    totalSteps.textContent = steps.length;
    stepPrompt.textContent = steps[0].prompt;



    console.log(`Loaded ${steps.length} steps. First step:`, steps[0]);

  } catch (error) {
    console.error("Something went wrong while decomposing:", error);
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
  const currentStep = steps[currentStepIndex];
  const userAnswer = codeEditor.getRange(
    { line: lockedLineCount, ch: 0 },
    { line: codeEditor.lineCount(), ch: 0 }
  ).replace(/\s+$/, "");


  try {
    const response = await fetch(`${API_URL}/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        step: currentStep,
        answer: userAnswer,
        context: lockedCode
      })
    });
    const llmFeedback = await response.json();

    feedbackSection.style.display = "block";

    console.log("userAnswer repr:", JSON.stringify(userAnswer));
    console.log("lockedCode repr:", JSON.stringify(lockedCode));


    if (llmFeedback.correct) {
      feedbackText.textContent = "✅ Correct! " + llmFeedback.short_reason;
      stepCode[currentStepIndex] = userAnswer;

      lockedCode = lockedCode + userAnswer + "\n";
      codeEditor.setValue(lockedCode);

      lockedLineCount = codeEditor.lineCount() - 1;

      codeEditor.markText(
        { line: 0, ch: 0 },
        { line: lockedLineCount, ch: 0 },
        { readOnly: true }
      );
      codeEditor.setCursor(lockedLineCount, 0);
      nextBtn.disabled = false;

    } else {

      nextBtn.disabled = true;
      if (attemptCount < MAX_ATTEMPTS) {
        feedbackText.textContent = "❌ Incorrect. " + llmFeedback.short_reason
      }
      else {
        feedbackText.textContent = "❌ Incorrect. " + llmFeedback.short_reason + " | Correct answer: " + llmFeedback.correct_answer;
      }
    }

    submitBtn.disabled = false;
    submitBtn.textContent = "Submit Answer";

  }

  catch (error) {
    console.error("Something went wrong while evaluating:", error);
    submitBtn.disabled = false;
    submitBtn.textContent = "Submit Answer";
  }
});

//
// NEXT BUTTON EVEN LISTENER 
//

nextBtn.addEventListener("click", function () {

  console.log("Next clicked");

  currentStepIndex++;

  if (currentStepIndex >= steps.length) {
    alert("🎉 You finished all steps!");
  }
  else {
    stepNumber.textContent = (currentStepIndex + 1);
    stepPrompt.textContent = steps[currentStepIndex].prompt;
    feedbackSection.style.display = "none";
    attemptCount = 0;
  }

});

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

    // ── reset any open problem/step view when switching tabs ──
    problemSection.style.display = "none";
    stepSection.style.display = "none";
    document.getElementById("step-problem-list-btn").style.display = "none";
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
// BACK BUTTON LISTENERS
//

backButtons.forEach(function (btn) {
  btn.addEventListener("click", function () {

    problemSection.style.display = "none";
    stepSection.style.display = "none";
    document.getElementById("step-problem-list-btn").style.display = "none";
    startBtn.style.display = "none";
    if (activeTab === "upload") {

      uploadListContainer.style.display = "block";

    } else {
      problemListContainer.style.display = "block";
    }
    feedbackSection.style.display = "none";
    codeEditor.setValue("");
    lockedCode = "";
    lockedLineCount = 0;

    steps = [];
    currentStepIndex = 0;
    attemptCount = 0;
    selectedProblem = null;
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

//
// PREVIOUS BUTTON LISTENER
//

prevBtn.addEventListener("click", function () {

  if (currentStepIndex === 0) return;
  currentStepIndex--;

  stepCode[currentStepIndex] = undefined;

  lockedCode = "";
  for (let i = 0; i < currentStepIndex; i++) {
    lockedCode = lockedCode + stepCode[i] + "\n";
  }

  codeEditor.setValue(lockedCode);
  lockedLineCount = lockedCode === "" ? 0 : codeEditor.lineCount() - 1;

  if (lockedLineCount > 0) {
    codeEditor.markText(
      { line: 0, ch: 0 },
      { line: lockedLineCount, ch: 0 },
      { readOnly: true }
    );
  }
  codeEditor.setCursor(lockedLineCount, 0);
  stepNumber.textContent = (currentStepIndex + 1);
  stepPrompt.textContent = steps[currentStepIndex].prompt;
  feedbackSection.style.display = "none";
  attemptCount = 0;
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

loadProblems();