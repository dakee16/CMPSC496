console.log("app.js loaded");

const PROBLEM = {
  slug: "two-sum",
  title: "Two Sum",
  description: "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice."
};

const API_URL = "http://localhost:8000";


let steps = [];          
let currentStepIndex = 0; 

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
const answersPanel = document.getElementById("answers-panel");
const answersList  = document.getElementById("answers-list");
const tabButtons = document.querySelectorAll(".tab-btn");
const tabDiv = document.querySelectorAll(".tab-div");
const problemList = document.getElementById("problem-list");
const problemListContainer = document.getElementById("problem-list-container");
const problemTitle = document.getElementById("problem-title");
const problemDifficulty = document.getElementById("problem-difficulty");
const problemDescription = document.getElementById("problem-description");
const backButtons = document.querySelectorAll(".back-btn");
let attemptCount = 0;
let selectedProblem = null;
const MAX_ATTEMPTS = 3;

//
// START BUTTON EVEN LISTENER 
//

startBtn.addEventListener("click", async function() {
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
    codeEditor.refresh();
    stepNumber.textContent = "1";
    totalSteps.textContent = steps.length;
    stepPrompt.textContent = steps[0].prompt;
    answersPanel.style.display = "block"


    console.log(`Loaded ${steps.length} steps. First step:`, steps[0]);

  } catch (error) {
    console.error("Something went wrong while decomposing:", error);
  }
});

//
// SUBMIT BUTTON EVEN LISTENER 
//

submitBtn.addEventListener("click", async function() {

  console.log("Submit clicked");
  attemptCount++;
  submitBtn.disabled = true;
  submitBtn.textContent = "Evaluating...";
  const currentStep = steps[currentStepIndex]; 
  const userAnswer = codeEditor.getValue();

  
  try {
    const response = await fetch(`${API_URL}/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        step: currentStep,
        answer: userAnswer,
        context: ""
      })
    });
  const llmFeedback = await response.json();

  feedbackSection.style.display = "block";

  if (llmFeedback.correct) {
  feedbackText.textContent = "✅ Correct! " + llmFeedback.short_reason;
  addAnswerToPanel(currentStepIndex + 1, userAnswer);
  nextBtn.disabled = false;
} else {

    nextBtn.disabled = true;
    if (attemptCount<MAX_ATTEMPTS){
        feedbackText.textContent = "❌ Incorrect. " + llmFeedback.short_reason 
    }
    else{
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

const nextBtn = document.getElementById("next-btn");

//
// NEXT BUTTON EVEN LISTENER 
//

nextBtn.addEventListener("click", function() {

  console.log("Next clicked");
  
  currentStepIndex++;

  if (currentStepIndex >= steps.length) {
   alert( "🎉 You finished all steps!");
  }
  else { 
    stepNumber.textContent = (currentStepIndex + 1);
    stepPrompt.textContent = steps[currentStepIndex].prompt;
    codeEditor.setValue("");
    feedbackSection.style.display = "none";
    attemptCount = 0;
  }
  
});

//
// HELPER FUNCTION FOR ADDING ANSWERS 
//

function addAnswerToPanel(stepNum, answer) {

  const answerListElement = document.createElement("li");
  answerListElement.textContent = `Step ${stepNum}: ${answer}`;
  answersList.appendChild(answerListElement);
}


// 
// FUNCTION FOR THE TAB SELECT BUTTONS 
//


tabButtons.forEach(function(clicked_button) {
  clicked_button.addEventListener("click", function() {
    
    const tabName = clicked_button.dataset.tab;
    tabButtons.forEach(function(btn){btn.classList.remove("active")});
    clicked_button.classList.add("active");
    tabDiv.forEach(function(div){div.style.display = "none"});
    const tabToDisplay = "tab-" + tabName;
    document.getElementById(tabToDisplay).style.display = "block";
    
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

    data.problems.forEach(function(problem){
      const problemsListElement = document.createElement("li");
      problemsListElement.classList.add("problem-item");
      problemsListElement.innerHTML = `
      <span class="problem-title">${problem.title}</span>
      <span class="difficulty difficulty-${problem.difficulty.toLowerCase()}">${problem.difficulty}</span>`;
      problemList.appendChild(problemsListElement);
      //
      // LIST ELEMENT CLICK LISTENER
      //
      problemsListElement.addEventListener("click", async function(){

        const response = await fetch(`${API_URL}/problems/${problem.slug}`);
        const data = await response.json();
        selectedProblem = data;
        problemListContainer.style.display = "none";        
        problemTitle.textContent = data.title;
        problemDifficulty.textContent = data.difficulty;
        problemDifficulty.className = `difficulty difficulty-${data.difficulty.toLowerCase()}`;
        problemDescription.innerHTML = formatDescription(data.description);
        problemSection.style.display = 'block';
        startBtn.style.display = 'block';

      })
        
    });
   
  } catch (error) {
    console.error("Failed to load problems:", error);
  }
}

loadProblems();

// 
// HELPER FUNCTION TO FORMAT THE PROBLEM DESCRIPTION
//
function formatDescription(text) {
  const constraintsIndex = text.indexOf("Constraints:");
  if (constraintsIndex !== -1) {
    text = text.substring(0, constraintsIndex);
  }
  
  return `<div class="desc-header">Description:</div>` + text
    .replace(/\s*Example \d+:\s*/g, function(match) {
      return `<div class="desc-header">${match.trim()}</div>`;
    })
    .replace(/\s*Input:\s*/g, `<em class="io-label">Input: </em>`)
    .replace(/\s*Output:\s*/g, `<em class="io-label">Output: </em>`)
    .replace(/\s*Explanation:\s*/g, `<em class="io-label">Explanation: </em>`);
}

//
// BACK BUTTON LISTENERS
//

backButtons.forEach(function(btn) {
  btn.addEventListener("click", function() {
    
    problemSection.style.display = "none";
    stepSection.style.display = "none";
    answersPanel.style.display = "none";
    startBtn.style.display = "none";
    problemListContainer.style.display = "block";
    feedbackSection.style.display = "none";
    codeEditor.setValue("");
    
    steps= [];
    currentStepIndex = 0;
    attemptCount = 0;
    selectedProblem = null;
    answersList.innerHTML = ""
    startBtn.textContent = "Start Tutoring Session"
    startBtn.disabled = false
        
  });
});

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