const diaryData = {
  1: {
    text: "Introduced to the basics of Python programming. Understood Python syntax, variables, data types, conditional statements, and loop structures. Practiced writing simple programs to build a strong programming foundation.",
    pdf: "pdfs/Day1.pdf"
  },
  2: {
    text: "Implemented a Grocery List program using Python. Learned to use loops and conditions to handle user input and manage list data. Introduced to Google Colab and explored basic Pandas operations like creating DataFrames and inspecting data.",
    pdf: "pdfs/Day2.pdf"
  },
  3: {
    text: "Gained insights into Artificial Intelligence and Machine Learning. Explored the Kaggle platform for datasets and competitions. Practiced loading datasets using Pandas, and applied functions like `.head()`, `.info()`, `.describe()`, `.iloc[]`, and `.loc[]` for data exploration.",
    pdf: "pdfs/Day3.pdf"
  },
  4: {
    text: "Started learning Machine Learning basics. Explored the California Housing dataset using Pandas in Google Colab. Applied data manipulation techniques including sorting, filtering, and summary statistics to understand the dataset structure and trends.",
    pdf: "pdfs/Day4.pdf"
  }
  
};

function showAllDays() {
  const daysContainer = document.getElementById('days-container');
  daysContainer.innerHTML = '';
  for (let day = 1; day <= 30; day++) {
    const btn = document.createElement('button');
    btn.innerText = `Day ${day}`;
    btn.onclick = () => showEntry(day); 
    daysContainer.appendChild(btn);
  }
}

function showEntry(day) {
  const entry = diaryData[day];
  document.getElementById('entry-title').innerText = `Day ${day}`;
  if (!entry) {
    document.getElementById('entry-content').innerHTML = `
      <div class="summary-text">No summary available for Day ${day}.</div>
    `;
    return;
  }
  let contentHtml = `<div class="summary-text">${entry.text || 'No summary available.'}</div>`;

  if (entry.pdf) {
    contentHtml += `
      <div class="pdf-wrapper">
        <a href="${entry.pdf}" target="_blank" class="pdf-link">
        <span class="pdf-icon">📄</span>View PDF</a>
      </div>`;
  }

  document.getElementById('entry-content').innerHTML = contentHtml;
}

document.addEventListener('DOMContentLoaded', showAllDays);
