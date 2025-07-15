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
  },
  5: {
  text: "Implemented Logistic Regression using scikit-learn on the Iris dataset. \
Learned the concept of classification in machine learning and explored its real-life applications. \
The Titanic dataset from Kaggle was also reviewed as an example of binary classification. \
Evaluated model accuracy and predicted class labels using the trained model.",
  pdf: "pdfs/Day5.pdf"
},
6: {
  text: "Extended Logistic Regression using Iris dataset loaded with pandas. Trained, tested, and saved the model using joblib. \
Evaluated performance with classification report and accuracy score. Introduced K-Nearest Neighbors (KNN) as a distance-based classification algorithm. \
Created a function to predict flower species based on user input.",
  pdf: "pdfs/Day6.pdf"
},
7: {
  text: "Implemented Logistic Regression on the Iris dataset using pandas and scikit-learn. \
Evaluated the model with accuracy and classification report. \
Also learned how to save models using joblib. \
Introduced K-Nearest Neighbors (KNN) as a distance-based classification algorithm.",
  pdf: "pdfs/Day7.pdf"
},
8: {
  text: "Explored Decision Tree and Random Forest classifiers. Visualized and analyzed the Iris dataset. \
Trained models using scikit-learn and evaluated them with accuracy scores and confusion matrices. \
Learned how Random Forest improves over a single Decision Tree by using ensemble techniques.",
  pdf: "pdfs/Day8.pdf"
},
9: {
  text: "Implemented Random Forest for Customer Churn Prediction using real-world Telco dataset. \
Learned Unsupervised Learning concepts like clustering and dimensionality reduction. \
Also discussed Heart Disease Prediction as a healthcare ML application.",
  pdf: "pdfs/Day9.pdf"
},
10: {
  text: "Explored Dimensionality Reduction using PCA on a vehicle dataset. \
Performed outlier removal, correlation analysis, and reduced features from 19 to 2 for visualization. \
Visualized class separation using principal components.",
  pdf: "pdfs/Day10.pdf"
},
11: {
  text: "Implemented K-Means Clustering using an unlabeled social media dataset. \
Performed preprocessing, label encoding, and scaling. \
Used Elbow Method to find optimal clusters and visualized cluster behavior.",
  pdf: "pdfs/Day11.pdf"
},
12: {
  text: "Learned and implemented Hierarchical Clustering on the Mall Customer dataset. \
Created a dendrogram using Ward linkage to identify optimal clusters. \
Used Agglomerative Clustering and visualized customer segments based on income and spending.",
  pdf: "pdfs/Day12.pdf"
},
13: {
  text: "Applied DBSCAN Clustering on Mall Customer dataset using age, income, and spending score. \
Used KNN to determine optimal epsilon. Detected dense regions and outliers without needing predefined clusters.",
  pdf: "pdfs/Day13.pdf"
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
  
  let contentHtml = `<div class="summary-text">${entry?.text || 'No summary available.'}</div>`;
  document.getElementById('entry-content').innerHTML = contentHtml;

  const viewer = document.getElementById('pdf-container');
  if (entry?.pdf) {
    viewer.innerHTML = `
      <div class="pdf-viewer fade-in">
        <button onclick="closePDF()" class="close-pdf-btn">✖ Close PDF</button>
        <iframe src="${entry.pdf}" width="100%" height="600px" style="border:1px solid #ccc; border-radius:10px;"></iframe>
      </div>
    `;
    viewer.scrollIntoView({ behavior: 'smooth' });
  } else {
    viewer.innerHTML = ''; 
  }
}


function openPDF(pdfPath) {
  const viewer = document.getElementById('pdf-container');
  viewer.innerHTML = `
    <div class="pdf-viewer fade-in">
      <button onclick="closePDF()" class="close-pdf-btn">✖ Close PDF</button>
      <iframe src="${pdfPath}" width="100%" height="900px" style="border:1px solid #ccc; border-radius:10px;"></iframe>
    </div>
  `;
  viewer.scrollIntoView({ behavior: 'smooth' });

  
}
function closePDF() {
  document.getElementById('pdf-container').innerHTML = '';
}
function openCertificate() {
  const viewer = document.getElementById('pdf-container');
  viewer.innerHTML = `
    <div class="pdf-viewer fade-in">
      <button onclick="closePDF()" class="close-pdf-btn">✖ Close PDF</button>
      <iframe src="pdfs/certificate.pdf" width="100%" height="600px"></iframe>
    </div>
  `;
  viewer.scrollIntoView({ behavior: 'smooth' });
}

function openProject() {
  const viewer = document.getElementById('pdf-container');
  viewer.innerHTML = `
    <div class="pdf-viewer fade-in">
      <button onclick="closePDF()" class="close-pdf-btn">✖ Close PDF</button>
      <iframe src="pdfs/project-report.pdf" width="100%" height="600px"></iframe>
    </div>
  `;
  viewer.scrollIntoView({ behavior: 'smooth' });
}



document.addEventListener('DOMContentLoaded', showAllDays);
