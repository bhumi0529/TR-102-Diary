const diaryData = {
  1: {
    text: `
    <strong>🐍 Day 1 - Python Basics</strong><br><br>
    🔤 <strong>Concepts Covered:</strong><br>
    - Python syntax, variables, data types<br>
    - Conditional statements and loops<br>
    - Wrote simple programs for practice<br><br>
    🚀 <strong>Highlight:</strong><br>
    Built a basic calculator and number game for hands-on understanding.
    `,
    pdf: "pdfs/Day1.pdf"
  },
  2: {
    text: `
    <strong>🛒 Day 2 - Grocery List & Pandas Start</strong><br><br>
    📦 <strong>Tasks Completed:</strong><br>
    - Grocery list program using loops and conditionals<br>
    - Explored Google Colab for coding in the cloud<br>
    - Introduction to Pandas: created and viewed DataFrames<br><br>
    💻 <strong>Experience:</strong><br>
    Enjoyed coding interactively in Colab and debugging in real-time.
    `,
    pdf: "pdfs/Day2.pdf"
  },
  3: {
    text: `
    <strong>🤖 Day 3 - AI/ML Introduction & Dataset Handling</strong><br><br>
    🔍 <strong>What I Did:</strong><br>
    - Introduction to Artificial Intelligence & Machine Learning<br>
    - Explored Kaggle platform<br>
    - Practiced with Pandas functions: <code>.head()</code>, <code>.info()</code>, <code>.describe()</code>, <code>.iloc[]</code>, <code>.loc[]</code><br><br>
    🌐 <strong>Highlight:</strong><br>
    Discovered hundreds of real-world datasets on Kaggle!
    `,
    pdf: "pdfs/Day3.pdf"
  },
  4: {
    text: `
    <strong>🏠 Day 4 - California Housing Dataset</strong><br><br>
    🧮 <strong>Learnings:</strong><br>
    - Explored housing dataset using Pandas<br>
    - Applied sorting, filtering, and summarization<br><br>
    📌 <strong>Takeaway:</strong><br>
    Understood how housing data can be used to train price prediction models.
    `,
    pdf: "pdfs/Day4.pdf"
  },
  5: {
    text: `
    <strong>📊 Day 5 - Logistic Regression on Iris</strong><br><br>
    🌼 <strong>Topics Covered:</strong><br>
    - Logistic Regression using scikit-learn<br>
    - Accuracy calculation and class prediction<br>
    - Reviewed Titanic dataset as a binary classification example<br><br>
    🎯 <strong>Insight:</strong><br>
    Learned how supervised learning is applied in real life.
    `,
    pdf: "pdfs/Day5.pdf"
  },
  6: {
    text: `
    <strong>🌸 Day 6 - Logistic Regression + KNN</strong><br><br>
    🧠 <strong>What I Learned:</strong><br>
    - Trained Logistic Regression & saved with joblib<br>
    - Introduction to KNN algorithm<br>
    - Built prediction function using user input<br><br>
    🔥 <strong>Highlight:</strong><br>
    Real-time input and output felt like magic!
    `,
    pdf: "pdfs/Day6.pdf"
  },
  7: {
    text: `
    <strong>📈 Day 7 - KNN Deep Dive</strong><br><br>
    ⚙️ <strong>Techniques Applied:</strong><br>
    - Implemented KNN on Iris dataset<br>
    - Used train_test_split, accuracy_score<br>
    - Understood distance-based classification<br><br>
    🎉 <strong>Win:</strong><br>
    Got 97% accuracy on my KNN model!
    `,
    pdf: "pdfs/Day7.pdf"
  },
  8: {
    text: `
    <strong>🌳 Day 8 - Decision Tree & Random Forest</strong><br><br>
    🧠 <strong>Learned:</strong><br>
    - Decision Tree implementation with scikit-learn<br>
    - Used Random Forest as an ensemble technique<br>
    - Visualized feature importance<br><br>
    🌟 <strong>Wow Moment:</strong><br>
    Saw how combining trees gives more accuracy.
    `,
    pdf: "pdfs/Day8.pdf"
  },
  9: {
    text: `
    <strong>📉 Day 9 - Random Forest & Unsupervised Intro</strong><br><br>
    🌐 <strong>Explorations:</strong><br>
    - Customer Churn prediction<br>
    - Heart Disease ML use case<br>
    - Intro to Clustering & Dimensionality Reduction
    `,
    pdf: "pdfs/Day9.pdf"
  },
  10: {
    text: `
    <strong>🔍 Day 10 - PCA on Vehicle Dataset</strong><br><br>
    📦 <strong>What I Did:</strong><br>
    - Removed outliers and checked correlations<br>
    - Applied PCA to reduce features from 19 ➝ 2<br>
    - Visualized class separation using scatter plots
    `,
    pdf: "pdfs/Day10.pdf"
  },
  11: {
  text: `<strong>📊 K-Means Clustering</strong><br>
  <ul>
    <li>🧮 Applied K-Means Clustering on an unlabeled social media dataset.</li>
    <li>🔄 Preprocessed the data with label encoding and scaling.</li>
    <li>📈 Used the Elbow Method to determine the optimal number of clusters.</li>
    <li>🔍 Visualized clusters and interpreted behavioral patterns.</li>
  </ul>`,
  pdf: "pdfs/Day11.pdf"
},
12: {
  text: `<strong>🔗 Hierarchical Clustering</strong><br>
  <ul>
    <li>🗂️ Implemented Hierarchical Clustering on the Mall Customer dataset.</li>
    <li>🌳 Created dendrograms using Ward linkage to find natural cluster cuts.</li>
    <li>👥 Used Agglomerative Clustering to segment customers by spending and income.</li>
  </ul>`,
  pdf: "pdfs/Day12.pdf"
},
13: {
  text: `<strong>🌐 DBSCAN Clustering</strong><br>
  <ul>
    <li>📉 Applied DBSCAN on Mall Customer data using Age, Income, and Spending Score.</li>
    <li>📏 Used KNN plot to determine the ideal ε (epsilon) value.</li>
    <li>🔍 Identified dense regions and detected outliers automatically.</li>
  </ul>`,
  pdf: "pdfs/Day13.pdf"
},
14: {
  text: `<strong>🧠 Neural Network Basics with TensorFlow</strong><br>
  <ul>
    <li>📊 Built a regression model on Auto MPG dataset to predict fuel efficiency.</li>
    <li>📉 Trained the neural network and visualized prediction performance.</li>
    <li>🤖 Understood difference between Traditional ML, Neural Networks, and Deep Learning.</li>
  </ul>`,
  pdf: "pdfs/Day14.pdf"
},
15: {
  text: `<strong>🔍 Deep Dive: Neural Networks & Activation Functions</strong><br>
  <ul>
    <li>🔁 Explored Feedforward NN, CNN, RNN, and LSTM architectures.</li>
    <li>⚙️ Understood how activation functions (ReLU, Sigmoid, Tanh) work.</li>
    <li>📉 Visualized activation curves using Matplotlib.</li>
  </ul>`,
  pdf: "pdfs/Day15.pdf"
},
16: {
  text: `<strong>👗 CNN for Fashion MNIST</strong><br>
  <ul>
    <li>👟 Trained a Convolutional Neural Network on the Fashion MNIST dataset.</li>
    <li>🧠 Applied Convolution, Pooling, and Flattening operations.</li>
    <li>🔍 Achieved high accuracy in clothing classification and visualized results.</li>
  </ul>`,
  pdf: "pdfs/Day16.pdf"
},
17: {
  text: `<strong>🎭 Sentiment Analysis with LSTM</strong><br>
  <ul>
    <li>🎬 Analyzed IMDB movie reviews using LSTM neural networks.</li>
    <li>🧹 Performed text preprocessing and sequence padding.</li>
    <li>📈 Achieved high accuracy and visualized loss & accuracy curves.</li>
  </ul>`,
  pdf: "pdfs/Day17.pdf"
},
18: {
  text: `<strong>🧮 Weights & Biases in Neural Networks</strong><br>
  <ul>
    <li>⚖️ Explored how weights and biases affect neuron outputs.</li>
    <li>📈 Visualized weight adjustment effects on model predictions.</li>
    <li>🛠️ Simulated updates with NumPy to build a strong conceptual foundation.</li>
  </ul>`,
  pdf: "pdfs/Day18.pdf"
},
19: {
  text: `<strong>🚀 Optimizers in Deep Learning</strong><br>
  <ul>
    <li>🧠 Compared optimizers: SGD, Momentum, RMSProp, and Adam.</li>
    <li>📊 Trained on MNIST dataset and visualized validation accuracy per epoch.</li>
    <li>📈 Learned how optimizers affect convergence and training speed.</li>
  </ul>`,
  pdf: "pdfs/Day19.pdf"
},
20: {
  text: `<strong>🔁 Backpropagation & Weight Updates</strong><br>
  <ul>
    <li>📉 Understood gradient descent and weight update mechanism.</li>
    <li>🧮 Implemented forward pass, error calculation, and backward pass with NumPy.</li>
    <li>💡 Learned how loss minimization happens through iterations.</li>
  </ul>`,
  pdf: "pdfs/Day20.pdf"
},
21: {
  text: `<strong>📉 Variance & Gradient Descent Types</strong><br>
  <ul>
    <li>📌 Understood low, high, and ideal variance in models.</li>
    <li>📉 Learned Batch, Stochastic, and Mini-Batch Gradient Descent.</li>
    <li>🎯 Applied learning rate tuning and loss optimization techniques.</li>
  </ul>`,
  pdf: "pdfs/Day21.pdf"
},
};
function showAllDays() {
  const daysContainer = document.getElementById('days-container');
  daysContainer.innerHTML = '';
  for (let day = 1; day <= 21; day++) {
    const btn = document.createElement('button');
    btn.innerText = `Day ${day}`;
    btn.onclick = () => {

      document.getElementById('entry-title').style.display = 'block';
      document.getElementById('entry-content').style.display = 'block';
      document.getElementById('project-details').style.display = 'none';
      document.getElementById('pdf-container').innerHTML = '';
      showEntry(day);
    };
    daysContainer.appendChild(btn);
  }
}

function showEntry(day) {
document.getElementById('project-buttons').style.display = 'none';
document.getElementById('project-details').style.display = 'none';
document.getElementById('project-details').innerHTML = '';
document.getElementById('diary-entry').style.display = 'block';

  const entry = diaryData[day];

  document.getElementById('entry-title').innerText = `Day ${day}`;
  let badges = '';
if (day <= 10) {
  badges += `<span class="badge">✔️ Completed</span>`;
} 
if (day === 14) {
  badges += `<span class="badge">🧠 Neural Networks Started</span>`;
} 
if (day === 21) {
  badges += `<span class="badge">🏆 Final Day</span>`;
}

let contentHtml = `
  <div class="summary-text">${entry?.text || 'No summary available.'}</div>
  <div>${badges}</div>
`;
  document.getElementById('entry-content').innerHTML = contentHtml;
  document.getElementById('entry-content').classList.add('fade-in');


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
      <iframe src="pdfs/training_certificate.pdf" width="100%" height="600px"></iframe>
    </div>
  `;
  viewer.scrollIntoView({ behavior: 'smooth' });
  confetti({
    particleCount: 120,
    spread: 90,
    origin: { y: 0.6 }
  });
}


function toggleProjects() {
  const projectSection = document.getElementById('project-buttons');
  const diarySection = document.getElementById('diary-entry');

  const isVisible = projectSection.style.display === 'block';
  projectSection.style.display = isVisible ? 'none' : 'block';

  diarySection.style.display = isVisible ? 'block' : 'none';

  if (isVisible) {
    document.getElementById('project-details').style.display = 'none';
    document.getElementById('project-details').innerHTML = '';
  }
}


function openFinalReport() {
  const viewer = document.getElementById('pdf-container');
  viewer.innerHTML = `
    <div class="pdf-viewer fade-in">
      <button onclick="closePDF()" class="close-pdf-btn">✖ Close PDF</button>
      <iframe src="pdfs/Training_Report.pdf" width="100%" height="700px" style="border:1px solid #ccc; border-radius:10px;"></iframe>
    </div>
  `;
  viewer.scrollIntoView({ behavior: 'smooth' });
}

function showProject(project) {
  const diarySection = document.getElementById('diary-entry');
  const container = document.getElementById('project-details');
  const viewer = document.getElementById('pdf-container');

  diarySection.style.display = 'none';

  const projects = {
    weather: {
      pdf: "pdfs/weather_prediction_report.pdf",
      github: "https://github.com/bhumi0529/Weather_Prediction_App.git",
      title: "🌦️ Weather Prediction App",
      text: "This mini-project uses a machine learning model to predict the likelihood of rain based on inputs like temperature, humidity, pressure, and wind speed. Built using Python, scikit-learn, and Streamlit, the app displays predictions with feature importance visualizations and user-friendly sliders."
    },
    iris: {
      pdf: "pdfs/Iris_prediction_report.pdf",
      github: "https://bhumi0529-iris-flower-prediction-app-iris-app-ayxzp9.streamlit.app",
      title: "🌸 Iris Flower Classification App",
      text: "The Iris project classifies flower species (Setosa, Versicolor, Virginica) based on their sepal and petal measurements. Built with Random Forest and Streamlit, it features feature importance plots and an elegant UI for input and prediction."
    },
    emotion: {
      pdf: "pdfs/Emotion_aware_report.pdf",
      github: "https://github.com/bhumi0529/Emotion-Aware-Virtual-Study-Companion.git",
      title: "🧠 Emotion-Aware Virtual Study Companion",
      text: "This major project detects student emotions in real-time using a webcam and suggests motivational tips accordingly. Built using TensorFlow, OpenCV, and Flask, the app enhances study efficiency through smart emotional feedback."
    }
  };

  const p = projects[project];

  container.innerHTML = `
    <div class="fade-in">
      <h3 style="color: #113F67;">${p.title}</h3>
      <p style="font-size: 1.4rem; color: #1B3C53; line-height: 1.6;">${p.text}</p>
      <div style="margin-top: 15px;">
        <button class="action-btn" style="background: #34699A; color: #fff; margin-right: 10px;" onclick="openPDF('${p.pdf}')">📄 View Report</button>
        <button class="action-btn" style="background: #34699A; color: #fff;" onclick="window.open('${p.github}', '_blank')">🔗 GitHub Repo</button>
      </div>
    </div>
  `;

  container.style.display = 'block';
  container.scrollIntoView({ behavior: 'smooth' });
  viewer.innerHTML = ''; 
  confetti({
  particleCount: 100,
  spread: 70,
  origin: { y: 0.6 }
});
}

function backToDiary() {
  document.getElementById('diary-entry').style.display = 'block';
  document.getElementById('project-details').style.display = 'none';
  document.getElementById('project-buttons').style.display = 'none';
  document.getElementById('pdf-container').innerHTML = '';
}



document.addEventListener('DOMContentLoaded', showAllDays);
