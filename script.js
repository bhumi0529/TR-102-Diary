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
},
14: {
  text: "Introduced to Neural Networks using TensorFlow. Built and trained a simple model to predict car MPG from horsepower using the Auto MPG dataset. Visualized predictions and understood key differences between ML, Neural Networks, and Deep Learning.",
  pdf: "pdfs/Day14.pdf"
},
15: {
  text: "Explored different types of Neural Networks including Feedforward, CNN, RNN, and LSTM. Studied the role of Activation Functions (ReLU, Sigmoid, Tanh) in adding non-linearity. Visualized them using Matplotlib for better understanding.",
  pdf: "pdfs/Day15.pdf"
},
16: {
  text: "Built a Convolutional Neural Network (CNN) using TensorFlow on Fashion MNIST dataset. Applied convolutional and pooling layers, visualized predictions, and achieved high accuracy in classifying clothing items.",
  pdf: "pdfs/Day16.pdf"
},
17: {
  text: "Implemented Sentiment Analysis on the IMDB movie review dataset using LSTM (Long Short-Term Memory) networks. Preprocessed sequences, trained on padded inputs, and achieved high accuracy in classifying reviews as positive or negative. Visualized training performance over epochs.",
  pdf: "pdfs/Day17.pdf"
},
18: {
  text: "Explored the difference between Weights and Bias in neural networks. Understood how weights control the influence of inputs and how bias shifts activation. Implemented a small demo to visualize how changing weights and bias affects neuron output.",
  pdf: "pdfs/Day18.pdf"
},
19: {
  text: "Explored different deep learning optimizers including SGD, Momentum, RMSProp, and Adam. \
Compared their training performance using MNIST dataset with TensorFlow and visualized validation accuracy over epochs.",
  pdf: "pdfs/Day19.pdf"
},
20: {
  text: "Explored how neural networks learn by updating weights using backpropagation and gradient descent. \
Understood the role of learning rate and gradients in adjusting weights to minimize loss. \
Implemented weight update logic using NumPy to simulate training a simple model.",
  pdf: "pdfs/Day20.pdf"
},
21: {
  text: "Explored the concept of Variance in Machine Learning including low, high, and ideal variance. \
Also understood the Gradient Descent algorithm in detail, including types like Batch, Stochastic, and Mini-Batch Gradient Descent. \
Learned how it helps in optimizing the loss function and updating weights in neural networks.",
  pdf: "pdfs/Day21.pdf"
}
 
};

function showAllDays() {
  const daysContainer = document.getElementById('days-container');
  daysContainer.innerHTML = '';
  for (let day = 1; day <= 21; day++) {
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
  const section = document.getElementById('project-buttons');
  section.style.display = section.style.display === 'none' ? 'block' : 'none';
}
function openFinalReport() {
  const viewer = document.getElementById('pdf-container');
  viewer.innerHTML = `
    <div class="pdf-viewer fade-in">
      <button onclick="closePDF()" class="close-pdf-btn">✖ Close PDF</button>
      <iframe src="pdfs/final_report.pdf" width="100%" height="700px" style="border:1px solid #ccc; border-radius:10px;"></iframe>
    </div>
  `;
  viewer.scrollIntoView({ behavior: 'smooth' });
}


function showProject(project) {
  const projects = {
    weather: {
      pdf: "pdfs/weather_prediction_report.pdf",
      github: "https://github.com/bhumi0529/Weather_Prediction_App.git",
      title: "🌦️ Weather Prediction App",
      text: "This mini-project uses a machine learning model to predict the likelihood of rain based on inputs like temperature, humidity, pressure, and wind speed. Built using Python, scikit-learn, and Streamlit, the app displays predictions with feature importance visualizations and user-friendly sliders.This project uses machine learning to predict rainfall using weather parameters"
    },
    iris: {
      pdf: "pdfs/Iris_prediction_report.pdf",
      github: "https://bhumi0529-iris-flower-prediction-app-iris-app-ayxzp9.streamlit.app/",
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
  const container = document.getElementById('project-details');
  container.innerHTML = `
    <div>
      <h3 style="color: #113F67;">${p.title}</h3>
      <p style="font-size: 1.4rem; color: #1B3C53; line-height: 1.6;">${p.text}</p>
      <div style="margin-top: 15px;">
        <button class="action-btn" style="background: #34699A; color: #fff; margin-right: 10px;" onclick="openPDF('${p.pdf}')">📄 View Report</button>
        <button class="action-btn" style="background: #34699A; color: #fff;" onclick="window.open('${p.github}', '_blank')">🔗 GitHub Repo</button>
      </div>
    </div>
  `;

  container.scrollIntoView({ behavior: 'smooth' });
}



document.addEventListener('DOMContentLoaded', showAllDays);
