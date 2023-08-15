<!DOCTYPE html>
<html>

<head>
  <title>Yoga Classifier</title>
</head>

<body>
  <h1>Yoga Classifier</h1>

  <h2>Table of Contents</h2>
  <ul>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#projectOverview">Project Overview</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#improvements">Improvements</a></li>
  </ul>

  <h2 id="introduction">Introduction</h2>
  <p>Welcome to the Yoga Classifier project! This project was created for testing and evaluating yoga poses. This
    classifier is designed to recognize and categorize different yoga poses.</p>

  <h2 id="Overview">Project Overview</h2>
  <p>The Yoga Classifier is an AI-based system that uses state-of-the-art machine learning techniques to identify various
    yoga poses based on input images or videos. The project's core functionality lies in its ability to analyze the key
    features of different yoga postures and classify them into predefined categories, making it a valuable asset for
    yoga instructors, fitness centers, and wellness companies.</p>

  <h2 id="features">Features</h2>
  <ul>
    <li><strong>Accuracy</strong>: The Yoga Classifier is trained on an extensive dataset, allowing it to achieve high
      accuracy in identifying yoga poses.</li>
    <li><strong>Customizability</strong>: The project can be further trained on specific yoga poses or modified to suit
      unique requirements.</li>
  </ul>

  <h2 id="usage">Usage</h2>
  <p><strong>Classification</strong>: The system will process the input and accurately identify the yoga pose from its
    trained categories.</p>

  <h2 id="improvements">Improvements</h2>
  <ul>
    <li><strong>First Improvement</strong>: The accuracy was 62%.</li>
    <li><strong>Second Improvement</strong>: The accuracy went to 69%, due to an increase in the number of epochs.</li>
    <li><strong>Third Improvement</strong>: The accuracy went to 86% again due to augmentation and another increase in
      the number of epochs.</li>
  </ul>

  <h1> Birds Species EfficientNetb0</h1>

  <h2>Table of Contents</h2>
  <ul>
    <li><a href="#birdsIntroduction">Introduction</a></li>
    <li><a href="#birdsDataset">Dataset</a></li>
    <li><a href="#birdsResults">Results</a></li>
  </ul>

  <h2 id="Introduction">Introduction</h2>
  <p>Welcome to the Bird Species Classification project! The primary objective of this project is to classify bird
    species. To achieve accurate and efficient classification, I have utilized the EfficientNet model, known for its
    superior performance and parameter efficiency.</p>

  <h2 id="Dataset">Dataset Source:</h2>
  <p>The dataset used for this project was obtained from Kaggle, a renowned platform for data science competitions and
    datasets. It comprises a diverse collection of bird images, representing various bird species from around the
    world.</p>
  <p>The dataset used for this project was obtained from <a href="https://www.kaggle.com/dataset-url"
      target="_blank">Kaggle</a></p>

  <h2>Key Features:</h2>
  <ul>
    <li>Bird species classification using EfficientNet model</li>
  </ul>

  <h2>Technologies Used:</h2>
  <ul>
    <li>Python for scripting and development</li>
    <li>PyTorch as the deep learning framework</li>
    <li>EfficientNet model for bird species classification</li>
    <li>Kaggle API for dataset access</li>
  </ul>

  <h2 id="Results">Results and Outcomes:</h2>
  <p>The model achieved impressive results after 20 epochs of training. Below are the key metrics obtained:</p>

  <table>
    <tr>
      <th>Epoch</th>
      <th>Train Loss</th>
      <th>Train Accuracy</th>
      <th>Train Precision</th>
      <th>Train Recall</th>
      <th>Validation Accuracy</th>
      <th>Validation Precision</th>
      <th>Validation Recall</th>
    </tr>
    <tr>
      <td>20</td>
      <td>0.0376</td>
      <td>0.9885</td>
      <td>0.9885</td>
      <td>0.9885</td>
      <td>0.9676</td>
      <td>0.9742</td>
      <td>0.9676</td>
    </tr>
  </table>

  <h1>Stick Blur or Not Classification</h1>

  <h2>Table of Contents</h2>
  <ul>
    <li><a href="#stickIntroduction">Introduction</a></li>
    <li><a href="#stickDataset">Dataset</a></li>
    <li><a href="#stickResults">Results and Outcomes</a></li>
  </ul>

  <h2 id="Introduction">Introduction</h2>
  <p>Welcome to the Stick Blur or Not Classification project! The main goal of this project is to classify images into
    two categories: stick blur or not. To achieve this task, a deep learning model has been trained and evaluated.</p>

  <h2 id="Dataset">Dataset</h2>
  <p>The dataset used for this project was self-collected. It consists of images that have either stick blur or not.
    The dataset was prepared to facilitate the training and evaluation of the classification model.</p>

  <h2>Key Features:</h2>
  <ul>
    <li>Stick Blur or Not classification using EfficientNet model</li>
  </ul>

  <h2>Technologies Used:</h2>
  <ul>
    <li>Python for scripting and development</li>
    <li>PyTorch as the deep learning framework</li>
    <li>EfficientNet model for Stick Blur classification</li>
  </ul>

  <h2 id="Results">Results and Outcomes:</h2>
  <p>The model was trained for 5 epochs and achieved the following results:</p>

  <table>
    <tr>
      <th>Epoch</th>
      <th>Train Loss</th>
      <th>Train Accuracy</th>
      <th>Validation Loss</th>
      <th>Validation Accuracy</th>
      <th>Test Accuracy</th>
      <th>Average Inference Time per Iteration</th>
    </tr>
    <tr>
      <td>5</td>
      <td>0.1277</td>
      <td>0.9729</td>
      <td>0.1004</td>
      <td>0.9667</td>
      <td>0.9355</td>
      <td>0.0473 seconds</td>
    </tr>
  </table>
</body>

</html>
