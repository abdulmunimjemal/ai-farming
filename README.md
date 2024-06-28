# 🌾 Agricultural AI Assistant 🌿

Welcome to the Agricultural AI Assistant! This application predicts the type of fertilizer required, recommends crops to plant, and diagnoses crop diseases from images.

## 📚 Table of Contents

- [🌾 Agricultural AI Assistant 🌿](#-agricultural-ai-assistant-)
  - [📚 Table of Contents](#-table-of-contents)
  - [🌟 Introduction](#-introduction)
  - [🚀 Features](#-features)
  - [📂 Project Structure](#-project-structure)
  - [🛠️ Installation](#️-installation)
  - [📈 Usage](#-usage)
  - [🌍 Deployment](#-deployment)
  - [🤝 Contributing](#-contributing)
  - [📝 License](#-license)
  - [💬 Acknowledgements](#-acknowledgements)
  - [👥 Group Members](#-group-members)

## 🌟 Introduction

This project is the final assignment for the Fundamentals of AI Course at Addis Ababa Institute of Technology (AAiT). It leverages machine learning models to assist farmers in making informed decisions about fertilizers, crops, and disease management.

## 🚀 Features

- **Fertilizer Prediction**: Predicts the type of fertilizer required based on soil composition.
- **Crop Recommendation**: Recommends the best crop to plant based on soil and weather conditions.
- **Disease Diagnosis**: Diagnoses crop diseases from images of crop leaves.

## 📂 Project Structure

```markdown
-/
├── models/
│ ├── crop_disease_model.pth
│ ├── crop-recommendation-model.joblib
│ └── fertilizer-recommendation-model.joblib
├── utils.py
├── main.py
├── requirements.txt
└── README.md
```

## 🛠️ Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/abdulmunimjemal/ai-farming.git
   cd ai-farming
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📈 Usage

1. **Run the Streamlit app**:

   ```bash
   streamlit run main.py
   ```

2. **Navigate to the app in your browser**: The app will typically open at `http://localhost:8501`.

## 🌍 Deployment

To deploy your Streamlit app, follow these steps:

1. **Push your project to GitHub**.
2. **Sign up on [Streamlit Community Cloud](https://streamlit.io/cloud)**.
3. **Deploy your app** by connecting to your GitHub repository and specifying the `main.py` file.

## 🤝 Contributing

We welcome contributions! Please follow these steps to contribute:

1. **Fork the repository**.
2. **Create a new branch**:

   ```bash
   git checkout -b feature-branch
   ```

3. **Make your changes**.
4. **Commit your changes**:

   ```bash
   git commit -m "feat: add new feature"
   ```

5. **Push to the branch**:

   ```bash
   git push origin feature-branch
   ```

6. **Create a pull request**.

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 💬 Acknowledgements

Created with ❤️ at [AAiT](https://www.aait.edu.et/).

Special Thanks to the Knowledge and Support of our Instructor [Amanuel Mersha](https://www.linkedin.com/in/leobitz/) and the Kaggle OpenSource Community.

## 👥 Group Members

- Abdulmunim Jundurahman (GL)
- Fethiya Safi
- Fuad Mohammed
- Salman Ali
- Obsu Kebede
