# 🏏 IPL Match Winner Prediction

This project is a **full-stack machine learning web application** designed to predict the outcome of Indian Premier League (IPL) matches using match-specific parameters. The system combines a responsive frontend, a robust FastAPI backend, and a trained machine learning model to deliver real-time predictions based on teams, venues, toss decisions, and match conditions.

The aim of this project is to demonstrate practical AI deployment by integrating model training, API development, and user-facing interfaces into a cohesive system.

---

## 🌟 Key Features

- 🧠 **ML Model Integration** – Predicts match outcomes with trained classification models
- 🌐 **User-Friendly Frontend** – Clean UI for team selection and result display
- 📊 **Swagger UI Docs** – Interactive API documentation using OpenAPI/Swagger
- 🧪 **Test Coverage** – Unit & integration tests for backend components
- 📁 **Modular Codebase** – Organized folder structure for easy maintenance

---

## 📌 Use Cases

- Fans curious about likely match outcomes
- Analysts exploring model-based insights
- Demonstrating practical ML application deployment
- Learning full-stack ML pipeline integration



## 📂 Project Structure
```markdown
- **frontend/**  
  UI implementation using React or HTML/CSS/JS

- **backend/**  
  FastAPI backend code  
  ├─ `main.py` – API entry point  
  ├─ `routes/` – API endpoint definitions  
  └─ `services/` – Business logic and model inference

- **models/**  
  Trained ML models (e.g., `model.pkl`)

- **data/**  
  ├─ `raw/` – Original dataset  
  └─ `processed/` – Cleaned dataset for training

- **tests/**  
  Unit and integration tests

- **docs/**  
  API documentation, model description, pipeline explanation

- `requirements.txt` – Python dependencies  
- `README.md` – Project documentation  
- `.gitignore` – Git ignored files

```
---

## 🚀 Getting Started

### 🔧 Setup Instructions

```bash
git clone https://github.com/[your-username]/ipl-prediction-model.git
cd ipl-prediction-model
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

## 🧠 Model Overview
**Algorithm:** Random Forest, XGBoost

**Features:** Team stats, venue, toss winner, innings, etc.

**Performance:** Accuracy, F1 Score, Confusion Matrix (details in model_training.ipynb)

## 🚀 Features Demo

Below is a preview of the IPL Prediction web app in action:

### 🖼️ Web Interface
![Screenshot (76)](https://github.com/user-attachments/assets/fec877f4-29d4-468d-b230-c18c7ed00349)
**Home Page**


## 🖥️ Frontend
Built with: [Django / HTML-CSS-JS]

**Features:**

- Team selection

- Venue input

- Live prediction output

![Screenshot (77)](https://github.com/user-attachments/assets/9b4c360d-3372-4cab-86eb-9fe33d1bf152)
**Predition Page**
- Select **teams**, **venue**, and **toss winner**
- Click **"Predict Winner"** to get real-time predictions
- Prediction results are shown with **confidence score**

## 📡 Backend API
**POST /predict**
- **Input:** Match info (teams, toss result, venue)

- **Output:** Predicted winner and confidence score

- Auto-documented with Swagger (/docs)


# 📊 Work in Progress
- Frontend Interface

- AI Model Training

- Backend Testing & Documentation (In Progress)

# 📬 Contact
**For questions or collaborations:**

- **📧 Email:** swatadru.paul2023@uem.edu.in

- **📱 Phone:** +91-9330776539

# 📝 License
This project is licensed under the MIT License.
```yaml
pytest --cov=backend tests/
```
