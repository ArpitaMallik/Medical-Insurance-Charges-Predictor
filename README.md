# Medical Insurance Charge Prediction

This is a FastAPI-based web application that predicts medical insurance charges based on user inputs like age, BMI, smoking status, and region. The model is trained using XGBoost with hyperparameter tuning.

## 📌 Features
- Predicts insurance charges using a trained **XGBoost** model.
- **FastAPI** backend for handling predictions.
- **Tailwind CSS** & **Bootstrap** frontend for a clean UI.
- **Scikit-learn** for feature scaling.
- CORS enabled for API accessibility.

## 🚀 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/medical-insurance-prediction.git
cd medical-insurance-prediction


### 2️⃣ Create a Virtual Environment
python -m venv venv
Activate it:

Windows: venv\Scripts\activate

Mac/Linux: source venv/bin/activate

### 3️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt

### 4️⃣ Run the Application
bash
Copy
Edit
uvicorn app:app --reload
The API will be available at http://127.0.0.1:8000
