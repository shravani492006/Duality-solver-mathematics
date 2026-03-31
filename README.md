# 🚀 Duality Solver – Linear Programming Web App

A web-based application built using Streamlit that solves Linear Programming (LP) problems using methods like Duality . The app supports multiple input formats and provides efficient solutions for optimization problems.

---

## 🌐 Live Demo

🔗 https://duality-solver-mathematics-5tfgpgmylqdm9p2fyvetzh.streamlit.app/

---

## 📌 Features

* ✅ Solve Linear Programming problems
* ✅ Supports Duality Method
* ✅ Supports Big M Method
* ✅ Multiple input formats:

  * Text input
  * CSV file upload
* ✅ Automatic parsing of equations
* ✅ User-friendly interface using Streamlit
* ✅ Error handling for invalid inputs

---

## 🛠️ Tech Stack

* Python 
* Streamlit
* NumPy
* Pandas
* SciPy
* Matplotlib
* OpenCV (for image preprocessing)
* Pytesseract (optional OCR support)

---

## 📥 Input Formats

### 1. Text Input

Example:
Minimize Z = 2x1 + 3x2
x1 + x2 >= 4
x1 - x2 = 1

---

### 2. CSV Input

Example format:

type,x1,x2,value
objective,2,3,
constraint,1,1,>=4
constraint,1,-1,=1

---

## ⚙️ How to Run Locally

1. Clone the repository:
   git clone https://github.com/your-username/duality-solver.git

2. Navigate to the project folder:
   cd duality-solver

3. Install dependencies:
   pip install -r requirements.txt

4. Run the app:
   streamlit run app.py

---

## 📸 Screenshots
<img width="1909" height="856" alt="Screenshot 2026-03-30 185249" src="https://github.com/user-attachments/assets/3b8b36ec-7059-4a14-b4cf-fb41c0b11d2b" />

---

## 👥 Team Members

* Shravani Kadam (Project Lead)
* Anushka Gadhave
* Tanvi Jaware
* Deepali Ganachari
* Vaishnavi Kunte
* Srushti Chavan

---

## 🚧 Future Scope

* 📷 Image-based input using OCR (Mathpix / Google Vision API)
* ✍️ Handwritten equation recognition
* 🤖 AI-based smart equation parsing
* 📱 Mobile-friendly UI improvements
* 📊 Step-by-step visualization of solving process

---

## 🤝 Contribution

Contributions are welcome! Feel free to fork the repository and submit a pull request.

-

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
