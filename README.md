# Heart Disease Prediction App

## Overview
This project focuses on developing a machine learning model to predict the likelihood of heart disease based on various patient health parameters. It features an end-to-end pipeline, from data preprocessing and model training to deployment as an interactive web application.

The solution includes:
* **Data Analysis & Preprocessing:** Thorough exploration and preparation of the heart disease dataset.
* **Machine Learning Model:** Training and evaluation of an optimal classification model (Support Vector Machine) for accurate predictions.
* **Interactive Web Application:** A user-friendly interface built with Streamlit for real-time predictions.
* **FastAPI Backend (Optional):** A robust API endpoint for model inference (though Streamlit app directly uses the model for this project).

## Key Features
* **Predictive Accuracy:** Utilizes a highly accurate machine learning model to provide reliable heart disease predictions.
* **Intuitive User Interface:** A simple and interactive web application built with Streamlit, allowing users to input patient parameters and get instant predictions.
* **Scalable Architecture:** Designed with modularity to allow easy integration with other systems or future enhancements.
* **Detailed Analysis Notebooks:** Comprehensive Jupyter notebooks detailing each step of the data science pipeline.

## Technologies & Libraries
* **Programming Language:** Python
* **Machine Learning:** scikit-learn, joblib
* **Data Manipulation:** Pandas, NumPy
* **Web Framework:** Streamlit (for frontend), FastAPI (for backend API, if used separately)
* **Development Tools:** Jupyter Notebooks, VS Code
* **Deployment:** Git, GitHub

## Project Structure

Heart_Disease_Project/
├── data/
├── deployment/
├── models/
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   └── 06_hyperparameter_tuning.ipynb
├── Screenshots/
├── ui/
│   └── streamlit_app.py
├── .gitignore
├── app.py
└── README.md

## How to Run Locally

To set up and run this project on your local machine, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/EmadAliEmad/Heart_Disease_Prediction_App.git](https://github.com/EmadAliEmad/Heart_Disease_Prediction_App.git)
    cd Heart_Disease_Prediction_App
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    * **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You might need to create a `requirements.txt` file first if you haven't already. You can generate it by running `pip freeze > requirements.txt` in your activated virtual environment).*

5.  **Run the Streamlit Application:**
    ```bash
    streamlit run ui/streamlit_app.py
    ```
    The application will open in your default web browser at `http://localhost:8501`.

## Screenshots

Here are some screenshots of the Heart Disease Prediction App in action:

### Streamlit User Interface
![Streamlit Interface](https://github.com/EmadAliEmad/Heart_Disease_Prediction_App/blob/main/Screenshots/Streamlit%20Interface.png?raw=true)

### Input Parameters
![Input Parameters](https://github.com/EmadAliEmad/Heart_Disease_Prediction_App/blob/main/Screenshots/Testing%20the%20labels.png?raw=true)

### Prediction Result
![Prediction Result](https://github.com/EmadAliEmad/Heart_Disease_Prediction_App/blob/main/Screenshots/Testing%20result.png?raw=true)

## Contributing
Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*(Note: You might need to create a simple LICENSE file in your root directory if you want to include it. A common MIT license file can be found online.)*

## Connect with Me

Best regards,

Emad Ali Emad
emadaliemad375@gmail.com

LinkedIn: [https://www.linkedin.com/in/emad-ali-emad-886647199](https://www.linkedin.com/in/emad-ali-emad-886647199)
GitHub: [https://github.com/EmadAliEmad](https://github.com/EmadAliEmad)
