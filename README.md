In-Silico-Prediction-of-Shelf-Life
This repository contains the code, data, and software associated with the manuscript "In Silico Prediction of Shelf-Life for Marinated Meat Products Using an Integrative Framework of Machine Learning and Text Mining".
This research introduces a novel framework that integrates machine learning (ML) and text mining (TM) to predict the shelf life of marinated meat products accurately and efficiently.
Key Features
Data Collection: A hybrid approach combining text mining and manual literature review to assemble a comprehensive dataset from Web of Science and CNKI.
Feature Engineering: Evaluation of 16 encoding methods to optimize the representation of categorical features, identifying 'Leave-One-Out' encoding as the optimal strategy.
Model Development: Comparison of 9 ML algorithms to identify the most effective model (RandomForest) for shelf-life prediction.
Feature Importance Analysis: Utilization of feature importance scores, SHAP values, and partial dependence plots to interpret key factors (Preservatives, Sterilization, etc.) influencing shelf life.
Robust Graphical User Interface (GUI): A user-friendly tool featuring automatic data validation and real-time performance verification, enabling continuous model retraining to ensure long-term reliability.
Repository Contents
The repository is organized into the following directories:
Data/: Contains the raw dataset (data22.csv) used for training and testing the machine learning models.
Scripts/: Contains the Jupyter Notebooks (.ipynb) for data preprocessing, statistical analysis, feature engineering, and model evaluation presented in the manuscript.
Software_Release/: (Recommended for End Users) Contains the standalone executable file (AIF.exe) and necessary dependencies. Windows users can run the software directly without installing Python.
Source_Code/: (For Developers) Contains the main Python source code (AIF.py), model files, and assets.
User_Guide/: Contains the detailed operation manual (Supplementary File 4) with step-by-step instructions for using the AIF software.
Figures/: Contains high-resolution figures used in the manuscript.
Quick Start
Option 1: Run the Standalone Software (No Python required)
Navigate to the Software_Release folder.
Download the entire folder content to your local machine.
Double-click AIF.exe to launch the application.
Refer to the User_Guide for detailed operating instructions.
Option 2: Run from Source Code
Clone this repository.
Install the required dependencies listed in Source_Code/requirements.txt:
code
Bash
pip install -r Source_Code/requirements.txt
Run the script:
code
Bash
python Source_Code/AIF.py
Contributing
Contributions to this project are welcome. Please feel free to open issues or submit pull requests for code improvements or data updates.
Citation
If you use this code, software, or data in your research, please cite the corresponding manuscript:
[Insert Citation Here after Publication]
Contact
For any questions or inquiries, please contact:
Qianli Huang: qianli_huang@yeah.net
