# Intelligent Report Generator

This project is designed to generate intelligent reports using deep learning and data preprocessing techniques. It includes data cleaning, merging, and advanced modeling combining Computer Vision for brand recognition through transfer Learning, CNN and BiLSTM for sentiment analysis, and pre-trained LLM to generate automated reports detailing top product features, usage scenarios…

## System Architecture

The following diagram illustrates the end-to-end workflow of the Intelligent Report Generator, from data preprocessing to automated report generation.

<p align="center">
  <img src="uploads/Architecture.png" alt="System Architecture" width="900">
</p>

## Project Structure

- `app.py`: Main application script.
- `cleaned_dataset.csv`: Cleaned dataset used for modeling.
- `merged_electronics_dataset.csv`: Merged dataset for electronics data.
- `DL_Project_Preprocessing.ipynb`: Jupyter notebook for data preprocessing and exploration.
- `Preliminary_Results/`: Folder containing preliminary results and experiments.
- `BiLSTM_Mistral.ipynb`: Notebook for BiLSTM model experiments.

## Final Interface

The web interface allows users to upload an image, perform brand recognition, analyze customer sentiment, and automatically generate an AI-powered product report.

<p align="center">
  <img src="uploads/Interface.png" alt="Final Interface" width="900">
</p>

## Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/inesmrad/Intelligent-Report-Generator.git
   cd Intelligent-Report-Generator
   ```

2. **Install dependencies:**
   - Ensure you have Python 3.8+ installed.
   - Install required packages (if any, e.g., pandas, numpy, torch, etc.).
   - You can use pip or conda as needed.

3. **Run the application:**
   - Use `app.py` as the entry point for the main functionality.
   - Explore the notebooks for data preprocessing and modeling steps.

## Notebooks

- **DL_Project_Preprocessing.ipynb**: Data cleaning, merging, and feature engineering.
- **Preliminary_Results/BiLSTM_Mistral.ipynb**: Deep learning experiments with BiLSTM models.

## Data

- `cleaned_dataset.csv` and `merged_electronics_dataset.csv` are provided for experimentation and model training.

## Results

- Preliminary results and model outputs are stored in the `Preliminary_Results/` directory.
