# Machine Learning Project – Summer Semester 2025

This project was developed as part of the "Machine Learning" course at HTW Saar in the summer semester 2025 in "Practical Computer Science". The goal is to predict the genres of a game based on its description using various machine learning techniques.

## Project Overview

We use a cleaned Steam dataset containing game descriptions and genre labels as well as many other feature values. The main challenge was to build a robust multi label classification model that can handle multiple genres per game and work with a relatively small dataset due to computational constraints.

Our workflow includes:
- Data cleaning and preprocessing
- Feature extraction
- Multi label genre encoding
- Model selection and evaluation
- Optimization suggestions for future work

## Dataset

The dataset used for this project is available here:  
[Steam Games Dataset from Kaggle](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset/data)

## Repository

The full project, including the Jupyter Notebook, code, results and all data set sizes used, can be found on GitHub:  
[GitHub FlorianSpeicher04/machine-learning](https://github.com/FlorianSpeicher04/machine-learning)

## Large File Storage (git-lfs)

Some files in this repository (such as the datasets) are managed using [git-lfs](https://git-lfs.github.com/).  
To clone the repository with all large files, please make sure you have git-lfs installed:

```sh
git lfs install
git clone https://github.com/FlorianSpeicher04/machine-learning
```

## How to Run

1. Clone the repository (see above).
2. Install the required Python packages.
3. Open `notebook.ipynb` in Jupyter Notebook or VS Code.
4. Follow the steps in the notebook to reproduce the results (Run All).

## Results

Our model achieves reasonable performance given the dataset size and computational limitations. For more details, see the evaluation and conclusion sections in the notebook.

## Contributors

- Maximilian Kany
- Florian Speicher 5014185
- Tim Wall 5014365
