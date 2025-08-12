# Kaggle bike sharing
Kaggle bike-sharing competition solution using day.csv dataset (Colab notebook)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/giommazz/kaggle-bike-sharing/blob/main/bike%E2%80%91sharing%E2%80%91solution-day.ipynb)


## Setup
Requires Python 3.10+.
```bash
git clone https://github.com/giommazz/kaggle-bike-sharing.git
cd kaggle-bike-sharing                                           # navigate into project dir
python3.10 -m venv .venv                                         # create virtual environment '.venv' using Python 3.10
source .venv/bin/activate                                        # activate virtual environment
pip install --upgrade pip                                        # ensure pip is upgraded inside isolated environment
pip install -r requirements.txt                                  # install project dependencies from 'requirements.txt'
```

## Data
The notebook automatically downloads the required `day.csv` file from its source.  
No manual data download required, simply run the notebook from start to finish, in Colab or locally (data will be fetched during execution).

## Usage
### Run in Google Colab
Click the **Open in Colab** badge at the top of the notebook (no local setup required).
### Run locally
1. Complete the [Setup](#setup) steps.
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook bike-sharing-solution-day.ipynb
   ```
