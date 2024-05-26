## This repository contains the code for the thesis "The estimation of Value-at-Risk and Expected Shortfall based on deep generative models"
---
Author:  **Belonovskiy Peter Ilich, HSE DSBA** 

Supervisor: **Naumenko Vladimir Vladimirovich, HSE Associate Professor**

### Installation
---
The dependency managament in project was implemented via [poetry](https://python-poetry.org).
To clone this repository and set up the environment, run the following commands:
```bash
git clone https://github.com/BELONOVSKII/var_es_dgm.git
cd var_es_dgm
poetry install
poetry shell
```
Note, poetry should be pre-installed in your system.

### Download data
---
Thesis uses daily stock prices data from yahoo finance. To parse the yahoo finance and download data run:
```python
python var_es_dgm/data_parcing/parse_yfinance.py 
```
This downloads individual stocks's data and produces combined file `data/complete_stocks.csv` that would be further used in the experiments.