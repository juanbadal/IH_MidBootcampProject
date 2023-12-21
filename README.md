Ironhack Mid Bootcamp Project

Collaborators:
- Olabisi Matthew: https://github.com/olabisimatthew
- Joaquín González: https://github.com/joacog86
- Juan Badal: https://github.com/juanbadal


Objective:

How can we assist Hosts in their pricing strategy for their listing? → Goal is to have a successful booking.


Conclusions:

Model is decent for a Proof of Concept, but it might not be good enough for production with real users.
We can also train the models with the group of listings that have good bookability, so as to model the successful listings


Next Steps:

Possible improvements to the model:
Different feature selection
Alternative preprocessing techniques
Feature engineering
Try additional models
When the model is good enough, expand Proof of Concept to other geographies.
Creating an app/API for recommending the price



Data obtained from insideairbnb.com/get-the-data/


Structure of the repository:
- README.md: This file
- mbproject_cleaning.ipynb: Jupyter notebook used to clean the raw data
- mbproject_analysis.ipynb: Jupyter notebook used to perform the data analysis, preprocessing and machine learning
- requirements-dev.in: Compilation of libraries and versions used in the project
- requirements.in: Compilation of the previous requirements plus other that were needed for the infrastructure (jupyter, ipykernel)
- .git: Git version tracker folder
- .ipynb_checkpoints: Ipynb version tracker folder
- data: Folder containing the datasets:
	- raw: Contains the raw dataset (listings.csv)
	- cleaned: Contains the cleaned dataset (listings_cleaned.csv)
- models: Contains models used in the analysis notebook
- scalers: Contains scalers used in the analysis notebook
- slides: Contains a link to the presentation in Google slides
- src:
	- lib: Contains the file with all the functions used in the project (functions.py)
- transformers: Contains the transformers used in the analysis notebook