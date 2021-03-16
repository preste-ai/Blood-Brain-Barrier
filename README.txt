Requirements:

1) Install Anaconda

2) Open a terminal

3) Clone repository

        git clone https://github.com/preste-ai/BBB.git

4) In the terminal navigate to the project root folder

	cd path/BBB

5) Create environment using Anaconda and activate it

        conda create --name bbb --file spec-file.txt
        conda activate bbb

Usage:

1) Paste the SMILES for molecules for which predictions to be made in the following file
 
	data/predictions/input.csv

2) Insert the output to predict in the "target" from "predictions" section of  parameters.json.
   Three values of output are possible: "substrates", "inhibitorsA2", "inhibitorsB1"

3) Run script for getting predictions  using following syntax:

        python3 run_predictions.py

4) Find the predictions in the *.csv files of data/predictions/output directory
