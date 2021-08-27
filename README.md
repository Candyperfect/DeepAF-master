# DeepDF
"[DeepDF: Improving protein function prediction via adaptively fusing information from protein texts and sequences]"

# Environment Settings 
* python == 3.7   
* Pytorch == 1.1.0  
* Numpy == 1.16.2  
* SciPy == 1.4.1    
* scikit-learn == 0.22.2  

# Usage 
````
CUDA_VISIBLE_DEVICES=1 nohup python model_main2016.py 
````

# Data
## Link
* **experimental annotations of proteins and corresponding SwissProt data**: (http://deepgoplus.bio2vec.net/data/)  
* **├─2016 version dataset
* **└─2021 version dataset  
* **protein texts**: (https://www.uniprot.org/citations/) 
 

## Usage
Please first **unzip** the data folders and then use. The files in folders are as follows:
````
2016data/
├─2016train.pkl: training set, include protein sequences, protein texts and protein function annotations.  
├─2016test.pkl: testing set, include protein sequences, protein texts and protein function annotations.  
├─2016Alltrain_Scores.mat: the Blast scores between train proteins.  
├─2016Alltest_Scores.mat: the Blast scores between train and test proteins
└─2016GOtermsnew_bmc.pkl: studied GO terms of three sub-ontologies

````

