Before running the code, please install the required dependencies:

```
conda create -n kge python=3.7
conda activate kge
pip install -r requirements.txt
```

preprocess the dataset to get .pickle files
```
python datasets/process.py
```

use run.py to train the knowledge graph embedding model.