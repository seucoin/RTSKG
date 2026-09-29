Required dependencies:

```
conda create -n kge python=3.7
conda activate kge
pip install -r requirements.txt
```

Preprocess the dataset to get .pickle files
```
python datasets/process.py
```

Use run.py to train the knowledge graph embedding model.