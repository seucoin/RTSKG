Before running the code, please install the required dependencies:

```
conda create -n rider python=3.7
conda activate rider
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

Use ./libcity/run_model.py to train spatial temporal prediction models.

