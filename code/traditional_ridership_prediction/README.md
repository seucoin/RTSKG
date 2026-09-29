Required dependencies:

```
conda create -n rider python=3.7
conda activate rider
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

The ridership data in `records` can be converted into atomic files, which can then be used to train models via `./libcity/run_model.py`. More information can be found in [UUKG](https://github.com/usail-hkust/UUKG) and [LibCity](https://github.com/LibCity/Bigscity-LibCity-Docs-zh_CN).