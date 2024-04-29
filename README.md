# NeuralFingerPaint
Audio generation using CGAN conditioned on audio fingerprints.

# Installation Guide


Clone the repository and install required packages:

```
git clone https://github.com/chymaera96/NeuralFingerPaint.git
pip install -r requirements.txt
cd NeuralFingerPaint
```

# Training

The models are trained on the PiJAMA [dataset](https://zenodo.org/record/8354955). Change the config file `config/default.yaml` to include the correct training directory containing the data files. 

```
python train.py --ckp=CHECKPOINT_NAME
```

## Evaluation

If required, adjust `config/default.yaml` according to required specfications. To reproduce Frechet distance metrics for a validation set, run:
```
python eval.py --ckp=CHECKPOINT_PATH --density=DENSITY --metric=EVAL_METRIC
```
The density hyperparameter is an optional argument which offers some degree of control over the input condition. Choose the eval metric as `FID` (Frechet Inception Distance) or `FAD` (Frechet Audio Distance).

