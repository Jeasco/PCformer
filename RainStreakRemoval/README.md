
## Training
- Download the [datasets](datasets/README.md) and run

```
python prepare.py
```
-  Download the LEI pre-trained model and place it in './checkpoints/'

- Train the model with default arguments by running

```
python train.py
```


## Evaluation

1. Download the pre-trained model and place it in `./checkpoints/`

2. Download test datasets and place them in `./datasets/`

3. Run
```
python test.py
```



