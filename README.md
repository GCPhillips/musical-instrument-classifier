### About

This is a musical instrument classifier that works with two different network architectures:

- Convolutional Neural Network
- Transformer

The report over this project can be found [here](https://github.com/GCPhillips/musical-instrument-classifier/blob/main/report.pdf).

### Data

The dataset used is the [Medley Solos DB](https://zenodo.org/records/1344103).  The instruments within the recordings are:

- Clarinet
- Distorted Electric Guitar
- Female Singer
- Flute
- Piano
- Tenor Saxophone
- Trumpet
- Violin


### Training

During training, a random slice of two seconds are used to have a more diverse training set.  Once training is complete, a file should be saved under `./trained_models/`.

To train, run:

##### CNN:
`python3 train.py --model cnn --epochs 15 --lr 1e-4`

##### Transformer:
`python3 train.py --model transformer --epochs 15 --lr 1e-4`

where `--lr` is the learning rate (`1e-4` was found to be the best) and `--epochs` is the number of epochs.

Once training is completed, the loss over training epochs should be displayed.  Also, a confusion matrix from the test set will be displayed along with the accuracy report printed to console.
