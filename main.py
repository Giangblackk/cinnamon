import argparse
import cv2
import numpy as np
import pickle

from model import build_model


WEIGHT_PATH = 'model/cnn.h5'
LABELS_PATH = 'model/labels.pickle'


def main():
    # config
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True,
                        help='Path to test image')
    args = parser.parse_args()

    # Load model
    model = build_model()
    model.load_weights(WEIGHT_PATH)

    # Load labels
    with open(LABELS_PATH, 'rb') as f:
        # idx_to_label
        labels = pickle.load(f)

    # Load data and preprocess
    X = cv2.imread(args.image)
    # resize
    X = cv2.resize(X, (224, 224))
    # expand to batch
    X = np.expand_dims(X, axis=0)

    # predict with X
    pred = model.predict(X)
    label_idx = int(pred[0][0])
    print(labels[label_idx])


if __name__ == "__main__":
    main()
