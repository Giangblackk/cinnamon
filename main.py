import argparse
import cv2
import numpy as np

from model import build_model


WEIGHT_PATH = 'model/cnn.h5'


def main():
    # config
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True,
                        help='Path to test image')
    args = parser.parse_args()

    # Load model
    model = build_model()
    model.load_weights(WEIGHT_PATH)

    # Predict
    X = cv2.imread(args.image)
    # resize
    X = cv2.resize(X, (224, 224))
    # expand to batch
    X = np.expand_dims(X, axis=0)

    # predict with X
    print(model.predict(X))


if __name__ == "__main__":
    main()
