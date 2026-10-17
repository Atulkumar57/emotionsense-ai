import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model import build_model
from utils import load_fer2013, split_train_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="emotionsense.h5")
    args = ap.parse_args()

    print("Loading data...")
    X, y, usage = load_fer2013(args.csv)
    X_tr, y_tr, X_va, y_va = split_train_val(X, y, usage)
    print("Train:", X_tr.shape, "Val:", X_va.shape)

    gen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
    )
    gen.fit(X_tr)

    model = build_model()
    model.summary()

    callbacks = [
        ModelCheckpoint(args.out, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]

    model.fit(
        gen.flow(X_tr, y_tr, batch_size=args.batch),
        validation_data=(X_va, y_va),
        epochs=args.epochs,
        callbacks=callbacks,
    )
    print("Saved best model to", args.out)


if __name__ == "__main__":
    main()
