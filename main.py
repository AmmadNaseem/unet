"""Shared training orchestration and CLI for the new three-class dataset."""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import tensorflow as tf
from data import prepare_dataset, make_dataset, save_prediction, plot_predictions
from model import unet


def prepare_run(args):
    """Inspect dataset, consume an optional preparation manifest, and record this run."""
    if args.size < 16 or args.size % 16:
        raise ValueError('Image size must be a positive multiple of 16')
    if min(args.epochs, args.batch_size, args.base_filters) < 1:
        raise ValueError('Epochs, batch size, and filters must be positive')
    tf.keras.utils.set_random_seed(42)
    images, masks, manifest = prepare_dataset(args.dataset, args.limit)
    prepared_path = Path('runs/prepared_dataset.json')
    prepared_used = False
    if prepared_path.exists():
        prepared = json.loads(prepared_path.read_text())
        if prepared['dataset'] == str(Path(args.dataset).resolve()) and prepared['limit'] == args.limit:
            if prepared['manifest'] != manifest:
                raise ValueError('Dataset selection changed; rerun dataPrepare.ipynb')
            manifest = prepared['manifest']
            prepared_used = True
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'split.json').write_text(json.dumps(manifest, indent=2))
    settings = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    settings.update(python_executable=sys.executable, prepared_manifest_used=prepared_used,
        evaluation_scope='same training samples - sanity check only' if args.limit else 'held-out test',
        seed=42, label_ids=[0,1,2], tensorflow=tf.__version__,
        devices=[str(x) for x in tf.config.list_physical_devices()])
    (args.output / 'config.json').write_text(json.dumps(settings, indent=2))
    return images, masks, manifest, settings


def train_model(args, train, val):
    """Build model.py's U-Net and save the best checkpoint and training history."""
    model = unet(input_size=(args.size,args.size,3), num_classes=3, base_filters=args.base_filters,
                 learning_rate=args.learning_rate, dropout_rate=args.dropout_rate)
    print('Dataset split:', {'train_batches': len(train), 'validation_batches': len(val)}, flush=True)
    print('Parameters:', model.count_params(), 'Hardware:', [str(x) for x in tf.config.list_physical_devices()], flush=True)
    history = model.fit(train, validation_data=val, epochs=args.epochs, verbose=2, shuffle=False, callbacks=[
        tf.keras.callbacks.ModelCheckpoint(str(args.output / 'best.keras'), monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-5),
        tf.keras.callbacks.CSVLogger(str(args.output / 'training.csv'))])
    (args.output / 'history.json').write_text(json.dumps(history.history, indent=2))
    return model, history


def evaluate_model(args, test, manifest, settings):
    """Evaluate the saved model and save real predictions via data.py."""
    model = tf.keras.models.load_model(args.output / 'best.keras')
    metrics = model.evaluate(test, return_dict=True, verbose=2)
    confusion = np.zeros((3,3), dtype=np.int64)
    pred_dir = args.output / 'predictions'
    pred_dir.mkdir()
    saved = 0
    for x,y in test:
        pred = np.argmax(model(x, training=False).numpy(), axis=-1)
        actual = y.numpy()
        confusion += np.bincount((actual * 3 + pred).ravel(), minlength=9).reshape(3,3)
        for j in range(len(pred)):
            if saved < 6:
                key = manifest['test'][saved]
                save_prediction(pred_dir, key, pred[j])
                saved += 1
    denom = confusion.sum(0) + confusion.sum(1)
    dice = np.divide(2*np.diag(confusion), denom, out=np.zeros(3), where=denom!=0)
    print("Predicted pixel counts by class ID:", confusion.sum(axis=0).tolist())
    metrics.update(predicted_pixels_by_class_id=confusion.sum(axis=0).tolist(), evaluation_scope=settings['evaluation_scope'], dice_by_class_id=dice.tolist(), macro_dice=float(dice.mean()), confusion_matrix=confusion.tolist())
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print('Evaluation metrics:', json.dumps(metrics), flush=True)
    print('Artifacts:', args.output.resolve(), flush=True)
    return model, metrics, pred_dir


def run_training(args):
    images, masks, manifest, settings = prepare_run(args)
    datasets = {name: make_dataset(images, masks, ids, size=args.size,
        batch_size=args.batch_size, training=name == 'train') for name,ids in manifest.items()}
    train_model(args, datasets['train'], datasets['validation'])
    model, metrics, pred_dir = evaluate_model(args, datasets['test'], manifest, settings)
    fig = plot_predictions(images, masks, manifest['test'], pred_dir, args.output)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('data/936'))
    parser.add_argument('--limit', type=int, default=5, help='Five-image sanity run by default')
    parser.add_argument('--all-images', action='store_true', help='Use all pairs with a held-out test split')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--base-filters', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    if args.all_images:
        args.limit = None
    args.dropout_rate = 0.0 if args.limit is not None else 0.5
    if args.output is None:
        args.output = Path('runs') / ('cli_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    run_training(args)


if __name__ == '__main__':
    main()
