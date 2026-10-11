"""Shared training orchestration and CLI for the new three-class dataset."""
import argparse
import csv
import time
from copy import deepcopy
from sklearn.metrics import f1_score, confusion_matrix
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



def configure_gpu(require_gpu=False):
    """Enable incremental GPU allocation and fail early when GPU is required."""
    gpus = tf.config.list_physical_devices('GPU')
    if require_gpu and not gpus:
        raise RuntimeError('GPU required but TensorFlow detects none. Use Colab GPU or the Ubuntu WSL2 .venv-wsl kernel; the Windows .venv is CPU-only.')
    for gpu in gpus:
        if not tf.config.experimental.get_memory_growth(gpu):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as error:
                raise RuntimeError('Restart the notebook kernel and configure the GPU before running TensorFlow operations.') from error
    return gpus


def prepare_run(args):
    """Inspect dataset, consume an optional preparation manifest, and record this run."""
    if args.size < 16 or args.size % 16:
        raise ValueError('Image size must be a positive multiple of 16')
    if min(args.epochs, args.batch_size, args.base_filters) < 1:
        raise ValueError('Epochs, batch size, and filters must be positive')
    configure_gpu(getattr(args, "require_gpu", False))
    # Defaults also support earlier notebook configurations.
    for key, value in {'seed':42, 'depth':4, 'loss_mode':'cce', 'dice_weight':1.0,
                       'augmentation':False, 'class_names':['class_0','class_1','class_2']}.items():
        if not hasattr(args, key): setattr(args, key, value)
    if args.size % (2**args.depth):
        raise ValueError('Image size must be divisible by 2**depth')
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    images, masks, manifest = prepare_dataset(args.dataset, args.limit, seed=args.seed)
    prepared_path = Path('runs/prepared_dataset.json')
    prepared_used = False
    if prepared_path.exists():
        prepared = json.loads(prepared_path.read_text())
        if prepared['dataset'] == str(Path(args.dataset).resolve()) and prepared['limit'] == args.limit:
            if prepared.get('seed',42) == args.seed and prepared['manifest'] == manifest:
                manifest = prepared['manifest']
                prepared_used = True
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'split.json').write_text(json.dumps(manifest, indent=2))
    settings = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    settings.update(python_executable=sys.executable, prepared_manifest_used=prepared_used,
        evaluation_scope='same training samples - sanity check only' if args.limit else 'held-out test',
        seed=args.seed, label_ids=[0,1,2], tensorflow=tf.__version__,
        devices=[str(x) for x in tf.config.list_physical_devices()])
    (args.output / 'config.json').write_text(json.dumps(settings, indent=2))
    return images, masks, manifest, settings


def train_model(args, train, val):
    """Build model.py's U-Net and save the best checkpoint and training history."""
    model = unet(input_size=(args.size,args.size,3), num_classes=3, base_filters=args.base_filters,
                 learning_rate=args.learning_rate, dropout_rate=args.dropout_rate,
                 depth=args.depth, loss_mode=args.loss_mode, dice_weight=args.dice_weight)
    print('Dataset split:', {'train_batches': len(train), 'validation_batches': len(val)}, flush=True)
    print('Model variable device:', model.variables[0].handle.device, flush=True)
    print('Parameters:', model.count_params(), 'Hardware:', [str(x) for x in tf.config.list_physical_devices()], flush=True)
    started = time.perf_counter()
    history = model.fit(train, validation_data=val, epochs=args.epochs, verbose=2, shuffle=False, callbacks=[
        tf.keras.callbacks.ModelCheckpoint(str(args.output / 'best.keras'), monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-5),
        tf.keras.callbacks.CSVLogger(str(args.output / 'training.csv'))])
    (args.output / 'history.json').write_text(json.dumps(history.history, indent=2))
    history.training_seconds = time.perf_counter() - started
    history.parameter_count = model.count_params()
    return model, history


def evaluate_model(args, test, manifest, settings, split_name='test'):
    """Dice is sklearn pixel F1, calculated on real argmax predictions."""
    model = tf.keras.models.load_model(args.output / 'best.keras')
    metrics = model.evaluate(test, return_dict=True, verbose=2)
    actuals, predictions = [], []
    pred_dir = args.output / 'predictions'
    pred_dir.mkdir()
    saved = 0
    for x,y in test:
        pred = np.argmax(model(x, training=False).numpy(), axis=-1)
        actual = np.argmax(y.numpy(), axis=-1) if y.shape.rank == 4 and y.shape[-1] == 3 else y.numpy()
        actuals.append(actual.ravel())
        predictions.append(pred.ravel())
        for j in range(len(pred)):
            if saved < 6:
                save_prediction(pred_dir, manifest[split_name][saved], pred[j])
                saved += 1
    actual = np.concatenate(actuals)
    pred = np.concatenate(predictions)
    dice = f1_score(actual, pred, labels=[0,1,2], average=None, zero_division=0)
    confusion = confusion_matrix(actual, pred, labels=[0,1,2])
    metrics.update(dice_by_class_id=dice.tolist(),
        macro_dice=float(f1_score(actual,pred,labels=[0,1,2],average='macro',zero_division=0)),
        foreground_dice=float(f1_score(actual,pred,labels=[1,2],average='macro',zero_division=0)),
        class_names=args.class_names, evaluation_scope=settings['evaluation_scope'] if args.limit else split_name,
        predicted_pixels_by_class_id=confusion.sum(0).tolist(), confusion_matrix=confusion.tolist())
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print('Evaluation metrics:',json.dumps(metrics),flush=True)
    print('Artifacts:',args.output.resolve(),flush=True)
    return model, metrics, pred_dir


def run_training(args, evaluation_split='test'):
    images, masks, manifest, settings = prepare_run(args)
    datasets = {name: make_dataset(images, masks, ids, size=args.size,
        batch_size=args.batch_size, training=name == 'train', seed=args.seed,
        augmentation=args.augmentation and name == 'train', one_hot=args.loss_mode != 'sparse')
        for name,ids in manifest.items()}
    _, history = train_model(args, datasets['train'], datasets['validation'])
    model, metrics, pred_dir = evaluate_model(args, datasets[evaluation_split], manifest, settings, evaluation_split)
    metrics.update(parameter_count=history.parameter_count, training_seconds=history.training_seconds,
        depth=args.depth, loss_mode=args.loss_mode, augmentation=args.augmentation,
        epochs_completed=len(history.history['loss']))
    (args.output / 'metrics.json').write_text(json.dumps(metrics,indent=2))
    fig = plot_predictions(images,masks,manifest[evaluation_split],pred_dir,args.output)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return metrics


def run_experiments(base_args, depths=(2,3,4,5), depth_tolerance=0.005):
    """Select depth with validation Dice, then freeze a 2x2 loss/augmentation comparison."""
    root = Path(base_args.output)
    root.mkdir(parents=True,exist_ok=False)
    rows = []
    def run(label, depth, loss, augmentation, split):
        args = deepcopy(base_args)
        args.depth, args.loss_mode, args.augmentation = depth, loss, augmentation
        args.output = root / label
        print('Experiment:',label,flush=True)
        result = run_training(args,evaluation_split=split)
        row = {'experiment':label,'split':split,**result}
        rows.append(row)
        (root / 'experiment_results.json').write_text(json.dumps(rows,indent=2))
        fields = ['experiment','split','depth','loss_mode','augmentation','macro_dice',
                  'foreground_dice','accuracy','parameter_count','training_seconds']
        with (root / 'experiment_results.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
            writer.writeheader(); writer.writerows(rows)
        return result
    depth_results = {depth:run(f'depth_{depth}',depth,'cce',False,'validation') for depth in depths}
    best = max(r['macro_dice'] for r in depth_results.values())
    selected = min(d for d,r in depth_results.items() if r['macro_dice'] >= best-depth_tolerance)
    (root / 'depth_selection.json').write_text(json.dumps({
        'selected_depth':selected,'tolerance':depth_tolerance,'best_validation_macro_dice':best,
        'criterion':'smallest depth within tolerance of best validation macro Dice; test not used'},indent=2))
    ablations = {}
    for loss in ('cce','cce_dice'):
        for aug in (False,True):
            ablations[(loss,aug)] = run(f'{loss}_aug_{int(aug)}',selected,loss,aug,'test')
    differences = {
        'dice_loss_without_augmentation':ablations[('cce_dice',False)]['macro_dice']-ablations[('cce',False)]['macro_dice'],
        'dice_loss_with_augmentation':ablations[('cce_dice',True)]['macro_dice']-ablations[('cce',True)]['macro_dice'],
        'augmentation_cce':ablations[('cce',True)]['macro_dice']-ablations[('cce',False)]['macro_dice'],
        'augmentation_cce_dice':ablations[('cce_dice',True)]['macro_dice']-ablations[('cce_dice',False)]['macro_dice']}
    (root / 'performance_differences.json').write_text(json.dumps(differences,indent=2))
    lines = ['# Measured U-Net experiments','',
        'Same training samples only; smoke/sanity results are not held-out evidence.' if base_args.limit is not None else
        '80/20 random development/test split; validation is 10% of development. Test is not used to select depth.',
        '',f'Selected encoder/decoder levels: {selected} (validation tolerance {depth_tolerance}).','',
        '| Run | Split | Levels | Loss | Augmentation | Macro Dice | Foreground Dice |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        lines.append(f"| {row['experiment']} | {row['split']} | {row['depth']} | {row['loss_mode']} | {row['augmentation']} | {row['macro_dice']:.4f} | {row['foreground_dice']:.4f} |")
    lines += ['','## Measured macro Dice differences','']
    lines += [f'- {key}: {value:+.4f}' for key,value in differences.items()]
    lines += ['','One fixed-seed run per configuration; differences are not statistical significance claims.',
              'Interpret the results and write your own discussion after inspecting the predictions.']
    (root / 'experiment_report.md').write_text('\n'.join(lines))
    print('Measured comparisons saved to',root.resolve(),flush=True)
    return rows, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('data/936'))
    parser.add_argument('--limit', type=int, default=5, help='Five-image sanity run by default')
    parser.add_argument('--all-images', action='store_true', help='Use all pairs with a held-out test split')
    parser.add_argument('--require-gpu', action='store_true', help='Fail instead of silently running on CPU')
    parser.add_argument('--suite', action='store_true', help='Depth study and 2x2 loss/augmentation comparison')
    parser.add_argument('--depth',type=int,default=4)
    parser.add_argument('--depths',type=int,nargs='+',default=[2,3,4,5])
    parser.add_argument('--loss-mode',choices=['cce','cce_dice','sparse'],default='cce')
    parser.add_argument('--dice-weight',type=float,default=1.0)
    parser.add_argument('--augmentation',action='store_true')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--class-names',nargs=3,default=['class_0','class_1','class_2'])
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
    if args.suite:
        run_experiments(args,depths=args.depths)
    else:
        run_training(args)


if __name__ == '__main__':
    main()
