import tensorflow as tf
import tensorflow_io as tfio
import pandas as pd

from biobuzz.yamnet.model import YamnetModel


@tf.function
def load_wav_16k_mono(filename: str):
    """Load audio file as a tensor and resample it to 16kHz single channel audio."""
    file_content = tf.io.read_file(filename)
    audio, sample_rate = tf.audio.decode_wav(file_content, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    return tfio.audio.resample(audio, rate_in=sample_rate, rate_out=16000)


def load_wav_for_map(filename, label, fold):
    return load_wav_16k_mono(filename), label, fold


def split_dataset(filenames: pd.Series, targets: pd.Series, folds: pd.Series, yamnet_model: YamnetModel):
    main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))
    main_ds = main_ds.map(load_wav_for_map)
    main_ds = main_ds.map(yamnet_model.extract_embedding).unbatch()
   
    cached_ds = main_ds.cache() 
    train_ds = cached_ds.filter(lambda embedding, label, fold: fold == "train")
    val_ds = cached_ds.filter(lambda embedding, label, fold: fold == "val")

    remove_fold_column = lambda embedding, label, fold: (embedding, label)
    train_ds = train_ds.map(remove_fold_column)
    val_ds = val_ds.map(remove_fold_column)

    train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds
