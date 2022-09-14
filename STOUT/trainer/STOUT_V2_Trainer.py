""" training module for STOUT """
#pylint: disable=wrong-import-position
import re
import pickle
from datetime import datetime
import time
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("Agg")
#import os
import tensorflow as tf
import nmt_model_transformer
#pylint: enable=wrong-import-position






tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="node-2")
print("Running on TPU ", tpu.master())

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.TPUStrategy(tpu)
print(f"Number of devices: {strategy.num_replicas_in_sync}")


f = open("Training_Report.txt", "w",encoding = "utf-8")
sys.stdout = f

numbers = re.compile(r"(\d+)")


def numerical_sort(value):
    """ performs numerical sort """
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

with open("tokenizer_input.pkl", "rb") as file:
    inp_lang = pickle.load(file)
with open("tokenizer_target.pkl", "rb") as file:
    targ_lang = pickle.load(file)
with open("max_length_inp.pkl", "rb") as file:
    inp_max_length = pickle.load(file)
with open("max_length_targ.pkl", "rb") as file:
    targ_max_length = pickle.load(file)

print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), "Data loaded", flush=True)

TOTAL_DATA = 1000000

EPOCHS = 50
NUM_LAYER = 4
D_MODEL = 512
DFF = 2048
NUM_HEADS = 8
DROPOUT_RATE = 0.1
BUFFER_SIZE = 10000
BATCH_SIZE = 160 * strategy.num_replicas_in_sync
steps_per_epoch = TOTAL_DATA // BATCH_SIZE
num_steps = TOTAL_DATA // BATCH_SIZE

input_vocab_size = len(inp_lang.word_index) + 1
target_vocab_size = len(targ_lang.word_index) + 1

AUTO = tf.data.experimental.AUTOTUNE


def read_tfrecord(example):
    """ reads tensorflow record for a given example  and decodes the result for a given example """
    feature = {
        #'image_id': tf.io.FixedLenFeature([], tf.string),
        "input_selfies": tf.io.FixedLenFeature([], tf.string),
        "target_iupac": tf.io.FixedLenFeature([], tf.string),
    }

    # decode the TFRecord
    example = tf.io.parse_single_example(example, feature)

    input_selfies = tf.io.decode_raw(example["input_selfies"], tf.int32)
    target_iupac = tf.io.decode_raw(example["target_iupac"], tf.int32)

    return input_selfies, target_iupac


def get_training_dataset(batch_size=BATCH_SIZE, buffered_size=BUFFER_SIZE):
    """ loads the training size and initialises the dataloader with given
        buffer size and batch size """
    options = tf.data.Options()
    filenames = sorted(
        tf.io.gfile.glob(
            "gs://tpu-test-koh/STOUT_V2/STOUT_V2_development/1mio_SMI_character/TF_rec/*.tfrecord"
        ),
        key=numerical_sort,
    )

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    # dataset_txt = tf.data.Dataset.from_tensor_slices((cap_train))

    train_dataset = (
        dataset.with_options(options)
        .map(read_tfrecord, num_parallel_calls=AUTO)
        .shuffle(buffered_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=AUTO)
    )
    return train_dataset


def get_validation_dataset(batch_size=BATCH_SIZE, buffered_size=BUFFER_SIZE):
    """ loads the validation size and initialises the dataloader with
    given buffer size and batch size """


    options = tf.data.Options()
    filenames = sorted(
        tf.io.gfile.glob(
            "gs://tpu-test-koh/STOUT_V2/STOUT_V2_development/1mio_SMI_character/TF_rec/*.tfrecord"
        ),
        key=numerical_sort,
    )

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    # dataset_txt = tf.data.Dataset.from_tensor_slices((cap_train))

    validation_dataset = (
        dataset.with_options(options)
        .map(read_tfrecord, num_parallel_calls=AUTO)
        .shuffle(buffered_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=AUTO)
    )
    return validation_dataset


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    "Class to perform learning rate scheduling "
    def __init__(self, d_model, warmup_steps=5000):
        """ initialises the learning rate scheduler class"""
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """ functionality for scheduling idea"""
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def create_look_ahead_mask(size):
    " creates a look ahead mask of a given size "
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_padding_mask(seq):
    """ creates a padding mask of given size """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_masks(inp, tar):
    """ creates encoding and decoding padding mask """
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


with strategy.scope():
    # encoder = nmt_model.Encoder(vocab_inp_size, embedding_dim, units)
    # decoder = nmt_model.Decoder(vocab_tar_size, embedding_dim, units)

    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999,
    #                                       epsilon=1e-07, amsgrad=False)
    learning_rate = CustomSchedule(D_MODEL)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )

    def loss_function(real, pred):
        """ computes the loss given labels and prediction """
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def accuracy_function(real, pred):
        """ computes accuracy given labels and prediction"""
        accuracies = tf.math.equal(
            real, tf.cast(tf.argmax(pred, axis=2), dtype=tf.int32)
        )

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name="train_loss", dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name="train_accuracy", dtype=tf.float32
    )
    validation_loss = tf.keras.metrics.Mean(name="validation_loss", dtype=tf.float32)
    validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name="validation_accuracy", dtype=tf.float32
    )

    # Initialize Transformer
    transformer = nmt_model_transformer.Transformer(
        NUM_LAYER,
        D_MODEL,
        NUM_HEADS,
        DFF,
        input_vocab_size,
        target_vocab_size,
        inp_max_length,
        targ_max_length,
        rate=DROPOUT_RATE,
    )


CHECKPOINT_PATH = (
    "gs://tpu-test-koh/STOUT_V2/STOUT_V2_development/1mio_SMI_character/checkpoints/"
)
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=150)

START_EPOCH = 0
if ckpt_manager.latest_checkpoint:
    ckpt.restore(tf.train.latest_checkpoint(CHECKPOINT_PATH))
    START_EPOCH = int(ckpt_manager.latest_checkpoint.split("-")[-1])

per_replica_batch_size = BATCH_SIZE // strategy.num_replicas_in_sync

print("Batch Size", BATCH_SIZE)
print("Per replica", per_replica_batch_size)
train_dataset = strategy.experimental_distribute_dataset(get_training_dataset())
validation_dataset = strategy.experimental_distribute_dataset(get_validation_dataset())
# the loss_plot array will be reset many times
loss_plot = []
train_accuracy_data = []
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


@tf.function
def train_step(iterator):
    """ runs each traing step"""
    def step_fn(inputs):
        """ maining functionality of a forward pass and subsequent backpropagation"""
        inp, target = inputs
        loss = 0

        tar_inp = target[:, :-1]
        tar_real = target[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = transformer(
                inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask
            )
            loss = loss_function(tar_real, predictions)
            train_accuracy.update_state(
                tar_real,
                predictions,
                sample_weight=tf.where(tf.not_equal(tar_real, PAD_TOKEN), 1.0, 0.0),
            )

        gradients = tape.gradient(loss, transformer.trainable_variables)

        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss.update_state(loss * strategy.num_replicas_in_sync)

    strategy.run(step_fn, args=(iterator,))


@tf.function
def validation_step(iterator):
    """ performs validation functionality"""
    def step_fn(inputs):
        """ performs the forward pass and the subsequent backprop for validation"""
        img_tensor, target = inputs

        tar_inp = target[:, :-1]
        tar_real = target[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = transformer(
                inp, tar_inp, False, enc_padding_mask, combined_mask, dec_padding_mask
            )
            loss = loss_function(tar_real, predictions)
            validation_accuracy.update_state(
                tar_real,
                predictions,
                sample_weight=tf.where(tf.not_equal(tar_real, PAD_TOKEN), 1.0, 0.0),
            )

        validation_loss.update_state(loss * strategy.num_replicas_in_sync)

    strategy.run(step_fn, args=(iterator,))


print(
    datetime.now().strftime("%Y/%m/%d %H:%M:%S"), "Actual Training Started", flush=True
)

loss_plot = []
accuracy_plot = []
val_loss_plot = []
val_acc = []

for epoch in range(START_EPOCH, EPOCHS):
    start = time.time()
    batch = 0
    validation_batch = 0
    train_loss.reset_states()
    train_accuracy.reset_states()

    for x in train_dataset:
        img_tensor, target = x
        train_step(x)
        batch += 1

        if batch % 100 == 0:
            print(
                f"Epoch {epoch + 1} Batch {batch} Loss : {train_loss.result()} \
                Accuracy {train_accuracy.result()}",
                flush=True,
            )

        if batch == num_steps:
            loss_plot.append(train_loss.result().numpy())
            accuracy_plot.append(train_accuracy.result().numpy())
            ckpt_manager.save()

            print(
                f"Epoch {epoch + 1} Training_Loss : {train_loss.result()} \
                Accuracy  : {train_accuracy.result()}",
                flush=True,
            )
            print(
                datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
                f"Time taken for 1 epoch {time.time() -start} sec\n",
                flush=True,
            )
            break

    for y in validation_dataset:
        start_val = time.time()
        img_tensor, target = y
        validation_step(y)
        validation_batch += 1

        if validation_batch == validation_steps:

            print(
                f"Validation_Loss : { validation_loss.result()} \
                Accuracy : {validation_accuracy.result()}",
                flush=True,
            )
            print(
                datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
                f"Time taken for validation {time.time() - start_val} sec\n",
                flush=True,
            )
            val_loss_plot.append(validation_loss.result().numpy())
            val_acc.append(validation_accuracy.result().numpy())
            break

    train_loss.reset_states()
    train_accuracy.reset_states()
    validation_loss.reset_states()
    validation_accuracy.reset_states()

plt.plot(loss_plot, "-o", label="Training loss")
plt.plot(val_loss_plot, "-o", label="Validation loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(ncol=2, loc="upper right")
# plt.show()
plt.gcf().set_size_inches(20, 20)
plt.savefig("Lossplot_SMILES.jpg")
plt.close()

plt.plot(accuracy_plot, "-o", label="Training accuracy")
plt.plot(val_acc, "-o", label="Validation accuracy")
plt.title("Training and Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(ncol=2, loc="lower right")
# plt.show()
plt.gcf().set_size_inches(20, 20)
plt.savefig("accuracyplot_SMILES.jpg")
plt.close()

print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), "Network Completed", flush=True)
f.close()
