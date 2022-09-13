""" trainer for neural machnine translation """
import numpy as np
import tensorflow as tf
#import efficientnet.tfkeras as efn


def get_angles(pos, index , d_model):
    """ computes angles given position """
    angle_rates = 1 / np.power(10000, (2 * (index // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    """ computes positional encoding given position and model dimension """
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def scaled_dot_product_attention(queries, keys, values, mask):
    """Calculate the attention weights.
    queries, keys, values must have matching leading dimensions.
    keys, values must have matching penultimate dimension, i.e.: seq_len_keys = seq_len_values.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      queries: query shape == (..., seq_len_q, depth)
      keys: key shape == (..., seq_len_k, depth)
      values: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(queries, keys, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dkeys = tf.cast(tf.shape(keys)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dkeys)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += mask * -1e9

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1
    )  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, values)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """ Implements multidead attention"""
    def __init__(self, d_model, num_heads):
        """ intialises multiheaded attention class"""
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wqueries = tf.keras.layers.Dense(d_model)
        self.wkeys = tf.keras.layers.Dense(d_model)
        self.wvalues = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, input_x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        input_x = tf.reshape(input_x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(input_x, perm=[0, 2, 1, 3])

    def call(self, values, keys, queries, mask):
        """ forward pass of the network"""
        batch_size = tf.shape(queries)[0]

        queries = self.wqueries(queries)  # (batch_size, seq_len, d_model)
        keys = self.wkeys(keys)  # (batch_size, seq_len, d_model)
        values = self.wvalues(values)  # (batch_size, seq_len, d_model)

        queries = self.split_heads(queries, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        keys = self.split_heads(keys, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        values = self.split_heads(values, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            queries, keys, values, mask
        )

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model)
        )  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    " pointwise feed forward network"
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(dff, activation="relu"),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model),  # (batch_size, seq_len, d_model)
        ]
    )


class EncoderLayer(tf.keras.layers.Layer):
    """ class for the Encoder Layer"""
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """ Implements the Encoder Layer"""
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, input_x, training, mask):
        """ forward pass of the Encoder Layer"""
        # (batch_size, input_seq_len, d_model)
        attn_output, _ = self.mha(input_x, input_x, input_x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(input_x + attn_output)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(
            out1 + ffn_output
        )  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    """ class for the Deooder Layer"""
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """ initialises the Decoder Layer"""
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, input_x, enc_output, training, look_ahead_mask, padding_mask):
        """ forward pass for the Decoder Layer"""
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(
            input_x, input_x, input_x, look_ahead_mask
        )  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + input_x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask
        )  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(
            ffn_output + out2
        )  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    """ class for the Encoder network"""
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        maximum_position_encoding,
        rate=0.1,
    ):
        """ initialises the Encoder"""
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, input_x, training, mask):
        """ forward pass of the Encoder"""

        seq_len = tf.shape(input_x)[1]

        # adding embedding and position encoding.
        input_x = self.embedding(input_x)  # (batch_size, input_seq_len, d_model)
        input_x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        input_x += self.pos_encoding[:, :seq_len, :]

        input_x = self.dropout(input_x, training=training)

        for i in range(self.num_layers):
            input_x = self.enc_layers[i](input_x, training, mask)

        return input_x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    """ class for the Decoder Network"""
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        target_vocab_size,
        maximum_position_encoding,
        rate=0.1,
    ):
        """ initialises the Decoder Network"""
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, input_x, enc_output, training, look_ahead_mask, padding_mask):
        """ forward pass of the Deooder"""

        seq_len = tf.shape(input_x)[1]
        attention_weights = {}

        input_x = self.embedding(input_x)  # (batch_size, target_seq_len, d_model)
        input_x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        input_x += self.pos_encoding[:, :seq_len, :]

        input_x = self.dropout(input_x, training=training)

        for i in range(self.num_layers):
            input_x, block1, block2 = self.dec_layers[i](
                input_x, enc_output, training, look_ahead_mask, padding_mask
            )

            attention_weights[f"decoder_layer{i+1}_block1"] = block1
            attention_weights[f"decoder_layer{i+1}_block2"] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return input_x, attention_weights


class Transformer(tf.keras.Model):
    """ class for the Transformer model"""
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        target_vocab_size,
        pe_input,
        pe_target,
        rate=0.1,
    ):
        """ initialises and defines the parameters of the Transformers"""
        super(Transformer, self).__init__()

        self.tokenizer = Encoder(
            num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate
        )

        self.decoder = Decoder(
            num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate
        )

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(
        self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask
    ):
        """ forward pass of the transformer"""

        enc_output = self.tokenizer(
            inp, training, enc_padding_mask
        )  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask
        )

        final_output = self.final_layer(
            dec_output
        )  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights
