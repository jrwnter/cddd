"""Base translation model with different variations"""
import os
import shutil
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf

class BaseModel(ABC):
    """
    This is the base class for the translation model. Child class defines encode and decode
    architecture.

    Attribures:
        mode: The mode the model is supposed to run (e.g. Train, EVAL, ENCODE, DECODE).
        iterator: The iterator of the input pipeline.
        embedding_size: The size of the bottleneck layer which is later used as molecular
        descriptor.
        encode_vocabulary: Dictonary that maps integers to unique tokens of the
        input strings.
        decode_vocabulary: Dictonary that maps integers to unique tokens of the
        output strings.
        encode_voc_size: Number of tokens in encode_vocabulary.
        decode_voc_size: Number of tokens in decode_vocabulary.
        char_embedding_size: Number of Dimensiones used to encode the one-hot encoded tokens
        in a contineous space.
        global_step: Counter for steps during training.
        save_dir: Path to directory used to save the model and logs.
        checkpoint_path: path to the model checkpoint file.
        batch_size: Number of samples per training batch.
        rand_input_swap: Flag to define if (for SMILES input) the input SMILES should be swapt
        randomly between canonical SMILES (usually output sequnce) and random shuffled SMILES
        (usually input sequnce).
        measures_to_log: Dictonary with values to log.
        emb_activation: Activation function used in the bottleneck layer.
        lr: Learning rate for training the model.
        lr_decay: Boolean to define if learning rate deacay is used.
        lr_decay_frequency: Number of steps between learning rate decay steps.
        lr_decay_factor: Amount of learning rate decay.
        beam_width: Width of the the window used for the beam search decoder.
    """

    def __init__(self, mode, iterator, hparams):
        """Constructor for base translation model class.

        Args:
            mode: The mode the model is supposed to run (e.g. Train, EVAL, ENCODE, DECODE).
            iterator: The iterator of the input pipeline.
            hparams: Hyperparameters defined in file or flags.
        Returns:
            None
        Raises:
            ValueError: if mode is not Train, EVAL, ENCODE, DECODE
            ValueError: if emb_activation is not tanh or linear
        """
        self.mode = mode
        self.iterator = iterator
        self.embedding_size = hparams.emb_size
        self.encode_vocabulary = {
            v: k for k, v in np.load(hparams.encode_vocabulary_file, allow_pickle=True).item().items()
        }
        self.encode_voc_size = len(self.encode_vocabulary)
        self.decode_vocabulary = {
            v: k for k, v in np.load(hparams.decode_vocabulary_file, allow_pickle=True).item().items()
        }
        self.decode_vocabulary_reverse = {v: k for k, v in self.decode_vocabulary.items()}
        self.decode_voc_size = len(self.decode_vocabulary)
        self.one_hot_embedding = hparams.one_hot_embedding
        self.char_embedding_size = hparams.char_embedding_size
        self.global_step = tf.get_variable('global_step',
                                           [],
                                           initializer=tf.constant_initializer(0),
                                           trainable=False)
        self.save_dir = hparams.save_dir
        self.checkpoint_path = os.path.join(self.save_dir, 'model.ckpt')
        self.batch_size = hparams.batch_size
        self.rand_input_swap = hparams.rand_input_swap
        self.measures_to_log = {}
        if hparams.emb_activation == "tanh":
            self.emb_activation = tf.nn.tanh
        elif hparams.emb_activation == "linear":
            self.emb_activation = lambda x: x
        else:
            raise ValueError("This activationfunction is not implemented...")
        if mode == "TRAIN":
            self.lr = hparams.lr
            self.lr_decay = hparams.lr_decay
            self.lr_decay_frequency = hparams.lr_decay_frequency
            self.lr_decay_factor = hparams.lr_decay_factor
        if mode == "DECODE":
            self.beam_width = hparams.beam_width
        if mode not in ["TRAIN", "EVAL", "ENCODE", "DECODE"]:
            raise ValueError("Choose one of following modes: TRAIN, EVAL, ENCODE, DECODE")

    def build_graph(self):
        """Method that defines the graph for a translation model instance."""
        if self.mode in ["TRAIN", "EVAL"]:
            with tf.name_scope("Input"):
                (self.input_seq,
                 self.shifted_target_seq,
                 self.input_len,
                 self.shifted_target_len,
                 self.target_mask,
                 encoder_emb_inp,
                 decoder_emb_inp) = self._input()

            with tf.variable_scope("Encoder"):
                encoded_seq = self._encoder(encoder_emb_inp)

            with tf.variable_scope("Decoder"):
                logits = self._decoder(encoded_seq, decoder_emb_inp)
                self.prediction = tf.argmax(logits, axis=2, output_type=tf.int32)

            with tf.name_scope("Measures"):
                self.loss = self._compute_loss(logits)
                self.accuracy = self._compute_accuracy(self.prediction)
                self.measures_to_log["loss"] = self.loss
                self.measures_to_log["accuracy"] = self.accuracy

            if self.mode == "TRAIN":
                with tf.name_scope("Training"):
                    self._training()

        if self.mode == "ENCODE":
            with tf.name_scope("Input"):
                self.input_seq = tf.placeholder(tf.int32, [None, None])
                self.input_len = tf.placeholder(tf.int32, [None])
                encoder_emb_inp = self._emb_lookup(self.input_seq)

            with tf.variable_scope("Encoder"):
                self.encoded_seq = self._encoder(encoder_emb_inp)

        if self.mode == "DECODE":
            if self.one_hot_embedding:
                self.decoder_embedding = tf.one_hot(
                    list(range(0, self.decode_voc_size)),
                    self.decode_voc_size
                )
            elif self.encode_vocabulary == self.decode_vocabulary:
                self.decoder_embedding = tf.get_variable(
                    "char_embedding",
                    [self.decode_voc_size, self.char_embedding_size]
                )
            else:
                self.decoder_embedding = tf.get_variable(
                    "char_embedding2",
                    [self.decode_voc_size, self.char_embedding_size]
                )

            with tf.name_scope("Input"):
                self.encoded_seq = tf.placeholder(tf.float32,
                                                  [None, self.embedding_size])
                self.maximum_iterations = tf.placeholder(tf.int32, [])
                self.maximum_iterations = tf.placeholder(tf.int32, [])

            with tf.variable_scope("Decoder"):
                self.output_ids = self._decoder(self.encoded_seq)

        self.saver_op = tf.train.Saver()

    def _input(self, with_features=False):
        """Method that defines input part of the graph for a translation model instance.

        Args:
            with_features: Defines if in addition to input and output sequnce futher
            molecular features e.g. logP are expected from the input pipleine iterator.
        Returns:
            input_seq: The input sequnce.
            shifted_target_seq: The target sequnce shifted by one charcater to the left.
            input_len: Number of tokens in input.
            shifted_target_len: Number of tokens in the shifted target sequence.
            target_mask: shifted target sequence with masked padding tokens.
            encoder_emb_inp: Embedded input sequnce (contineous character embedding).
            decoder_emb_inp: Embedded input sequnce (contineous character embedding).
            mol_features: if Arg with_features is set to True, the molecular features of the
            input pipleine are passed.
        """
        with tf.device('/cpu:0'):
            if with_features:
                seq1, seq2, seq1_len, seq2_len, mol_features = self.iterator.get_next()
            else:
                seq1, seq2, seq1_len, seq2_len = self.iterator.get_next()
            if self.rand_input_swap:
                rand_val = tf.random_uniform([], dtype=tf.float32)
                input_seq = tf.cond(tf.greater_equal(rand_val, 0.5),
                                    lambda: seq1, lambda: seq2)
                input_len = tf.cond(tf.greater_equal(rand_val, 0.5),
                                    lambda: seq1_len, lambda: seq2_len)
            else:
                input_seq = seq1
                input_len = seq1_len
            target_seq = seq2
            target_len = seq2_len
            shifted_target_len = tf.reshape(target_len, [tf.shape(target_len)[0]]) - 1
            shifted_target_seq = tf.slice(target_seq, [0, 1], [-1, -1])
            target_mask = tf.sequence_mask(shifted_target_len, dtype=tf.float32)
            target_mask = target_mask / tf.reduce_sum(target_mask)
            input_len = tf.reshape(input_len, [tf.shape(input_len)[0]])

        encoder_emb_inp, decoder_emb_inp = self._emb_lookup(input_seq, target_seq)
        if with_features:
            return (input_seq, shifted_target_seq, input_len, shifted_target_len,
                    target_mask, encoder_emb_inp, decoder_emb_inp, mol_features)
        else:
            return (input_seq, shifted_target_seq, input_len, shifted_target_len,
                    target_mask, encoder_emb_inp, decoder_emb_inp)

    def _emb_lookup(self, input_seq, target_seq=None):
        """Method that performs an embedding lookup to embed the one-hot encoded input
        and output sequnce into the trainable contineous character embedding.

        Args:
            input_seq: The input sequnce.
            target_seq: The target sequnce.
        Returns:
            encoder_emb_inp: Embedded input sequnce (contineous character embedding).
            decoder_emb_inp: Embedded input sequnce (contineous character embedding).
        """
        if self.one_hot_embedding:
            self.encoder_embedding = tf.one_hot(
                list(range(0, self.encode_voc_size)),
                self.encode_voc_size
            )
        else:
            self.encoder_embedding = tf.get_variable(
                "char_embedding",
                [self.encode_voc_size, self.char_embedding_size]
            )
        encoder_emb_inp = tf.nn.embedding_lookup(self.encoder_embedding, input_seq)
        if self.mode != "ENCODE":
            assert target_seq is not None
            if self.encode_vocabulary == self.decode_vocabulary:
                self.decoder_embedding = self.encoder_embedding
            elif self.one_hot_embedding:
                self.decoder_embedding = tf.one_hot(
                    list(range(0, self.decode_voc_size)),
                    self.decode_voc_size
                )
            else:
                self.decoder_embedding = tf.get_variable(
                    "char_embedding2",
                    [self.decode_voc_size, self.char_embedding_size]
                )
            decoder_emb_inp = tf.nn.embedding_lookup(self.decoder_embedding, target_seq)
            return encoder_emb_inp, decoder_emb_inp
        else:
            return encoder_emb_inp

    def _training(self):
        """Method that defines the training opertaion of the training model's graph."""

        if self.lr_decay:
            self.lr = tf.train.exponential_decay(self.lr,
                                                 self.global_step,
                                                 self.lr_decay_frequency,
                                                 self.lr_decay_factor,
                                                 staircase=True,)
        self.opt = tf.train.AdamOptimizer(self.lr, name='optimizer')
        grads = self.opt.compute_gradients(self.loss)
        grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]
        self.train_step = self.opt.apply_gradients(grads, self.global_step)

    @abstractmethod
    def _encoder(self, encoder_emb_inp):
        """Method that defines the encoder part of the translation model graph."""
        raise NotImplementedError("Must override _encoder in child class")

    @abstractmethod
    def _decoder(self, encoded_seq, decoder_emb_inp=None):
        """Method that defines the decoder part of the translation model graph."""
        raise NotImplementedError("Must override _decoder in child class")

    def _compute_loss(self, logits):
        """Method that calculates the loss function."""
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.shifted_target_seq,
            logits=logits)
        loss = (tf.reduce_sum(crossent * self.target_mask))
        return loss

    def _compute_accuracy(self, prediction):
        """Method that calculates the character-wise translation accuracy."""
        right_predictions = tf.cast(tf.equal(prediction, self.shifted_target_seq), tf.float32)
        accuracy = (tf.reduce_sum(right_predictions * self.target_mask))
        return accuracy

    def train(self, sess):
        """Method that can be called to perform a training step.

        Args:
            sess: The Session the model is running in.
        Returns:
            step: The global step.
        """
        assert self.mode == "TRAIN"
        _, step = sess.run([self.train_step, self.global_step])
        return step

    def eval(self, sess):
        """Method that can be called to perform a evaluation step.

        Args:
            sess: The Session the model is running in.
        Returns:
            step: The loged measures.
        """
        return sess.run(list(self.measures_to_log.values()))

    def idx_to_char(self, seq):
        """Helper function to transform the one-hot encoded sequnce tensor back to string-sequence.

        Args:
            seq: sequnce of one-hot encoded characters.
        Returns:
            string sequnce.
        """
        return ''.join([self.decode_vocabulary_reverse[idx] for idx in seq
                        if idx not in [-1, self.decode_vocabulary["</s>"],
                                       self.decode_vocabulary["<s>"]]])

    def seq2emb(self, sess, input_seq, input_len):
        """Method to run a forwards path up to the bottneck layer (ENCODER).
        Encodes a one-hot encoded input sequnce.

        Args:
            sess: The Session the model is running in.
            input_seq: sequnces of one-hot encoded characters.
            input_len: number of characters per sequnce.
        Returns:
            Embedding of the input sequnces.
        """
        assert self.mode == "ENCODE"
        return sess.run(self.encoded_seq, {self.input_seq: input_seq,
                                           self.input_len: input_len})
    def emb2seq(self, sess, embedding, num_top, maximum_iterations=1000):
        """Method to run a forwards path from bottlneck layer to output sequnce (DECODER).
        Decodes the embedding (molecular descriptor) back to a sequnce representaion.

        Args:
            sess: The Session the model is running in.
            embedding: Embeddings (molecular descriptors) of the input sequnces.
            num_top: Number of most probable sequnces as output of the beam search decoder
        Returns:
            Embedding of the input sequnces.
        """
        assert self.mode == "DECODE"
        output_seq = sess.run(self.output_ids, {self.encoded_seq: embedding,
                                                self.maximum_iterations: maximum_iterations})
        return [[self.idx_to_char(seq[:, i]) for i in range(num_top)] for seq in output_seq]

    def initilize(self, sess, overwrite_saves=False):
        """Function to initialize variables in the model graph and creation of save folder.

        Args:
            sess: The Session the model is running in.
            overwrite_saves: Defines whether to overwrite the files (recreate directory) if a folder
            with same save file path exists.
        Returns:
            step: Initial value of global step.
        """
        assert self.mode == "TRAIN"
        sess.run(tf.global_variables_initializer())
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print('Create save file in: ', self.save_dir)
        elif overwrite_saves:
            shutil.rmtree(self.save_dir)
            os.makedirs(self.save_dir)
        else:
            raise ValueError("Save directory %s already exist." %(self.save_dir))
        return sess.run(self.global_step)

    def restore(self, sess, restore_path=None):
        """ Helper Function to restore the variables in the model graph."""

        if restore_path is None:
            restore_path = self.checkpoint_path
        self.saver_op.restore(sess, restore_path)
        if self.mode == "TRAIN":
            step = sess.run(self.global_step)
            print("Restarting training at step %d" %(step))
            return step

    def save(self, sess):
        """Wrapper function save model to file."""
        self.saver_op.save(sess, self.checkpoint_path)

class GRUSeq2Seq(BaseModel):
    """Translation model class with a multi-layer Recurrent Neural Network as Encoder
    and Decoder with Gate Recurrent Units (GRUs). Encoder and Decoder architecutre are
    the same.

    Attribures:
        cell_size: list defining the number of Units in each GRU cell.
        reverse_decoding: whether to invert the cell_size list for the Decoder.
    """
    def __init__(self, mode, iterator, hparams):
        """Constructor for the GRU translation model class.

        Args:
            mode: The mode the model is supposed to run (e.g. Train, EVAL, ENCODE, DECODE).
            iterator: The iterator of the input pipeline.
            hparams: Hyperparameters defined in file or flags.
        Returns:
            None
        Raises:
            ValueError: if mode is not Train, EVAL, ENCODE, DECODE
            ValueError: if emb_activation is not tanh or linear
        """
        super().__init__(mode, iterator, hparams)
        self.cell_size = hparams.cell_size
        self.reverse_decoding = hparams.reverse_decoding

    def _encoder(self, encoder_emb_inp):
        """Method that defines the encoder part of the translation model graph."""
        encoder_cell = [tf.nn.rnn_cell.GRUCell(size) for size in self.cell_size]
        encoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_cell)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                           encoder_emb_inp,
                                                           sequence_length=self.input_len,
                                                           dtype=tf.float32,
                                                           time_major=False)
        emb = tf.layers.dense(tf.concat(encoder_state, axis=1),
                              self.embedding_size,
                              activation=self.emb_activation
                             )
        return emb

    def _decoder(self, encoded_seq, decoder_emb_inp=None):
        """Method that defines the decoder part of the translation model graph."""
        if self.reverse_decoding:
            self.cell_size = self.cell_size[::-1]
        decoder_cell = [tf.nn.rnn_cell.GRUCell(size) for size in self.cell_size]
        decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cell)
        decoder_cell_inital = tf.layers.dense(encoded_seq, sum(self.cell_size))
        decoder_cell_inital = tuple(tf.split(decoder_cell_inital, self.cell_size, 1))
        projection_layer = tf.layers.Dense(self.decode_voc_size, use_bias=False)
        if self.mode != "DECODE":
            helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp,
                                                       sequence_length=self.shifted_target_len,
                                                       time_major=False)
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                      helper,
                                                      decoder_cell_inital,
                                                      output_layer=projection_layer)
            outputs, output_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                         impute_finished=True,
                                                                         output_time_major=False)
            return outputs.rnn_output
        else:
            decoder_cell_inital = tf.contrib.seq2seq.tile_batch(decoder_cell_inital,
                                                                self.beam_width)
            start_tokens = tf.fill([tf.shape(encoded_seq)[0]], self.decode_vocabulary['<s>'])
            end_token = self.decode_vocabulary['</s>']
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=decoder_cell,
                embedding=self.decoder_embedding,
                start_tokens=start_tokens,
                end_token=end_token,
                initial_state=decoder_cell_inital,
                beam_width=self.beam_width,
                output_layer=projection_layer,
                length_penalty_weight=0.0)

            outputs, output_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                impute_finished=False,
                output_time_major=False,
                maximum_iterations=self.maximum_iterations
            )

            return outputs.predicted_ids
        
class GRUVAE(GRUSeq2Seq):
    def __init__(self, mode, iterator, hparams):
        super().__init__(mode, iterator, hparams)
        self.div_loss_scale = hparams.div_loss_scale
        self.div_loss_rate = hparams.div_loss_rate
        
    def _encoder(self, encoder_emb_inp):

        """Method that defines the encoder part of the translation model graph."""
        encoder_cell = [tf.nn.rnn_cell.GRUCell(size) for size in self.cell_size]
        encoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_cell)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                           encoder_emb_inp,
                                                           sequence_length=self.input_len,
                                                           dtype=tf.float32,
                                                           time_major=False)
        loc = tf.layers.dense(tf.concat(encoder_state, axis=1),
                              self.embedding_size
                             )
        log_scale = tf.layers.dense(tf.concat(encoder_state, axis=1),
                                self.embedding_size
                               )
        return loc, log_scale
    
    def _sampler(self, loc, log_scale):
        epsilon = tf.random_normal(
            shape=[tf.shape(loc)[0], self.embedding_size],
            mean=0,
            stddev=1
        )
        return loc + tf.exp(log_scale) * epsilon
    
    def _compute_loss(self, logits, loc, log_scale):
        """Method that calculates the loss function."""
        
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.shifted_target_seq,
            logits=logits)
        crossent = tf.reduce_sum(crossent * self.target_mask, axis=1)
        
        divergence = -0.5 * tf.reduce_sum(1 + 2*log_scale - tf.square(loc) - tf.square(tf.exp(log_scale)), axis=-1)
    
        self.measures_to_log["crossent"] = tf.reduce_mean(crossent)
        self.measures_to_log["divergence"] = tf.reduce_mean(divergence)
        
        div_loss_scale = self.div_loss_scale - tf.train.exponential_decay(self.div_loss_scale,
                                                 self.global_step,
                                                 10000,
                                                 self.div_loss_rate,
                                                 staircase=True,)
        
        self.measures_to_log["div_loss_scale"] = div_loss_scale
        return tf.reduce_mean(crossent + div_loss_scale * divergence)
    
    def build_graph(self):
        """Method that defines the graph for a translation model instance."""
        if self.mode in ["TRAIN", "EVAL"]:
            with tf.name_scope("Input"):
                (self.input_seq,
                 self.shifted_target_seq,
                 self.input_len,
                 self.shifted_target_len,
                 self.target_mask,
                 encoder_emb_inp,
                 decoder_emb_inp) = self._input()

            with tf.variable_scope("Encoder"):
                loc, log_scale = self._encoder(encoder_emb_inp)
                encoded_seq = self._sampler(loc, log_scale)

            with tf.variable_scope("Decoder"):
                logits = self._decoder(encoded_seq, decoder_emb_inp)
                self.prediction = tf.argmax(logits, axis=2, output_type=tf.int32)

            with tf.name_scope("Measures"):
                #rossent, divergence, self.loss = self._compute_loss(logits, posterior)
                self.loss = self._compute_loss(logits, loc, log_scale)
                self.accuracy = self._compute_accuracy(self.prediction)
                self.measures_to_log["loss"] = self.loss
                self.measures_to_log["accuracy"] = self.accuracy

            if self.mode == "TRAIN":
                with tf.name_scope("Training"):
                    self._training()

        if self.mode == "ENCODE":
            with tf.name_scope("Input"):
                self.input_seq = tf.placeholder(tf.int32, [None, None])
                self.input_len = tf.placeholder(tf.int32, [None])
                encoder_emb_inp = self._emb_lookup(self.input_seq)

            with tf.variable_scope("Encoder"):
                loc, log_scale = self._encoder(encoder_emb_inp)
                self.encoded_seq = self._sampler(loc, log_scale)

        if self.mode == "DECODE":
            if self.one_hot_embedding:
                self.decoder_embedding = tf.one_hot(
                    list(range(0, self.decode_voc_size)),
                    self.decode_voc_size
                )
            elif self.encode_vocabulary == self.decode_vocabulary:
                self.decoder_embedding = tf.get_variable(
                    "char_embedding",
                    [self.decode_voc_size, self.char_embedding_size]
                )
            else:
                self.decoder_embedding = tf.get_variable(
                    "char_embedding2",
                    [self.decode_voc_size, self.char_embedding_size]
                )

            with tf.name_scope("Input"):
                self.encoded_seq = tf.placeholder(tf.float32,
                                                  [None, self.embedding_size])

            with tf.variable_scope("Decoder"):
                self.output_ids = self._decoder(self.encoded_seq)

        self.saver_op = tf.train.Saver()

class NoisyGRUSeq2Seq(GRUSeq2Seq):
    """Translation model class with a multi-layer Recurrent Neural Network as Encoder and
    Decoder with Gate Recurrent Units (GRUs) with input dropout and a Gaussian Noise term
    after the bottlneck layer. Encoder and Decoder architecutre are the same.

    Attribures:
        input_dropout: Dropout rate of a Dropout layer after the character embedding of the
        input sequnce.
        emb_noise: Standard deviation of the Gaussian Noise term after the bottlneck layer.
    """

    def __init__(self, mode, iterator, hparams):
        """Constructor for the Noisy GRU translation model class.

        Args:
            mode: The mode the model is supposed to run (e.g. Train, EVAL, ENCODE, DECODE).
            iterator: The iterator of the input pipeline.
            hparams: Hyperparameters defined in file or flags.
        Returns:
            None
        Raises:
            ValueError: if mode is not Train, EVAL, ENCODE, DECODE
            ValueError: if emb_activation is not tanh or linear
        """
        super().__init__(mode, iterator, hparams)
        self.input_dropout = hparams.input_dropout
        self.emb_noise = hparams.emb_noise

    def _encoder(self, encoder_emb_inp):
        """Method that defines the encoder part of the translation model graph."""
        if (self.mode == "TRAIN") & (self.input_dropout > 0.0):
            max_time = tf.shape(encoder_emb_inp)[1]
            encoder_emb_inp = tf.nn.dropout(encoder_emb_inp,
                                            1. - self.input_dropout,
                                            noise_shape=[self.batch_size, max_time, 1])
        encoder_cell = [tf.nn.rnn_cell.GRUCell(size) for size in self.cell_size]
        encoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_cell)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                           encoder_emb_inp,
                                                           sequence_length=self.input_len,
                                                           dtype=tf.float32,
                                                           time_major=False)
        emb = tf.layers.dense(tf.concat(encoder_state, axis=1),
                              self.embedding_size
                             )
        if (self.mode == "TRAIN") & (self.emb_noise > 0.0):
            emb += tf.random_normal(shape=tf.shape(emb),
                                    mean=0.0,
                                    stddev=self.emb_noise,
                                    dtype=tf.float32)
        emb = self.emb_activation(emb)
        return emb

class LSTMSeq2Seq(BaseModel):
    """Translation model class with a multi-layer Recurrent Neural Network as Encoder
    and Decoder with Long short-term memory units (LSTM). Encoder and Decoder architecutre
    are the same.

    Attribures:
        cell_size: list defining the number of Units in each GRU cell.
    """
    def __init__(self, mode, iterator, hparams):
        """Constructor for the LSTM translation model class.

        Args:
            mode: The mode the model is supposed to run (e.g. Train, EVAL, ENCODE, DECODE).
            iterator: The iterator of the input pipeline.
            hparams: Hyperparameters defined in file or flags.
        Returns:
            None
        Raises:
            ValueError: if mode is not Train, EVAL, ENCODE, DECODE
            ValueError: if emb_activation is not tanh or linear
        """
        super().__init__(mode, iterator, hparams)
        self.cell_size = hparams.cell_size

    def _encoder(self, encoder_emb_inp):
        """Method that defines the encoder part of the translation model graph."""
        encoder_cell = [tf.nn.rnn_cell.LSTMCell(size) for size in self.cell_size]
        encoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_cell)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                           encoder_emb_inp,
                                                           sequence_length=self.input_len,
                                                           dtype=tf.float32,
                                                           time_major=False)
        encoder_state_c = [state.c for state in encoder_state]
        emb = tf.layers.dense(tf.concat(encoder_state_c, axis=1),
                              self.embedding_size,
                              activation=self.emb_activation
                             )
        return emb

    def _decoder(self, encoded_seq, decoder_emb_inp=None):
        """Method that defines the decoder part of the translation model graph."""
        decoder_cell = [tf.nn.rnn_cell.LSTMCell(size) for size in self.cell_size]
        decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cell)
        initial_state_c_full = tf.layers.dense(encoded_seq, sum(self.cell_size))
        initial_state_c = tuple(tf.split(initial_state_c_full, self.cell_size, 1))
        initial_state_h_full = tf.zeros_like(initial_state_c_full)
        initial_state_h = tuple(tf.split(initial_state_h_full, self.cell_size, 1))
        decoder_cell_inital = tuple(
            [tf.contrib.rnn.LSTMStateTuple(
                initial_state_c[i],
                initial_state_h[i]) for i in range(len(self.cell_size))
            ]
        )
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp,
                                                   sequence_length=self.shifted_target_len,
                                                   time_major=False)
        projection_layer = tf.layers.Dense(self.decode_voc_size, use_bias=False)
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                  helper,
                                                  decoder_cell_inital,
                                                  output_layer=projection_layer)
        outputs, output_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                     impute_finished=True,
                                                                     output_time_major=False)
        return outputs.rnn_output

class Conv2GRUSeq2Seq(GRUSeq2Seq):
    """Translation model class with a multi-layer 1-D Convolutional Neural Network as Encoder.
    The Decoder is still a RNN with GRU cells.

    Attribures:
        conv_hidden_size: List defining the number of filters in each layer.
        kernel_size: List defining the width of the 1-D conv-filters in each layer.
    """
    def __init__(self, mode, iterator, hparams):
        """Constructor for the Convolutional translation model class.

        Args:
            mode: The mode the model is supposed to run (e.g. Train, EVAL, ENCODE, DECODE).
            iterator: The iterator of the input pipeline.
            hparams: Hyperparameters defined in file or flags.
        Returns:
            None
        Raises:
            ValueError: if mode is not Train, EVAL, ENCODE, DECODE
            ValueError: if emb_activation is not tanh or linear
        """
        super().__init__(mode, iterator, hparams)
        self.conv_hidden_size = hparams.conv_hidden_size
        self.kernel_size = hparams.kernel_size

    def _encoder(self, encoder_emb_inp):
        """Method that defines the encoder part of the translation model graph."""
        for i, size in enumerate(self.conv_hidden_size):
            x = tf.layers.conv1d(encoder_emb_inp,
                                 size,
                                 self.kernel_size[i],
                                 activation=tf.nn.relu,
                                 padding='SAME')
            if i+1 < len(self.conv_hidden_size):
                x = tf.layers.max_pooling1d(x, 3, 2, padding='SAME')

        x = tf.layers.conv1d(x,
                             self.conv_hidden_size[-1],
                             1,
                             activation=tf.nn.relu,
                             padding='SAME')

        emb = tf.layers.dense(tf.reduce_mean(x, axis=1),
                              self.embedding_size,
                              activation=self.emb_activation
                             )
        return emb

class GRUSeq2SeqWithFeatures(GRUSeq2Seq):
    """Translation model class with a multi-layer Recurrent Neural Network as Encoder
    and Decoder with  Gate Recurrent Units (GRUs) with an additional feature classification
    task. Encoder and Decoder architecutre are the same.

    Attribures:
        num_features: Number of features to prediced.
    """
    def __init__(self, mode, iterator, hparams):
        """Constructor for the GRU translation model with feature classification class.

        Args:
            mode: The mode the model is supposed to run (e.g. Train, EVAL, ENCODE, DECODE).
            iterator: The iterator of the input pipeline.
            hparams: Hyperparameters defined in file or flags.
        Returns:
            None
        Raises:
            ValueError: if mode is not Train, EVAL, ENCODE, DECODE
            ValueError: if emb_activation is not tanh or linear
        """
        super().__init__(mode, iterator, hparams)
        self.num_features = hparams.num_features

    def build_graph(self):
        """Method that defines the graph for a translation model instance with the additional
        feature prediction task.
        """
        if self.mode in ["TRAIN", "EVAL"]:
            with tf.name_scope("Input"):
                (self.input_seq,
                 self.shifted_target_seq,
                 self.input_len,
                 self.shifted_target_len,
                 self.target_mask,
                 encoder_emb_inp,
                 decoder_emb_inp,
                 self.mol_features) = self._input(with_features=True)

            with tf.variable_scope("Encoder"):
                encoded_seq = self._encoder(encoder_emb_inp)

            with tf.variable_scope("Decoder"):
                sequence_logits = self._decoder(encoded_seq, decoder_emb_inp)
                self.sequence_prediction = tf.argmax(sequence_logits,
                                                     axis=2,
                                                     output_type=tf.int32)

            with tf.variable_scope("Feature_Regression"):
                feature_predictions = self._feature_regression(encoded_seq)

            with tf.name_scope("Measures"):
                self.loss_sequence, self.loss_features = self._compute_loss(sequence_logits,
                                                                            feature_predictions)
                self.loss = self.loss_sequence + self.loss_features
                self.accuracy = self._compute_accuracy(self.sequence_prediction)
                self.measures_to_log["loss"] = self.loss
                self.measures_to_log["accuracy"] = self.accuracy

            if self.mode == "TRAIN":
                with tf.name_scope("Training"):
                    self._training()

        if self.mode == "ENCODE":
            with tf.name_scope("Input"):
                self.input_seq = tf.placeholder(tf.int32, [None, None])
                self.input_len = tf.placeholder(tf.int32, [None])
                encoder_emb_inp = self._emb_lookup(self.input_seq)

            with tf.variable_scope("Encoder"):
                self.encoded_seq = self._encoder(encoder_emb_inp)

        if self.mode == "DECODE":
            if self.one_hot_embedding:
                self.decoder_embedding = tf.one_hot(
                    list(range(0, self.decode_voc_size)),
                    self.decode_voc_size
                )
            elif self.encode_vocabulary == self.decode_vocabulary:
                self.decoder_embedding = tf.get_variable(
                    "char_embedding",
                    [self.decode_voc_size, self.char_embedding_size]
                )
            else:
                self.decoder_embedding = tf.get_variable(
                    "char_embedding2",
                    [self.decode_voc_size, self.char_embedding_size]
                )

            with tf.name_scope("Input"):
                self.encoded_seq = tf.placeholder(tf.float32, [None, self.embedding_size])
                self.maximum_iterations = tf.placeholder(tf.int32, [])
            with tf.variable_scope("Decoder"):
                self.output_ids = self._decoder(self.encoded_seq)
        self.saver_op = tf.train.Saver()

    def _feature_regression(self, encoded_seq):
        """Method that defines the feature classification part of the graph."""
        x = tf.layers.dense(inputs=encoded_seq,
                            units=512,
                            activation=tf.nn.relu
                            )
        x = tf.layers.dense(inputs=x,
                            units=128,
                            activation=tf.nn.relu
                            )
        x = tf.layers.dense(inputs=x,
                            units=self.num_features,
                            activation=None
                            )

        return x

    def _compute_loss(self, sequence_logits, features_predictions):
        """Method that calculates the loss function."""
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.shifted_target_seq,
                                                                  logits=sequence_logits)
        loss_sequence = (tf.reduce_sum(crossent * self.target_mask))
        loss_features = tf.losses.mean_squared_error(labels=self.mol_features,
                                                     predictions=features_predictions,
                                                    )
        return loss_sequence, loss_features

class NoisyGRUSeq2SeqWithFeatures(GRUSeq2SeqWithFeatures):
    """Translation model class with a multi-layer Recurrent Neural Network as Encoder and Decoder
    with Gate Recurrent Units (GRUs) with input dropout and a Gaussian Noise Term after the
    bottlneck layer and an additional feature classification task. Encoder and Decoder architecutre
    are the same.

    Attribures:
        input_dropout: Dropout rate of a Dropout layer after the character embedding of the input
        sequnce.
        emb_noise: Standard deviation of the Gaussian Noise term after the bottlneck layer.
    """
    def __init__(self, mode, iterator, hparams):
        """Constructor for the Noisy GRU translation model with feature vlassification class.

        Args:
            mode: The mode the model is supposed to run (e.g. Train, EVAL, ENCODE, DECODE).
            iterator: The iterator of the input pipeline.
            hparams: Hyperparameters defined in file or flags.
        Returns:
            None
        Raises:
            ValueError: if mode is not Train, EVAL, ENCODE, DECODE
            ValueError: if emb_activation is not tanh or linear
        """
        super().__init__(mode, iterator, hparams)
        self.input_dropout = hparams.input_dropout
        self.emb_noise = hparams.emb_noise

    def _encoder(self, encoder_emb_inp):
        """Method that defines the encoder part of the translation model graph."""
        if self.mode == "TRAIN":
            max_time = tf.shape(encoder_emb_inp)[1]
            encoder_emb_inp = tf.nn.dropout(encoder_emb_inp,
                                            1. - self.input_dropout,
                                            noise_shape=[self.batch_size, max_time, 1])
        encoder_cell = [tf.nn.rnn_cell.GRUCell(size) for size in self.cell_size]
        encoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_cell)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                           encoder_emb_inp,
                                                           sequence_length=self.input_len,
                                                           dtype=tf.float32,
                                                           time_major=False)
        emb = tf.layers.dense(tf.concat(encoder_state, axis=1),
                              self.embedding_size
                             )
        if (self.emb_noise >= 0) & (self.mode == "TRAIN"):
            emb += tf.random_normal(shape=tf.shape(emb),
                                    mean=0.0,
                                    stddev=self.emb_noise,
                                    dtype=tf.float32)
        emb = self.emb_activation(emb)
        return emb

class ModelWithGrads(NoisyGRUSeq2SeqWithFeatures):
    def __init__(self, mode, iterator, hparams):
        super().__init__(mode, iterator, hparams)

    def build_graph(self):
        with tf.name_scope("Input"):
            self.input_seq = tf.placeholder(tf.int32, [None, None])
            self.input_len = tf.placeholder(tf.int32, [None])
            self.start_grads = tf.placeholder(tf.float32, [None, ndims])
            encoder_emb_inp = self._emb_lookup(self.input_seq)

        with tf.variable_scope("Encoder"):
            self.encoded_seq = self._encoder(encoder_emb_inp)
        self.grads = tf.gradients(self.encoded_seq, encoder_emb_inp, self.start_grads)

        self.saver_op = tf.train.Saver()
