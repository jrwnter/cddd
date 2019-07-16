"""Functions that build the data input pipeline for the translation model."""
import re
import numpy as np
import tensorflow as tf

REGEX_SML = r'Cl|Br|[#%\)\(\+\-1032547698:=@CBFIHONPS\[\]cionps]'
REGEX_INCHI = r'Br|Cl|[\(\)\+,-/123456789CFHINOPSchpq]'

class InputPipeline():
    """Base input pipeline class. Iterates through tf-record file to produce inputs
    for training the translation model.

    Atributes:
        mode: The mode the model is supposed to run (e.g. Train).
        batch_size: Number of samples per batch.
        buffer_size: Number of samples in the shuffle buffer.
        input_sequence_key: Identifier of the input_sequence feature in the
        tf-record file.
        output_sequence_key: Identifier of the output_sequence feature in the
        tf-record file.
        encode_vocabulary: Dictonary that maps integers to unique tokens of the
        input strings.
        decode_vocabulary: Dictonary that maps integers to unique tokens of the
        output strings.
        num_buckets: Number of buckets for batching together sequnces of
        similar length.
        min_bucket_lenght: All sequnces below this legth are put in the
        same bucket.
        max_bucket_lenght: All sequnces above this legth are put in the
        same bucket.
        regex_pattern_input: Expression to toeknize the input sequnce with.
        regex_pattern_output: Expression to toeknize the output sequnce with.
    """

    def __init__(self, mode, hparams):
        """Constructor for base input pipeline class.

        Args:
            mode: The mode the model is supposed to run (e.g. Train).
            hparams: Hyperparameters defined in file or flags.
        Returns:
            None
        """
        self.mode = mode
        self.batch_size = hparams.batch_size
        self.buffer_size = hparams.buffer_size
        self.input_sequence_key = hparams.input_sequence_key
        self.output_sequence_key = hparams.output_sequence_key
        if self.mode == "TRAIN":
            self.file = hparams.train_file
        else:
            self.input_sequence_key = "canonical_smiles"
            self.file = hparams.val_file
        self.encode_vocabulary = {
            v: k for k, v in np.load(hparams.encode_vocabulary_file, allow_pickle=True).item().items()
        }
        self.decode_vocabulary = {
            v: k for k, v in np.load(hparams.decode_vocabulary_file, allow_pickle=True).item().items()
        }
        self.num_buckets = hparams.num_buckets
        self.min_bucket_lenght = hparams.min_bucket_length
        self.max_bucket_lenght = hparams.max_bucket_length
        if "inchi" in self.input_sequence_key:
            self.regex_pattern_input = REGEX_INCHI
        elif "smiles" in self.input_sequence_key:
            self.regex_pattern_input = REGEX_SML
        else:
            raise ValueError("Could not understand the input typ. SMILES or INCHI?")
        if "inchi" in self.output_sequence_key:
            self.regex_pattern_output = REGEX_INCHI
        elif "smiles" in self.output_sequence_key:
            self.regex_pattern_output = REGEX_SML
        else:
            raise ValueError("Could not understand the output typ. SMILES or INCHI?")

    def make_dataset_and_iterator(self):
        """Method that builds a TFRecordDataset and creates a iterator."""
        self.dataset = tf.data.TFRecordDataset(self.file)
        if self.mode == "TRAIN":
            self.dataset = self.dataset.repeat()
        self.dataset = self.dataset.map(self._parse_element, num_parallel_calls=32)
        self.dataset = self.dataset.map(
            lambda element: tf.py_func(self._process_element,
                                       [element[self.input_sequence_key],
                                        element[self.output_sequence_key]],
                                       [tf.int32, tf.int32, tf.int32, tf.int32]),
            num_parallel_calls=32)
        self.dataset = self.dataset.apply(tf.contrib.data.group_by_window(
            key_func=lambda in_seq, out_seq, in_len, out_len: self._length_bucket(in_len),
            reduce_func=lambda key, ds: self._pad_batch(
                ds,
                self.batch_size,
                ([None], [None], [1], [1]),
                (self.encode_vocabulary["</s>"], self.decode_vocabulary["</s>"], 0, 0)
            ),
            window_size=self.batch_size
        ))
        if self.mode == "TRAIN":
            self.dataset = self.dataset.shuffle(buffer_size=self.buffer_size)
        self.iterator = self.dataset.make_initializable_iterator()

    def _parse_element(self, example_proto):
        """Method that parses an element from a tf-record file."""
        feature_dict = {self.input_sequence_key: tf.FixedLenFeature([], tf.string),
                        self.output_sequence_key: tf.FixedLenFeature([], tf.string),
                        }
        parsed_features = tf.parse_single_example(example_proto, feature_dict)
        element = {name: parsed_features[name] for name in list(feature_dict.keys())}
        return element

    def _process_element(self, input_seq, output_seq):
        """Method that tokenizes input an output sequnce, pads it with start and stop token.

        Args:
            input_seq: Input sequnce.
            output_seq: Target sequnce.
        Returns
            Array with ids of each token in the tokenzized input sequence.
            Array with ids of each token in the tokenzized output sequence.
            Array with length of the input sequnce.
            Array with length of output sequence.
        """
        input_seq = input_seq.decode("ascii")
        output_seq = output_seq.decode("ascii")
        input_seq = np.array(self._char_to_idx(input_seq,
                                               self.regex_pattern_input,
                                               self.encode_vocabulary)
                            ).astype(np.int32)
        output_seq = np.array(self._char_to_idx(output_seq,
                                                self.regex_pattern_output,
                                                self.decode_vocabulary)
                             ).astype(np.int32)
        input_seq = self._pad_start_end_token(input_seq, self.encode_vocabulary)
        output_seq = self._pad_start_end_token(output_seq, self.decode_vocabulary)
        input_seq_len = np.array([len(input_seq)]).astype(np.int32)
        output_seq_len = np.array([len(output_seq)]).astype(np.int32)
        return input_seq, output_seq, input_seq_len, output_seq_len

    def _char_to_idx(self, seq, regex_pattern, vocabulary):
        """Helper function to tokenize a sequnce.

        Args:
            seq: Sequence to tokenize.
            regex_pattern: Expression to toeknize the input sequnce with.
            vocabulary: Dictonary that maps integers to unique tokens.
        Returns:
            List with ids of the tokens in the tokenized sequnce.
        """
        char_list = re.findall(regex_pattern, seq)
        return [vocabulary[char_list[j]] for j in range(len(char_list))]

    def _pad_start_end_token(self, seq, vocabulary):
        """Helper function to pad start and stop token to a tokenized sequnce.

        Args:
            seq: Tokenized sequnce to pad.
            vocabulary: Dictonary that maps integers to unique tokens.
        Returns:
            Array with ids of each token in the tokenzized input sequence
            padded by start and stop token.
        """
        seq = np.concatenate([np.array([vocabulary['<s>']]),
                              seq,
                              np.array([vocabulary['</s>']])
                             ]).astype(np.int32)
        return seq

    def _length_bucket(self, length):
        """Helper function to assign the a bucked for certain sequnce length.

        Args:
            length: The length of a sequnce.
        Returns:
            ID of the assigned bucket.
        """
        length = tf.cast(length, tf.float32)
        num_buckets = tf.cast(self.num_buckets, tf.float32)
        cast_value = (self.max_bucket_lenght - self.min_bucket_lenght) / num_buckets
        minimum = self.min_bucket_lenght / cast_value
        bucket_id = length / cast_value - minimum + 1
        bucket_id = tf.cast(tf.clip_by_value(bucket_id, 0, self.num_buckets + 1), tf.int64)

        return bucket_id

    def _pad_batch(self, ds, batch_size, padded_shapes, padded_values):
        """Helper function that pads a batch."""
        return ds.padded_batch(
            batch_size,
            padded_shapes=padded_shapes,
            padding_values=padded_values,
            drop_remainder=True
        )

class InputPipelineWithFeatures(InputPipeline):
    """Input pipeline class with addtional molecular feature output. Iterates through tf-record
    file to produce inputs for training the translation model.

    Atributes:
        mode: The mode the model is supposed to run (e.g. Train).
        batch_size: Number of samples per batch.
        buffer_size: Number of samples in the shuffle buffer.
        input_sequence_key: Identifier of the input_sequence feature in the
        tf-record file.
        output_sequence_key: Identifier of the output_sequence feature in the
        tf-record file.
        encode_vocabulary: Dictonary that maps integers to unique tokens of the
        input strings.
        decode_vocabulary: Dictonary that maps integers to unique tokens of the
        output strings.
        num_buckets: Number of buckets for batching together sequnces of
        similar length.
        min_bucket_lenght: All sequnces below this legth are put in the
        same bucket.
        max_bucket_lenght: All sequnces above this legth are put in the
        same bucket.
        regex_pattern_input: Expression to toeknize the input sequnce with.
        regex_pattern_output: Expression to toeknize the output sequnce with.
    """

    def __init__(self, mode, hparams):
        """Constructor for input pipeline class with features.

        Args:
            mode: The mode the model is supposed to run (e.g. Train).
            hparams: Hyperparameters defined in file or flags.
        Returns:
            None
        """
        super().__init__(mode, hparams)
        self.features_key = "mol_features"
        self.num_features = hparams.num_features

    def make_dataset_and_iterator(self):
        """Method that builds a TFRecordDataset and creates a iterator."""
        self.dataset = tf.data.TFRecordDataset(self.file)
        self.dataset = self.dataset.map(self._parse_element, num_parallel_calls=32)
        if self.mode == "TRAIN":
            self.dataset = self.dataset.repeat()
        self.dataset = self.dataset.map(
            lambda element: tf.py_func(
                self._process_element,
                [element[self.input_sequence_key],
                 element[self.output_sequence_key],
                 element[self.features_key]],
                [tf.int32, tf.int32, tf.int32, tf.int32, tf.float32]),
            num_parallel_calls=32)
        self.dataset = self.dataset.apply(tf.contrib.data.group_by_window(
            key_func=lambda in_seq, out_seq, in_len, out_len, feat: self._length_bucket(in_len),
            reduce_func=lambda key, ds: self._pad_batch(
                ds,
                self.batch_size,
                ([None], [None], [1], [1], [self.num_features]),
                (self.encode_vocabulary["</s>"], self.decode_vocabulary["</s>"], 0, 0, 0.0)
            ),
            window_size=self.batch_size))
        if self.mode == "TRAIN":
            self.dataset = self.dataset.shuffle(buffer_size=self.buffer_size)
        self.iterator = self.dataset.make_initializable_iterator()

    def _parse_element(self, example_proto):
        """Method that parses an element from a tf-record file."""
        feature_dict = {self.input_sequence_key: tf.FixedLenFeature([], tf.string),
                        self.output_sequence_key: tf.FixedLenFeature([], tf.string),
                        self.features_key: tf.FixedLenFeature([self.num_features], tf.float32)
                        }
        parsed_features = tf.parse_single_example(example_proto, feature_dict)
        element = {name: parsed_features[name] for name in list(feature_dict.keys())}
        return element

    def _process_element(self, input_seq, output_seq, features):
        """Method that tokenizes input an output sequnce, pads it with start and stop token.

        Args:
            input_seq: Input sequnce.
            output_seq: target sequnce.
        Returns
            Array with ids of each token in the tokenzized input sequence.
            Array with ids of each token in the tokenzized output sequence.
            Array with length of the input sequnce.
            Array with length of output sequence.
            Array with molecular features.
        """
        input_seq = input_seq.decode("ascii")
        output_seq = output_seq.decode("ascii")
        input_seq = np.array(self._char_to_idx(input_seq,
                                               self.regex_pattern_input,
                                               self.encode_vocabulary)
                            ).astype(np.int32)
        output_seq = np.array(self._char_to_idx(output_seq,
                                                self.regex_pattern_output,
                                                self.decode_vocabulary)
                             ).astype(np.int32)
        input_seq = self._pad_start_end_token(input_seq, self.encode_vocabulary)
        output_seq = self._pad_start_end_token(output_seq, self.decode_vocabulary)
        input_seq_len = np.array([len(input_seq)]).astype(np.int32)
        output_seq_len = np.array([len(output_seq)]).astype(np.int32)
        return input_seq, output_seq, input_seq_len, output_seq_len, features

class InputPipelineInferEncode():
    """Class that creates a python generator for list of sequnces. Used to feed
    sequnces to the encoing part during inference time.

    Atributes:
        seq_list: List with sequnces to iterate over.
        batch_size: Number of samples to output per iterator call.
        encode_vocabulary: Dictonary that maps integers to unique tokens of the
        input strings.
        input_sequence_key: Identifier of the input_sequence feature in the
        tf-record file.
        regex_pattern_input: Expression to toeknize the input sequnce with.
    """

    def __init__(self, seq_list, hparams):
        """Constructor for the inference input pipeline class.

        Args:
            seq_list: List with sequnces to iterate over.
            hparams: Hyperparameters defined in file or flags.
        Returns:
            None
        """
        self.seq_list = seq_list
        self.batch_size = hparams.batch_size
        self.encode_vocabulary = {
            v: k for k, v in np.load(hparams.encode_vocabulary_file, allow_pickle=True).item().items()
        }
        self.input_sequence_key = hparams.input_sequence_key
        if "inchi" in self.input_sequence_key:
            self.regex_pattern_input = REGEX_INCHI
        elif "smiles" in self.input_sequence_key:
            self.regex_pattern_input = REGEX_SML
        else:
            raise ValueError("Could not understand the input typ. SMILES or INCHI?")

    def _input_generator(self):
        """Function that defines the generator."""
        l = len(self.seq_list)
        for ndx in range(0, l, self.batch_size):
            samples = self.seq_list[ndx:min(ndx + self.batch_size, l)]
            samples = [self._seq_to_idx(seq) for seq in samples]
            seq_len_batch = np.array([len(entry) for entry in samples])
            # pad sequences to max len and concatenate to one array
            max_length = seq_len_batch.max() #pro
            seq_batch = np.concatenate(
                [np.expand_dims(
                    np.append(
                        seq,
                        np.array([self.encode_vocabulary['</s>']]*(max_length - len(seq)))
                    ),
                    0
                )
                 for seq in samples]
            ).astype(np.int32)
            yield seq_batch, seq_len_batch

    def initilize(self):
        """Helper function to initialiize the generator"""
        self.generator = self._input_generator()

    def get_next(self):
        """Helper function to get the next batch from the iterator"""
        return next(self.generator)

    def _char_to_idx(self, seq):
        """Helper function to tokenize a sequnce.

        Args:
            seq: Sequence to tokenize.
        Returns:
            List with ids of the tokens in the tokenized sequnce.
        """
        char_list = re.findall(self.regex_pattern_input, seq)
        return [self.encode_vocabulary[char_list[j]] for j in range(len(char_list))]

    def _seq_to_idx(self, seq):
        """Method that tokenizes a sequnce and pads it with start and stop token.

        Args:
            seq: Sequence to tokenize.
        Returns:
            seq: List with ids of the tokens in the tokenized sequnce.
        """
        seq = np.concatenate([np.array([self.encode_vocabulary['<s>']]),
                              np.array(self._char_to_idx(seq)).astype(np.int32),
                              np.array([self.encode_vocabulary['</s>']])
                             ]).astype(np.int32)
        return seq

class InputPipelineInferDecode():
    """Class that creates a python generator for arrays of embeddings (molecular descriptor).
    Used to feed embeddings to the decoding part during inference time.

    Atributes:
        embedding: Array with embeddings (molecular descriptors) (n_samples x n_features).
        batch_size: Number of samples to output per iterator call.
    """

    def __init__(self, embedding, hparams):
        """Constructor for the inference input pipeline class.

        Args:
            embedding: Array with embeddings (molecular descriptors) (n_samples x n_features).
            hparams: Hyperparameters defined in file or flags.
        Returns:
            None
        """
        self.embedding = embedding
        self.batch_size = hparams.batch_size

    def _input_generator(self):
        """Function that defines the generator."""
        l = len(self.embedding)
        for ndx in range(0, l, self.batch_size):
            samples = self.embedding[ndx:min(ndx + self.batch_size, l)]
            yield samples

    def initilize(self):
        """Helper function to initialiize the generator"""
        self.generator = self._input_generator()

    def get_next(self):
        """Helper function to get the next batch from the iterator"""
        return next(self.generator)
    