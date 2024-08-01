
import tensorflow as tf
import kerastuner as kt
import tensorflow_transform as tft
from kerastuner import Hyperband
from tfx.components.tuner.component import TunerFnResult
from tensorflow.keras import layers

LABEL_KEY = "humor"
FEATURE_KEY = "text"

def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, 
             tf_transform_output,
             num_epochs,
             batch_size=64)->tf.data.Dataset:
    """Get post_tranform feature & create batches of data"""
    
    # Get post_transform feature spec
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())
    
    # create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key = transformed_name(LABEL_KEY))
    return dataset

# Vocabulary size and number of words in a sequence.
VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 100
 
vectorize_layer = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH)
 
embedding_dim=16

def model_builder(hp):
    """Build machine learning model"""

    # Define the hyperparameters to be tuned
    hp_embedding_dim = hp.Int('embedding_dim', min_value=32, max_value=128, step=32)
    hp_dense_1_units = hp.Int('dense_1_units', min_value=32, max_value=128, step=32)
    hp_dense_2_units = hp.Int('dense_2_units', min_value=16, max_value=64, step=16)
    hp_learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
    
    inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)
    reshaped_narrative = tf.reshape(inputs, [-1])
    x = vectorize_layer(reshaped_narrative)
    x = layers.Embedding(VOCAB_SIZE, hp_embedding_dim, name="embedding")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(hp_dense_1_units, activation='relu')(x)
    x = layers.Dense(hp_dense_2_units, activation="relu")(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs = outputs)
    
    model.compile(
        loss = 'binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(hp_learning_rate),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )
    
    # print(model)
    model.summary()
    
    return model 

def tuner_fn(fn_args):
    """
    Build the tuner using the KerasTuner API.
    Args:
    fn_args: Holds args used to tune models as name/value pairs.
    
    Returns:
    A namedtuple contains the following:
    - tuner: A BaseTuner that will be used for tuning.
    - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                  model , e.g., the training and validation dataset. Required
                  args depend on the above tuner's implementation.
    """
    # Load training and validation dataset that have been preprocessed
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    train_set = input_fn(fn_args.train_files[0], tf_transform_output, 10)
    val_set = input_fn(fn_args.eval_files[0], tf_transform_output, 10)
    vectorize_layer.adapt(
        [j[0].numpy()[0] for j in [
            i[0][transformed_name(FEATURE_KEY)]
                for i in list(train_set)]])
    
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    
    # Define hyperparameter tuning strategy
    tuner = kt.RandomSearch(
        model_builder,
        objective='val_binary_accuracy',
        max_trials=10,  # Specify the number of trials you want to run
        directory=fn_args.working_dir,
        project_name='kt_randomsearch'
    )
    
    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={ 
            "callbacks": [stop_early],
            'x': train_set,
            'validation_data': val_set,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
        }
    )
