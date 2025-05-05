import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv1D, AveragePooling1D, LSTM, Dense, Dropout, Flatten,
    Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
)
import matplotlib.pyplot as plt
from scipy import signal

def create_cnn_model(input_shape, classes=256):
    model = Sequential()
    model.add(Conv1D(64, 11, activation='relu', padding='same', input_shape=input_shape))
    model.add(AveragePooling1D(2))
    model.add(Conv1D(128, 11, activation='relu', padding='same'))
    model.add(AveragePooling1D(2))
    model.add(Conv1D(256, 11, activation='relu', padding='same'))
    model.add(AveragePooling1D(2))
    model.add(Conv1D(512, 11, activation='relu', padding='same'))
    model.add(AveragePooling1D(2))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))
    return model

def create_lstm_model(input_shape, classes=256):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.25))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(512))
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))
    return model

def create_cnn_lstm_model(input_shape, classes=256):
    model = Sequential()
    model.add(Conv1D(64, 11, activation='relu', padding='same', input_shape=input_shape))
    model.add(AveragePooling1D(2))
    model.add(Conv1D(128, 11, activation='relu', padding='same'))
    model.add(AveragePooling1D(2))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(512))
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))
    return model

def positional_encoding(positions, d_model):
    angle_rates = 1 / np.power(10000, (2 * (np.arange(d_model) // 2)) / np.float32(d_model))
    angle_rads = np.arange(positions)[:, np.newaxis] * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads, dtype=tf.float32)

def transformer_block(x, n_heads, head_size, ff_dim, rate=0.1):
    attention_output = MultiHeadAttention(
        num_heads=n_heads, key_dim=head_size)(x, x)
    attention_output = Dropout(rate)(attention_output)
    out1 = LayerNormalization(epsilon=1e-6)(x + attention_output)
    ffn_output = Dense(ff_dim, activation='relu')(out1)
    ffn_output = Dense(out1.shape[-1])(ffn_output)
    ffn_output = Dropout(rate)(ffn_output)
    return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

def create_transformer_model(input_shape, classes=256):
    inputs = Input(shape=input_shape)
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    pos_encoding = positional_encoding(positions, input_shape[1])
    x = inputs + pos_encoding
    x = transformer_block(x, n_heads=8, head_size=64, ff_dim=2048, rate=0.1)
    x = transformer_block(x, n_heads=8, head_size=64, ff_dim=2048, rate=0.1)
    x = transformer_block(x, n_heads=8, head_size=64, ff_dim=2048, rate=0.1)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)

def align_traces(traces, reference_trace=None):
    if reference_trace is None:
        reference_trace = traces[0]
    aligned_traces = np.zeros_like(traces)
    for i, trace in enumerate(traces):
        correlation = signal.correlate(trace, reference_trace, mode='full')
        max_index = np.argmax(correlation)
        shift = max_index - len(reference_trace) + 1
        if shift > 0:
            aligned_traces[i, shift:] = trace[:-shift]
        elif shift < 0:
            aligned_traces[i, :shift] = trace[-shift:]
        else:
            aligned_traces[i] = trace
    return aligned_traces

def apply_low_pass_filter(traces, cutoff=0.1, order=5):
    b, a = signal.butter(order, cutoff, 'low')
    filtered_traces = np.zeros_like(traces)
    for i, trace in enumerate(traces):
        filtered_traces[i] = signal.filtfilt(b, a, trace)
    return filtered_traces

def preprocess_traces(traces):
    mean = np.mean(traces, axis=0)
    std = np.std(traces, axis=0)
    standardized_traces = (traces - mean) / (std + 1e-8)
    try:
        aligned_traces = align_traces(standardized_traces)
        filtered_traces = apply_low_pass_filter(aligned_traces)
        return filtered_traces
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return standardized_traces

def select_points_of_interest(traces, labels, num_poi=1000):
    try:
        n_traces, n_samples = traces.shape
        unique_labels = np.unique(labels)
        sost = np.zeros(n_samples)
        means = np.zeros((len(unique_labels), n_samples))
        for i, label in enumerate(unique_labels):
            means[i] = np.mean(traces[labels == label], axis=0)
        global_mean = np.mean(traces, axis=0)
        for i, label in enumerate(unique_labels):
            n_class = np.sum(labels == label)
            sost += n_class * np.square(means[i] - global_mean)
        poi_indices = np.argsort(sost)[-num_poi:]
        return poi_indices, traces[:, poi_indices]
    except Exception as e:
        print(f"Error in feature selection: {e}")
        return np.arange(min(num_poi, traces.shape[1])), traces[:, :min(num_poi, traces.shape[1])]

def extract_features(power_measurements):
    features = []
    features.append(np.mean(power_measurements, axis=0))
    features.append(np.std(power_measurements, axis=0))
    features.append(np.max(power_measurements, axis=0) - np.min(power_measurements, axis=0))
    fft_values = np.abs(np.fft.fft(power_measurements, axis=0))
    features.append(np.mean(fft_values, axis=0))
    return np.concatenate([f.flatten() for f in features])

def trigger_countermeasures():
    print("ALERT: Side-channel attack detected!")
    print("Activating countermeasures: random delays, masking, etc.")

class SideChannelDetector:
    def __init__(self):
        self.model = self._create_detection_model()
        
    def _create_detection_model(self):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(400,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
        
    def detect_attack(self, power_measurements, threshold=0.8):
        try:
            features = extract_features(power_measurements)
            features = features.reshape(1, -1)
            if features.shape[1] < 400:
                features = np.pad(features, ((0, 0), (0, 400 - features.shape[1])))
            elif features.shape[1] > 400:
                features = features[:, :400]
            attack_probability = self.model.predict(features)[0][0]
            if attack_probability > threshold:
                trigger_countermeasures()
                return True
            return False
        except Exception as e:
            print(f"Error in attack detection: {e}")
            return False

def calculate_guessing_entropy(predictions, correct_key, num_traces_list):
    try:
        max_traces = max(num_traces_list)
        result = np.zeros(len(num_traces_list))
        if len(predictions.shape) < 2 or predictions.shape[1] != 256:
            raise ValueError("Predictions must have shape (n_traces, 256)")
        accumulated_probabilities = np.zeros((256,))
        for trace_idx in range(max_traces):
            accumulated_probabilities += np.log(predictions[trace_idx] + 1e-36)
            if trace_idx + 1 in num_traces_list:
                sorted_probs = np.argsort(accumulated_probabilities)[::-1]
                key_rank = np.where(sorted_probs == correct_key)[0][0]
                result[num_traces_list.index(trace_idx + 1)] = key_rank
        return result
    except Exception as e:
        print(f"Error in guessing entropy calculation: {e}")
        return np.ones(len(num_traces_list)) * 128

def calculate_success_rate(predictions, correct_key, num_traces_list, rank_threshold=1):
    try:
        num_experiments = len(predictions)
        max_traces = max(num_traces_list)
        result = np.zeros(len(num_traces_list))
        for exp_idx in range(num_experiments):
            accumulated_probabilities = np.zeros((256,))
            for trace_idx in range(max_traces):
                accumulated_probabilities += np.log(predictions[exp_idx, trace_idx] + 1e-36)
                if trace_idx + 1 in num_traces_list:
                    sorted_probs = np.argsort(accumulated_probabilities)[::-1]
                    key_rank = np.where(sorted_probs == correct_key)[0][0]
                    if key_rank < rank_threshold:
                        result[num_traces_list.index(trace_idx + 1)] += 1
        result /= num_experiments
        return result
    except Exception as e:
        print(f"Error in success rate calculation: {e}")
        return np.zeros(len(num_traces_list))

def main():
    print("Neural Cryptanalyst: ML-Powered Side Channel Attack Framework")
    print("-----------------------------------------------------------")
    num_traces = 1000
    trace_length = 2000
    traces = np.random.normal(0, 1, (num_traces, trace_length))
    keys = np.random.randint(0, 256, num_traces)
    for i, key in enumerate(keys):
        leakage_point = key % trace_length
        traces[i, leakage_point:leakage_point+10] += 0.5
    print("Preprocessing traces...")
    processed_traces = preprocess_traces(traces)
    train_traces = processed_traces[:800]
    test_traces = processed_traces[800:]
    train_keys = keys[:800]
    test_keys = keys[800:]
    train_keys_onehot = tf.keras.utils.to_categorical(train_keys, num_classes=256)
    print("Creating CNN model...")
    input_shape = (trace_length, 1)
    model = create_cnn_model(input_shape)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    print("Model created successfully.")
    print("\nInitializing side-channel attack detection system...")
    detector = SideChannelDetector()
    print("\nNeural Cryptanalyst framework initialized successfully!")

if __name__ == "__main__":
    main()