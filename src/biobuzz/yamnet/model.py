import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub


class YamnetModel:
    def __init__(self) -> None:
        self.model_handle = "https://tfhub.dev/google/yamnet/1"

    def load(self) -> None:
        # Download the model from Tensorflow Hub
        self.model = hub.load(self.model_handle)
        # Load classes mapping
        class_map_path = self.model.class_map_path().numpy().decode("utf-8")
        self.classes = pd.read_csv(class_map_path)["display_name"].tolist()

    def extract_embedding(self, wav_data, label, fold):
        _, embeddings, _ = self.model(wav_data)
        num_embeddings = tf.shape(embeddings)[0]
        return (
            embeddings,
            tf.repeat(label, num_embeddings),
            tf.repeat(fold, num_embeddings),
        )

    def predict(self, audio_data: tf.Tensor) -> str:
        """For now, audio must have been already preprocessed."""
        scores, _, _ = self.model(audio_data)
        class_scores = tf.reduce_mean(scores, axis=0)
        top_class = tf.math.argmax(class_scores)
        return self.classes[top_class]
