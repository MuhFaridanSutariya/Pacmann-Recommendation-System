import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class RecommenderNet(tf.keras.Model):

    def __init__(self, num_users, num_place, num_age_categories, num_encoded_locations, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_place = num_place
        self.embedding_size = embedding_size

        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.user_bias = layers.Embedding(num_users, 1)

        self.place_embedding = layers.Embedding(
            num_place,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.place_bias = layers.Embedding(num_place, 1)

        self.age_embedding = layers.Embedding(
            num_age_categories,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )

        self.location_embedding = layers.Embedding(
            num_encoded_locations,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )

        self.flatten = layers.Flatten()  # Flatten layer to convert embeddings to a 1D vector
        self.concat = layers.Concatenate(axis=1)  # Concatenate layer to combine all features

        self.dense1 = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])

        place_vector = self.place_embedding(inputs[:, 1])
        place_bias = self.place_bias(inputs[:, 1])

        age_vector = self.age_embedding(inputs[:, 2])

        location_vector = self.location_embedding(inputs[:, 3])

        combined_features = self.concat([user_vector, place_vector, age_vector, location_vector])

        x = self.flatten(combined_features)
        x = self.dense1(x)
        x = self.output_layer(x)

        return x