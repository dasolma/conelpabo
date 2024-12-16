import tensorflow as tf

class DistanceLayer(tf.keras.layers.Layer):
    """
    """

    def __init__(self, distance_type='euclidean', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.distance_type = distance_type

    def call(self, q, s):

        if self.distance_type == 'l1':
            distance = tf.reduce_sum(tf.math.abs(q - s), axis=-1)
            return distance
        elif self.distance_type == 'l2':
            distance = tf.reduce_sum(tf.square(q - s), axis=-1)
            return distance
        elif self.distance_type == 'cosine':
            normalize_a = tf.math.l2_normalize(q, -1)
            normalize_b = tf.math.l2_normalize(s, -1)
            cos_similarity = -tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=-1)

            return cos_similarity
        elif self.distance_type == 'product':
            return -tf.reduce_sum(tf.multiply(q, s), axis=-1)

    def get_config(self):
        config = {
            "distance_type": self.distance_type,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))