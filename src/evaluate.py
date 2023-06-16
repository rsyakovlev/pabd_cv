import tensorflow as tf
import os

model = tf.keras.models.load_model('models/dummy_model')

def classify_binary(img_path: str) -> int:
    img = tf.keras.utils.load_img(img_path, target_size=(180, 180))
    # img = tf.io.decode_jpeg(data)
    img_t = tf.expand_dims(img, axis=0)

    predictions = model.predict(img_t)
    dog_probability = float(predictions[0])
    print(dog_probability)
    idx = dog_probability > 0.5
    return int(idx)

if __name__ == '__main__':
    cat_imgs = os.listdir('data/raw/pinterest/Cat')
    y_pred = [
        classify_binary(
            os.path.join('data/raw/pinterest/Cat', img_path)) for img_path in cat_imgs
    ]
    print(y_pred)