import tensorflow as tf
import os

model = tf.keras.models.load_model('C:/Users/221781/PycharmProjects/pabd_cv/models/my_model')

def classify_binary(img_path: str) -> int:
    img = tf.keras.utils.load_img(img_path, target_size=(180, 180))
    img_t = tf.expand_dims(img, axis=0)
    predictions = model.predict(img_t)
    dog_probability = float(predictions[0])
    print(dog_probability)
    idx = dog_probability > 0.5
    return int(idx)

def resize(img, label):
    img_r = tf.image.resize(img, (180, 180))
    return img_r, label

if __name__ == '__main__':
    data_dir = 'C:/Users/221781/PycharmProjects/pabd_cv/data/raw/pinterest'
    test_dataset = tf.keras.utils.image_dataset_from_directory(data_dir)
    test_dataset = test_dataset.map(resize)
    metrics = [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.Precision()
    ]
    model.compile(metrics=metrics)
    history = model.evaluate(test_dataset)
    loss, accuracy, precision, recall = history
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {accuracy}')
    print(f'Recall: {accuracy}')
