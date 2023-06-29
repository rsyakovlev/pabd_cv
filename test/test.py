import io
import unittest
from PIL import Image
# import Image
import requests
from requests import request

# http://localhost:1781/
class MyTestCase(unittest.TestCase):
    def test_home(self):
        response = requests.request('GET', 'http://127.0.0.1:5000')
        sample = response.content.decode()
        self.assertEqual(sample, 'Home page')  # add assertion here

    # def test_imagenet_classify(self):
    #     img = Image.open('../data/dog.jpg')
    #     buffer = io.BytesIO()
    #     img.save(buffer, format='JPEG')
    #
    #     with buffer as buf:
    #         buffer.seek(0)
    #         response = request('POST', 'http://192.168.88.106:5000/classify/imagenet', data=buf)
    #
    #     out = response.content.decode('utf-8')
    #     print(out)
    #     expected = 'Пембрук'
    #
    #     self.assertIn(expected, out)

    def test_binary_classify(self):
        img = Image.open('../data/cat.jpg')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')

        with buffer as buf:
            buffer.seek(0)
            response = request('POST', 'http://127.0.0.1:5000/classify/binary', data=buf)

        out = response.content.decode('utf-8')
        print(out)
        expected = 'Cat'
        self.assertEqual(expected, out)


if __name__ == '__main__':
    unittest.main()
