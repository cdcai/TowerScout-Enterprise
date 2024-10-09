import unittest
from webapp.ts_azmaps import AzureMap
from webapp.tests.config import API_KEY

class TestAzureMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api_key = API_KEY
        cls.azure_map = AzureMap(cls.api_key)

    def test_get_url(self):
        tile = {
            'lng': 123.456,
            'lat_for_url': 78.9
        }
        url = self.azure_map.get_url(tile)
        self.assertIsNotNone(url)
        # Add more assertions to validate the URL format and content

    def test_get_meta_url(self):
        tile = {
            'lng': 123.456,
            'lat': 78.9
        }
        url = self.azure_map.get_meta_url(tile)
        self.assertIsNotNone(url)
        # Add more assertions to validate the URL format and content

    def test_checkCutOffs(self):
        object = {
            'x1': 0.1,
            'x2': 0.8,
            'y2': 0.98
        }
        confidence = self.azure_map.checkCutOffs(object)
        self.assertEqual(confidence, 0.1)
        
    def test_get_date(self):
      md = '{"resourceSets": [{"resources": [{"vintageStart": "2022-01-01"}]}]}'
      date = self.azure_map.get_date(md)
      self.assertEqual(date, "2022-01-01")

if __name__ == '__main__':
    unittest.main()