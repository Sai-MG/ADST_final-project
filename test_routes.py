"""
Integration tests for Flask application routes, REST API endpoints, and failure handling.
"""

import unittest
import json
from app import app

class TestAppRoutes(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_home_page(self):
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Bengaluru Road Accident Risk', res.data)

    def test_dashboard_page(self):
        res = self.client.get('/dashboard')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Analytics Dashboard', res.data)

    def test_route_prediction_page(self):
        res = self.client.get('/predict-route?origin=Majestic&destination=Whitefield')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Route Risk Prediction', res.data)

    def test_chatbot_page(self):
        res = self.client.get('/chatbot')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'AI Road Safety Chatbot', res.data)

    def test_api_blackspots(self):
        res = self.client.get('/api/blackspots')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('type'), 'FeatureCollection')
        self.assertGreaterEqual(len(data.get('features', [])), 10)

    def test_api_btp_stations(self):
        res = self.client.get('/api/btp-stations')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('type'), 'FeatureCollection')

    def test_api_weather(self):
        res = self.client.get('/api/weather')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn('temperature_c', data)
        self.assertIn('condition', data)

    def test_api_stats(self):
        res = self.client.get('/api/stats')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn('monthly_labels', data)
        self.assertIn('daily_labels', data)

    def test_api_predict_route_valid(self):
        payload = {
            'origin': 'Majestic',
            'destination': 'Whitefield'
        }
        res = self.client.post('/api/predict-route',
                               data=json.dumps(payload),
                               content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn('overall_risk', data)
        self.assertIn('ai_explanation', data)
        self.assertIn('route', data)
        self.assertGreater(data['route']['distance_km'], 10.0)

    def test_api_predict_route_empty_payload(self):
        # API failure handling
        res = self.client.post('/api/predict-route',
                               data=json.dumps({}),
                               content_type='application/json')
        self.assertEqual(res.status_code, 400)

    def test_api_chat(self):
        payload = {'message': 'What are the top blackspots in Bengaluru?'}
        res = self.client.post('/api/chat',
                               data=json.dumps(payload),
                               content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn('reply', data)
        self.assertIn('Silk Board', data['reply'])

if __name__ == '__main__':
    unittest.main()
