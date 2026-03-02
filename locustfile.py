from locust import HttpUser, task

class DriverUser(HttpUser):

    @task
    def test_analyze(self):
        self.client.post("/v1/analyze", json={
            "Speed": 110,
            "Alertness": 0.4,
            "Seatbelt": 1,
            "HR": 95,
            "Fatigue": 7,
            "speed_change": 5,
            "prev_alertness": 0.6
        })