"""scheduler 테스트 (게이트 2 승인 2케이스)."""

from app.report.scheduler import create_scheduler, run_weekly_job
from tests.test_report_weekly import fake_embedder


class FakeSlackClient:
    def __init__(self):
        self.calls = []

    def chat_postMessage(self, *, channel, text):
        self.calls.append({"channel": channel, "text": text})


def test_cron_trigger_monday_9am_kst():
    scheduler = create_scheduler(lambda: None)
    jobs = scheduler.get_jobs()
    assert len(jobs) == 1
    trigger = jobs[0].trigger
    fields = {field.name: str(field) for field in trigger.fields}
    assert fields["day_of_week"] == "mon"
    assert fields["hour"] == "9"
    assert fields["minute"] == "0"
    assert str(trigger.timezone) == "Asia/Seoul"


def test_run_weekly_job_posts_report(conn):
    client = FakeSlackClient()
    run_weekly_job(client, "C123", conn=conn, embedder=fake_embedder)
    assert client.calls[0]["channel"] == "C123"
    assert "주간 VoC 리포트" in client.calls[0]["text"]
