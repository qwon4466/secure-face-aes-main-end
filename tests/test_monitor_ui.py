from pathlib import Path


def test_monitor_uses_local_svg_icons_and_recovers_stuck_snapshot_request():
    html = (Path(__file__).parents[1] / 'templates' / 'monitor.html').read_text()
    assert '<symbol id="icon-camera"' in html
    assert '<symbol id="icon-user"' in html
    assert '<symbol id="icon-alert"' in html
    assert '<symbol id="icon-record"' in html
    assert '<symbol id="icon-bell"' in html
    assert 'FRAME_REQUEST_TIMEOUT_MS' in html
    assert 'clearTimeout(timer)' in html
    assert "camera_state === 'stale'" in html
    assert "camera_state === 'reconnecting'" in html
    for unsupported in ('🔒', '📹', '👤', '🚨', '🔴', '🟢'):
        assert unsupported not in html


def test_monitor_alert_toggle_is_local_persistent_and_does_not_replay_events():
    html = (Path(__file__).parents[1] / 'templates' / 'monitor.html').read_text()
    assert html.index('id="recordingBadge"') < html.index('id="alertStateCard"') < html.index('id="modePill"')
    assert 'id="alertSwitch"' in html
    assert "ALERT_ENABLED_KEY = 'secureface.alerts.enabled'" in html
    assert 'localStorage.getItem(ALERT_ENABLED_KEY)' in html
    assert 'localStorage.setItem(ALERT_ENABLED_KEY, String(alertsEnabled))' in html
    assert 'if (alertsEnabled) {' in html
    assert 'lastAlertSeq = seq;' in html
    assert 'alertOverlay.hidden = true;' in html
    assert 'stopAlertSound();' in html
    assert 'playAlertSound();' in html


def test_templates_do_not_depend_on_emoji_fonts():
    templates = Path(__file__).parents[1] / 'templates'
    for path in templates.glob('*.html'):
        html = path.read_text()
        assert not any(ord(char) >= 0x1f000 for char in html), path.name
        assert not any(char in html for char in ('✅', '❌', '⚠', '⭐', '⏸', '▶')), path.name
