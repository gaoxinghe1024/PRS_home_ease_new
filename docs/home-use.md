# Household Entry

The household entry (`/home`) is the simplified interface for the older adult. It reuses the code, models and event-handling rules of Algorithms 1, 2 and 3 unchanged.

## Pages

| Address | User and behaviour |
| --- | --- |
| `/home` | older adult's home page: choose Home Monitoring or Exercise |
| `/home#safety` | starts video fall monitoring automatically; once consent is given, sound listening and the voice inquiry connect automatically |
| `/home#exercise` | the four exercises with movement guides; a session starts only when the older adult taps Start |
| `/#reports` | family members' reports |
| `/` | family members' care workbench: event handling, profile and settings |

On first use a family member fills in the profile and contacts, opens Home Monitoring, checks that the camera can see the older adult, and then grants local audio processing and the browser's microphone permission. The system distinguishes video connected, no person in view, audio not authorized, microphone unavailable and sensor not connected. Granting audio never changes existing contacts or the profile.

Home Monitoring and Exercise switch within the same page while sound listening keeps running. An exercise session shares the monitoring camera, and pausing or ending it only affects the session. From the household page the older adult can ask for help, answer the voice inquiry, or withdraw an accidental alarm under the original rules; alarms are still verified and handled by family members and never close by themselves.

## Continuous operation

- The monitoring intent and the camera index are stored in a separate SQLite table, `home_monitoring`. Once on, the background scheduler keeps it running without depending on a preview or a browser heartbeat.
- After a restart of the local service, video monitoring resumes once the original models are ready. After an input interruption it retries every 15 seconds, and errors are shown on the page and in the status API.
- Pause turns automatic recovery off persistently; choosing Home Monitoring again or tapping Resume turns it back on. The video stop button of the workbench also turns automatic recovery off.
- While household video monitoring is on, the workbench cannot replace the live camera with a replay; pause monitoring first.
- Sound listening and the automatic inquiry use the browser microphone with the joint listening and inquiry interfaces of the original Algorithm 3, so the page must stay open. After permission is granted, reloading the page recovers it; when permission is denied or no device is found, the reason is shown and the page waits for a manual reconnect instead of asking for permission in a loop.
- While the question is being played, continuous listening is muted so the question from the speaker is not taken as a call for help; the answer is recorded from a cloned microphone track, and continuous listening keeps running afterwards.
- The page asks the screen to stay awake where possible, but nothing can be monitored while the computer sleeps, is shut down or the service is stopped. The accelerometer branch still needs continuous input from the sensor page or an external client.

## Validation scope

During development the household entry was tested end to end with an isolated data folder and a real server: switch persistence, recovery after restart, rate-limited retries, pausing, camera switching, independent exercise sessions, background model recovery, shared camera use, the voice inquiry and denied permissions, at desktop and phone widths. Browser tests used a synthetic microphone source, so they are not an acceptance test of speech recognition with real voices. Long-term stability on household devices and real alarm delivery still need on-site validation.
