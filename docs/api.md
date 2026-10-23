# API 与调用格式

启动后可在 `/docs` 查看请求 schema。HTTP 与 WebSocket 使用相同访问控制；非同源写请求拒绝。错误响应为 `error` 或 FastAPI `detail`，未找到 404、状态冲突 409、输入错误 422、模块不可用 503。

| 功能 | 主要接口 |
| --- | --- |
| 状态、配置 | `GET /api/health`、`GET /api/status`、`GET/PUT /api/settings` |
| 模块恢复 | `POST /api/modules/{name}/initialize` |
| 家庭守护 | `PUT /api/home/monitoring`、`POST /api/home/audio-consent` |
| 视频 | `POST /api/uploads/video`、`POST /api/video/start`、`POST /api/video/stop`、`GET /api/video/frame` |
| 连续画面 | `GET /api/video/stream`、`GET /api/sessions/{sid}/camera/stream`，MJPEG |
| 证据片段 | `GET /api/video/clip/{event_id}` |
| 信号回放 | `POST /api/signal/sample`、`POST /api/signal/analyze`，后者 multipart `file` |
| 实时信号 | `WS /ws/sensor` |
| 训练 | `POST /api/sessions`、`GET /api/sessions/{sid}`、`POST /api/sessions/{sid}/control` |
| 后台算法 | `GET/PUT /api/algorithms` |
| 运动计划 | `GET/PUT /api/plan`、`POST /api/plan/defer` |
| 长者端 | `GET /api/elder/context`、`POST /api/elder/open` |
| 训练输入 | `POST /api/sessions/{sid}/frames`、`POST /api/sessions/{sid}/image` |
| 音频 | `POST /api/audio/sample`、`POST /api/audio/analyze`、`WS /ws/audio` |
| 事件 | `GET /api/events`、`GET /api/events/{id}`、`POST /api/events/{id}/action` |
| 问询 | `POST /api/events/{id}/inquiry`、`/inquiry/processing`、`/inquiry/unavailable` |
| 主动求助 | `POST /api/help` |
| 通知 | `GET /api/notifications`、`POST /api/notifications/{id}/seen`、`POST /api/notifications/test` |
| 报告 | `GET /api/reports`、`/api/reports/export.csv`、`/reports/print`、`/reports/events/{id}` |
| 训练导出 | `GET /api/sessions/{sid}/export.json` |

视频开始请求 `{"source":"sample"}` 或 `{"source":"camera","camera_index":0}`；上传后使用 `{"source":"upload","upload_id":"上传接口返回的值"}`。

家庭入口 `/home#safety` 请求 `PUT /api/home/monitoring`，body 为 `{"enabled":true,"camera_index":0}`。设置持久保存，后台调度自动启动真实摄像头并在故障后限速重试，与预览页面是否打开无关。`GET /api/status` 中的 `home_monitoring` 包含 `enabled`、`camera_index` 和最近启动错误 `error`。`enabled:false` 暂停守护；原 `POST /api/video/stop` 同样暂停自动恢复。守护开启时禁止用视频样例替换实时监测，须先暂停。

`POST /api/home/audio-consent` 仅将本机设置中的 `audio_consent` 置为 true，保留资料、联系人等其他配置；浏览器仍需独立授予麦克风权限，音频通过原 `/ws/audio` 监听。进入家庭运动页不自动新建训练，也不中断已开启的守护。

训练开始请求：

```json
{"exercise":"chair_sit_to_stand","target_reps":10,"source":"camera"}
```

摄像头默认 `capture:"shared"`、`camera_index:0`，由服务统一采集，可与视频监测并行。外部图片／骨架采集客户端使用 `capture:"client"`，自行提交原输入契约。动作另有 `squat`、`standing_leg_raise`、`hurdle_step`，来源另有 `replay`、`upload`。控制为 `{"action":"pause"}`、`resume`、`end`。骨架批次遵循原算法2规范：

系统摄像头和上传训练指定算法2的 `action_scope:"training"`，只在四个训练动作内部计算动作支持度；其中直腿抬腿、跨步还对原输入与左右镜像输入的模型概率取均值，状态标记 `inference_views:"mirror_average"`。深蹲、坐站和样例回放均为 `inference_views:"original"`，样例回放使用原默认 `open`。返回状态、训练报告和诊断包含这两个字段。算法2独立 API 的默认值仍为 `open`。完整差异与验证边界见 `docs/exercise-integration.md`。

```json
{"request_id":"batch-0","frames":[{"timestamp_ms":0,"person_id":"no_person","keypoints":[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]}]}
```

上例表示一帧无人体观测，不计数；实际人体输入需 17 个原姿态模型的 `[x,y,confidence]` 点。图片输入为 multipart `file`，query 包含严格递增 `timestamp_ms` 和唯一 `request_id`。同一编号的不同内容返回冲突。

音频分析为 multipart `file`、`task`、`mode`；页面使用 `task:"monitor"`，返回 `sounds` 与 `help_result` 两个原算法结果。保留 `sounds`、`help` 和 `reply` 独立调用；reply 必须带关联 `event_id` 且已启动问询。样例默认联合分析，例如 JSON `{"sample":"reply_help_en"}`，固定作为回放。连续监听自动处理全部声音类别与语音求助，没有类别选择参数。

实时信号连接成功返回 `{"state":"connected"}`，发送：

```json
{"samples":[[0.0,0.0,0.0,1.0],[0.02,0.0,0.0,1.0]]}
```

每批 1 至 200 点，时间秒、加速度 g。返回原窗口/告警结果。结束发送 `{"type":"flush"}`，等待 `{"state":"stopped"}` 后断开；最终事件已入库。

连续音频连接成功返回 `{"state":"listening"}`。发送 `{"pcm":"base64 little-endian float32 mono PCM","timestamp_ms":0}`，16 kHz、每块不超过 32000 点、时间递增；浏览器每块 8000 点。处理后返回原 pipeline/listener 状态及声音/求助结果。停止发送 `{"type":"stop"}` 并等待 stopped 确认。

事件操作有 `acknowledge`、`contact`、`escalate`、`safe`、`resolve`。照护者更新附 `revision` 可避免覆盖并发处理。resolve 需要 `outcome`：`confirmed_fall`、`false_alarm`、`safe`、`unresolved`，可附 `note`。报告参数 `days=1..90` 与 `mode=live|replay`。

## Withdrawing a self-triggered alarm

`POST /api/events/{id}/action` accepts `{"action": "withdraw"}` from the older adult's page.

- An event raised by the help button (`channel: "manual"`) or a spoken call for help (`channel: "audio"`) can be withdrawn within `WITHDRAW_WINDOW_S` (30 s) of detection.
- A fall alarm of a sensor (`accel`, `vision`) can be withdrawn only after the voice inquiry has raised it to an emergency (no answer, or a call for help; the event then carries `escalated_at`), within the same 30 s counted from `escalated_at`. A fall candidate that is still PENDING, one that a valid answer cancelled, and one that a caregiver escalated cannot be withdrawn.
- Otherwise the request is refused with a 409 response.
- A withdrawal never closes or downgrades the event. It stays open at its level, the contact escalation timer stops, the caregivers get a notification, and the caregiver still confirms and resolves it.
- `GET /api/events` and `GET /api/events/{id}` add `withdrawn`, `withdrawable` and `withdraw_left_s`; the stored event carries `withdrawal: {"at", "by"}`.
- A later spoken call for help on the same event clears the withdrawal and restarts the escalation clock; a caregiver `escalate` overrides it.

## 后台算法与运动计划

`GET /api/algorithms` 返回按通道拆分的开关和运行状态，`PUT` 只提交要改的键（未提交的保持原值）：

```json
{"switches":{"resident":true,"vision":true,"accel":true,"chair_sit_to_stand":true,"squat":true,"standing_leg_raise":true,"hurdle_step":false},
 "channels":{"resident":{"label":"后台常驻","enabled":true,"state":"running","detail":"…"}},
 "blockers":["…"],"resident_ready":true}
```

`resident` 是总闸；只有它开启时才会按各自的开关拉起视频、加速度与四种运动检测。运动轮播与视频检测共用一路摄像头：真人工作台的训练会话优先，没人用时镜头直接交接，交接过程不拆设备、不断帧。`channels` 的 `state` 取 `running`、`starting`、`idle`、`unavailable` 等；`blockers` 说明为什么某个通道跑不起来（如摄像头被占用）。被关闭的动作不能新建训练，`POST /api/sessions` 返回 409。

`GET/PUT /api/plan` 读写照护者设定的每日计划，返回体附带 `progress`（今天的完成情况）。`items` 最多 4 项、同一动作只能出现一次，重复或未知动作返回 422：

```json
{"enabled":true,"title":"今天的康复计划","note":"…","target_reps":10,
 "items":[{"exercise":"chair_sit_to_stand","target_reps":10,"enabled":true}],
 "skipped_dates":["2026-10-18"]}
```

`progress.items[]` 的 `reps` 来自当天已结束（`status:"ended"`、`mode:"live"`）的训练记录，`reps` 达到 `target_reps` 才算 `done`；`disabled` 表示该动作的检测开关已关闭。`progress.skipped` 为 true 表示长者今天选择了「先不做」。

`POST /api/plan/defer` 供长者端记录跳过，body 可省略：

```json
{"days":1}
```

`days` 取 1–7，默认 1，写入 `skipped_dates`（今天起 `days` 天）。计划内容不变，被跳过的日期里长者端不显示计划卡片，次日自动恢复；照护者端仍能看到计划与今天的完成情况。`GET /api/elder/context` 一次返回称呼、房间、计划与进度；`POST /api/elder/open` 记录长者端打开时间。

## 清空报告范围

`POST /api/records/clear` 会整段删除范围内的安全事件、训练记录、通知与录像片段，**不可撤销**。因为各字段都有默认值，空 body 就等于「近 7 天实时记录全清」，所以 `confirm` 是必填的字面量确认词：

```json
{"mode":"live","days":7,"events":true,"sessions":true,"confirm":"DELETE"}
```

缺少 `confirm` 或值不等于 `DELETE` 时返回 422，什么都不会删。

## Interface language

The interface is English only; there is no language switch, and a language saved in `localStorage["homeease.lang"]` by an earlier version is ignored. Every page loads `static/i18n-en.js` (Chinese source text to English phrase table) and `static/i18n.js`, which translates the DOM text at load and as it changes, the text of `alert`/`confirm`/`prompt` dialogs and the browser notifications. What the server hands out directly is translated by `care/i18n.py` with the same table and rules: the CSV export (`/api/reports/export.csv`, English headers) and the `title`/`message` of webhook deliveries (the stored notification keeps its original text). `tests/test_i18n_server.py` checks that both translators give the same output. `tests/test_i18n_coverage.py` fails when a Chinese string in the templates, scripts or server code has no English phrase, and `node scripts/test-i18n.cjs` checks the translator.

## Sound events as an alarm channel

Every new HELP, IMPACT or ABNORMAL sound event of the live listening (`/ws/audio`) is looked at three seconds after it starts, so that a spoken call or a fall detection of the same moment is in the database first.

- If a live event is open and its first marker is within 10 s of the sound, an audio marker of kind `sound_event` is added. A fall candidate (`accel`, `vision`) becomes `dual_channel` and is raised to EMERGENCY (`escalated_at` is set, so it can be withdrawn for 30 s) and the caregivers are notified.
- Otherwise a PENDING event with `channel: "audio"` and the source "sound monitoring" is opened for a caregiver to check.
- DAILY and UNKNOWN sounds are no alarm. The sound model is weak on the three classes (HELP precision about 0.5), so a lone sound only asks for a check. `SOUND_ALARM_CATEGORIES` in `care/events.py` lists the categories; an empty tuple switches this off.
