# ব্যবহার নির্দেশিকা (বাংলা)

এই প্রজেক্টটি SUMO সিমুলেটরে ঢাকার মতো মিশ্র ট্রাফিকের (বাস, গাড়ি, মোটরসাইকেল, সিএনজি, রিকশা, পথচারী) একটি
সিগন্যালাইজড জংশন তৈরি করে, সেখানে একটি **নিরাপদ** রিইনফোর্সমেন্ট-লার্নিং (MaskablePPO) সিগন্যাল কন্ট্রোলার
ট্রেন করে, ফিক্সড-টাইম ও ম্যাক্স-প্রেশার বেসলাইনের সাথে তুলনা করে, এবং ফলাফল একটি 3D ওয়েবসাইটে দেখায়।

> সব ডিমান্ড ও যানবাহনের প্যারামিটার **সিন্থেটিক অনুমান**, ঢাকার ট্রাফিক থেকে অনুপ্রাণিত; কোনো বাস্তব ডেটা বা
> বাস্তব ভ্যালিডেশন নেই।

## ১. এক নজরে কোথায় কী আছে

| জায়গা | কী আছে |
|---|---|
| `configs/` | সিমুলেশন, ট্রেনিং, ইভ্যালুয়েশন ও সার্ভিসের YAML কনফিগ |
| `src/drs/` | পুরো পাইথন প্যাকেজ (এনভায়রনমেন্ট, শিল্ড, কন্ট্রোলার, ট্রেনিং, ইভ্যালুয়েশন, সার্ভিস) |
| `results/` | সব সংখ্যা: `eval_summary.json`, `main_table.csv`, `edge_benchmark.json` ইত্যাদি |
| `figures/` | চার্ট (PNG) |
| `docs/report.md` | টেকনিক্যাল রিপোর্ট; `docs/media/` এ GIF, ভিডিও, স্ক্রিনশট |
| `web/` | 3D ওয়েব অ্যাপ (Vite + React + three.js) |
| `models/final/` | ট্রেন করা পলিসি (SB3 zip + ONNX) |
| লাইভ সাইট | https://minus69to.github.io/Hobby-Traffic/ |

## ২. সেটআপ (একবার)

প্রয়োজন: Python 3.11+, Node 22, `ffmpeg`, (ঐচ্ছিক) `mosquitto`। লিনাক্স/ম্যাকে:

```bash
make setup
```

এটি `.venv` তৈরি করে, SUMO (pip এর `eclipse-sumo`), Stable-Baselines3, torch (CPU) ইত্যাদি ইনস্টল করে এবং
`web/` এর npm ডিপেন্ডেন্সি নামায়। এরপর প্রতিটি কমান্ডের আগে:

```bash
source .venv/bin/activate
export PYTHONPATH=src LIBSUMO_AS_TRACI=1
```

(`make` টার্গেটগুলো এগুলো নিজেই সেট করে।)

## ৩. সিমুলেশন চালানো

নেটওয়ার্ক ও ডিমান্ড তৈরি:

```bash
make networks     # sim/networks/single.net.xml ও grid2x2.net.xml (netconvert, লেফট-হ্যান্ড)
make demand       # প্রতিটি সিনারিও ও সিডের রুট ফাইল (sim/demand/, git এ রাখা হয় না)
```

স্পিড টেস্ট (এক ঘণ্টার মর্নিং পিক, ফিক্সড-টাইম কন্ট্রোলার):

```bash
make speedtest    # results/speedtest.json
```

একটি সিনারিও GUI সহ নিজে দেখতে চাইলে (SUMO-GUI দরকার):

```bash
python - <<'PY'
from drs.config import SimConfig
from drs.controllers import FixedTimeController
from drs.env import TrafficSignalEnv
from drs.runner import run_episode
from drs.sumo_sim import SumoSim
SumoSim.__init__.__defaults__  # noqa
cfg = SimConfig.load()
env = TrafficSignalEnv(cfg, "single", "morning_peak", [13], episode_seconds=600)
env.reset(seed=13)
print("junction state:", env.last_states["C"].lane_halting)
env.close()
PY
```

সিনারিওগুলো (`configs/sim.yaml`): `offpeak`, `morning_peak`, `evening_peak`, `incident` (উত্তর দিকের এক লেন
১৫ মিনিট বন্ধ), `surge` (+২০%), `noisy` (সেন্সর নয়েজ + ৫% ড্রপ)। যানবাহনের মিশ্রণ বদলাতে `vehicle_mix`
অংশটি সম্পাদনা করুন।

## ৪. সেফটি শিল্ড ও টেস্ট

```bash
make lint         # ruff + mypy
make test         # ইউনিট ও property-based (hypothesis) টেস্ট
HYPOTHESIS_PROFILE=thorough .venv/bin/python -m pytest tests/test_shield.py   # ২০০০ র‍্যান্ডম সিকোয়েন্স
make test-all     # slow (SUMO) ও integration (MQTT) টেস্টসহ, mosquitto লাগবে
```

শিল্ডের নিয়ম: ন্যূনতম সবুজ ১০ সেকেন্ড, সর্বোচ্চ ৬০, হলুদ ৩, অল-রেড ২, পথচারী ওয়াক ন্যূনতম ৮। প্রতিটি
হস্তক্ষেপ লগ হয় এবং `results/` এর সব টেবিলে "shield interventions" কলামে দেখানো হয় (শূন্য হলেও)।

## ৫. ট্রেনিং

```bash
make train                        # configs/train.yaml: সিড 13, 42, 2026; প্রতিটি ~৯০ মিনিট (৪ কোর)
.venv/bin/python -m drs.train --config configs/train.yaml --seed 13   # শুধু একটি সিড
.venv/bin/tensorboard --logdir models/single/seed13/tb                # লার্নিং কার্ভ
```

ছোট বাজেটে পুরো পাইপলাইন ১৫ মিনিটে পরখ করতে:

```bash
make reproduce
```

ট্রেনিং শেষে সেরা চেকপয়েন্ট `models/final/` এ নিতে ও ONNX এক্সপোর্ট করতে:

```bash
scripts/after_seed.sh 13
```

## ৬. ইভ্যালুয়েশন ও ফলাফল

```bash
make eval         # ৬ সিনারিও x ১০ সিড x সব কন্ট্রোলার -> results/eval_summary.json, main_table.csv
make report       # figures/*.png
make docs         # README.md ও docs/report.md টেমপ্লেট থেকে রেন্ডার (সব সংখ্যা results/ থেকে)
make check-numbers   # README, রিপোর্ট ও ওয়েবসাইট বান্ডেলের সংখ্যা results/ এর সাথে মেলে কি না
```

`results/main_table.csv` এ প্রতিটি কন্ট্রোলার ও সিনারিওর জন্য গড়, ৯৫% CI এবং ফিক্সড-টাইমের তুলনায় শতাংশ
পরিবর্তন আছে। মেট্রিকগুলো: গড় ও ৯৫তম পার্সেন্টাইল ভ্রমণ সময়, গড় অপেক্ষা, গড় কিউ, থ্রুপুট (প্রতি ঘণ্টা),
ক্লাসভিত্তিক অপেক্ষা (রিকশা, বাস...), পথচারীর অপেক্ষা, শিল্ড হস্তক্ষেপ, ফেজ পরিবর্তন/ঘণ্টা, insertion delay,
unfinished share।

**ফলাফল পড়ার সময় মনে রাখুন:** পিক সিনারিওগুলো অতি-স্যাচুরেটেড (প্রায় এক-তৃতীয়াংশ গাড়ি এক ঘণ্টায় ট্রিপ শেষ
করতে পারে না)। এই বাজেটে PPO ফিক্সড-টাইমের কাছাকাছি, ম্যাক্স-প্রেশার অপেক্ষার দিক থেকে সবচেয়ে ভালো; PPO
পথচারীর অপেক্ষা প্রায় ৪০% কমায় এবং শিল্ডকে কখনো হস্তক্ষেপ করতে হয়নি। বিস্তারিত `docs/report.md` সেকশন ৬।

## ৭. 3D ওয়েবসাইট

রিপ্লে ডেটা ও জিওমেট্রি তৈরি (ট্রেন করা `ppo_s13` মডেল লাগে):

```bash
make geometry     # web/public/data/geometry_*.json
make replay       # web/public/data/replays/*.drsr.gz (প্রতিটি ~৩ MB)
make web-dev      # http://localhost:5173/dhaka-rl-signals/  (ডেভ সার্ভার)
make web          # প্রোডাকশন বিল্ড web/dist
cd web && npx playwright test    # স্মোক টেস্ট: ক্যানভাস লোড হয়, কনসোলে কোনো এরর নেই
```

সাইটে যা যা পরখ করবেন:

- **স্প্লিট ভিউ**: বাঁয়ে ফিক্সড-টাইম, ডানে RL, একই ডিমান্ড ও ঘড়ি; মাঝের দাগ টেনে ভাগ বদলান (কীবোর্ডে ←/→)।
- **Explore the simulation** বোতাম: ফুল-স্ক্রিন কন্ট্রোল প্যানেল।
- **প্লেব্যাক**: play/pause (space), গতি ০.৫x–৮x, টাইমলাইন স্ক্রাবার (incident উইন্ডো লাল দাগে), সিনারিও ও
  কন্ট্রোলার সিলেক্টর।
- **ক্যামেরা**: aerial (1), top (2), street (3); কোনো গাড়িতে ক্লিক করলে follow মোড (ক্লাস, গতি, অপেক্ষা দেখায়); Esc রিসেট।
- **চেহারা**: golden hour / night (n; নাইটে জানালা, ল্যাম্প, হেডলাইট; সিঙ্গেল ভিউতে bloom), rain (r), heat colours
  (h; অপেক্ষার সময় অনুযায়ী রং), lane heatmap (লেনে কিউ ঘনত্ব), reduced motion।
- **HUD**: অপেক্ষমাণ গাড়ি, গড় অপেক্ষা, পথচারী, বর্তমান ফেজ ও টাইমার, শিল্ড-হস্তক্ষেপ ব্যাজ, দুই কন্ট্রোলারের
  লাইভ লাইন চার্ট — সব রিপ্লে ম্যানিফেস্ট থেকে, কিছু হার্ড-কোড নয়।
- **নিচে স্ক্রল করলে**: রেজাল্ট কার্ড, "how it works" ডায়াগ্রাম, সেফটি স্টেট মেশিন, সিনারিও/ক্লাসভিত্তিক চার্ট ও
  পূর্ণ টেবিল, এজ ডিপ্লয়মেন্ট সংখ্যা, অতিরিক্ত ML কম্পোনেন্ট, সীমাবদ্ধতা।
- মোবাইলে একটি ভিউ, "show left/right" বোতামে বদল।

## ৮. এজ এক্সপোর্ট ও MQTT সার্ভিস

```bash
make export       # ONNX (opset 17) + int8; parity ও latency -> results/edge_benchmark.json
```

পুরো লুপ (ব্রোকার + সার্ভিস + SUMO ব্রিজ) লোকালি:

```bash
mosquitto -p 1883 &                                  # ব্রোকার
make service                                         # সার্ভিস: MQTT + http://localhost:8080/health, /metrics
.venv/bin/python -m drs.service.bridge --scenario morning_peak --seconds 300   # SUMO ব্রিজ, ৫ মিনিট
```

টপিক: `drs/junction/C/detectors` (ইনপুট), `/command`, `/status`, `/alarm`। মডেল না থাকলে বা ডেটা বাসি হলে
সার্ভিস ফিক্সড-টাইমে নেমে যায় ও অ্যালার্ম পাঠায়। TLS চাইলে `scripts/gen_certs.sh` চালিয়ে
`configs/service.yaml` এ `tls: true` দিন। Docker: `docker compose up --build` (ডেভ VM এ Docker ছিল না, তাই
ফাইলগুলো চালিয়ে দেখা হয়নি)।

## ৯. মিডিয়া (GIF/ভিডিও/স্ক্রিনশট)

```bash
make media        # হেডলেস Chromium (SwiftShader) দিয়ে docs/media/ এ hero.gif, mp4, png
```

## ১০. অতিরিক্ত ML কম্পোনেন্ট

```bash
.venv/bin/python -m drs.forecast --config configs/forecast.yaml     # GRU/TCN কিউ ফোরকাস্ট -> results/forecast.json
.venv/bin/python -m drs.imitation --config configs/train_bc.yaml    # ম্যাক্স-প্রেশার থেকে behaviour cloning
.venv/bin/python -m drs.train --config configs/train_bc.yaml        # BC ওয়ার্ম-স্টার্ট + PPO
.venv/bin/python scripts/collect_imitation.py                        # তুলনা -> results/imitation.json
.venv/bin/python -m drs.train --config configs/train_grid_gnn.yaml   # 2x2 গ্রিড, গ্রাফ-অ্যাটেনশন পলিসি
```

## ১১. সমস্যা হলে

- `libsumo` এক প্রসেসে একটিই সিমুলেশন চালাতে পারে; তাই সব ওয়ার্কার আলাদা প্রসেস।
- ডেডলক এড়াতে `configs/sim.yaml` এ `ignore_junction_blocker: 30` ও `pedestrian_jamtime: 60` আছে (DECISIONS.md ৪)।
- সিদ্ধান্তের কারণ `DECISIONS.md`, কাজের ধারা `PROGRESS.md`, জানা দুর্বলতা `REVIEW.md` এ।
