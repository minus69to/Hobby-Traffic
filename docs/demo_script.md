# Two-minute narrated demo script

Target: a 2-minute screen recording of the website (split view) with voice-over. Times are approximate.

**0:00 - 0:15, hero.** "This is a synthetic Dhaka-like junction in SUMO: buses, cars, motorcycles,
CNG auto-rickshaws, cycle-rickshaws and pedestrians sharing lanes. On the left the signal runs a
Webster fixed-time plan; on the right a reinforcement-learning controller. Same demand, same clock."

**0:15 - 0:40, the problem.** Switch to the street-level camera. "Lane discipline is loose: the
sublane model lets motorcycles and CNGs weave, rickshaws crawl at twelve kilometres per hour, and
pedestrians cross at the signal. A controller that only counts vehicles starves the slow classes."

**0:40 - 1:05, how the agent works.** Open the HUD panels. "Every five seconds the agent sees queues,
waiting times and counts per vehicle class, plus the pedestrians waiting at each corner, and picks the
next green. The counters on the right show the queues it is clearing."

**1:05 - 1:30, safety.** Scroll to the safety section. "The agent can only choose what the mask
allows: no change before ten seconds of green, a forced change after sixty, nothing during yellow or
all-red. Behind it a runtime safety shield, a small state machine every controller passes through,
inserts yellow and all-red and rejects conflicting greens. Property-based tests throw thousands of
random action sequences at it; the intervention badge shows how often it had to step in."

**1:30 - 1:50, results.** Scroll to the result cards and charts. Read the morning-peak change in mean
waiting time and the pedestrian wait from the cards (they are generated from `results/`), mention the
confidence intervals and the per-class fairness chart.

**1:50 - 2:00, edge.** "The policy exports to a few kilobytes of int8 ONNX, runs in well under a
millisecond on one CPU thread, and ships as an MQTT service with a fixed-time watchdog fallback.
Everything is synthetic; the next step is real detector data."
