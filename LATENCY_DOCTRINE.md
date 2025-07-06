# 🎯 JAMNet Latency Doctrine

## Purpose
To define the philosophy, rules, and performance ceiling of JAMNet in absolute terms.

Latency is not an optimization target.  
Latency is not a feature.  
**Latency is the enemy: The operating LIMIT of reality itself. If we are going to virtually put people in the same room together the illusion of continuity HINGES on redefining traditional latency standards — JAMNet is purpose built with this philosophy in mind.**

---

## 👁️‍🗨️ Guiding Principle

> **Saved time is not spare time — it’s reclaimed opportunity.**

If we shave 5 ms off our round-trip, that time isn’t banked.  
It is immediately reinvested in:
- Additional prediction
- Tighter synchronization
- Greater recovery capability
- Higher musical accuracy

There is no such thing as “good enough” timing.  
There is only **closer to true time** or **not**.

---

## ⏱️ Latency Budget Rules

| Category                       | Target Maximum (Round-Trip) | Status         |
|--------------------------------|------------------------------|----------------|
| Metro-Area LAN (subnet/local) | **5 ms**                     | ✅ Surpassed    |
| Regional WAN (<300 km)        | **10 ms**                    | ✅ Viable       |
| Inter-state (~1000 km)        | **15 ms**                    | ⚠️ At limit     |
| Cross-continent               | Best-effort, fallback to prediction | 🔁 In progress  |
| Interplanetary (Future)       | Sub-luminal until wormholes | 🧪 Theoretical  |

---

## 🚦 Behavioral Requirements

- **No rounding to ms:** All scheduling must be µs-level or better.
- **No jitter acceptance:** Variability in block delivery must be compensated or rejected.
- **No fallback to CPU unless GPU is unavailable:** All primary transport must be GPU-clocked.
- **PNBTR must always smooth packet loss before signal dropout reaches perceptible threshold.**
- **All clients must be able to self-assess drift and correct within 2 ms autonomously.**

---

## 🔩 Engineering Mandates

- All audio transmission events must carry:
  - `timestamp_gpu`
  - `sync_calibration_block` offset if available
- Frame scheduling must be deterministic from GPU clock, not system wall time.
- If latency is reduced, **budget is updated downward** permanently.
- No buffering strategy may exceed the current round-trip threshold.

---

## 🧠 Psychological Consideration

- Perceived latency in ensemble music becomes noticeable at:
  - ~12 ms for groove-sensitive players
  - ~8 ms for percussionists or vocalists
  - ~<5 ms to feel "in the same room"
- JAMNet targets **under 7 ms** to make physical separation *emotionally invisible.*

---

## 🔮 Future Integration

- When available, latency-reducing physics (See document: FTL.md) may permit:
  - **Nonlinear data arrival**
  - **Preemptive synchronization**
  - **Wormhole-anchored TOAST endpoints**

Latency goals will update in real-time with breakthroughs in causality-bound communication.

---

## 🧬 Final Statement

**JAMNet does not tolerate latency.  
It eliminates it. There is no system we do not challenge, or length we won’t go to (including challenging the perceived limits of physical reality) to keep Latency’s throat under our boots.**

There is no buffer big enough to justify delay of the groove.  
There is no round-trip long enough to excuse disconnection.  
**JAMNet redefines what “real time processing” even means.**

