# 🚀 Quick Launch Commands for PNBTR+JELLIE Training Testbed

## Option 1: Use the Launch Script (Recommended)

```bash
cd /Users/timothydowler/Projects/JAMNet/PNBTR_JELLIE_DSP/standalone/training_testbed/PNBTR-JELLIE-TRAINER
./launch_app.sh
```

## Option 2: Use macOS `open` Command (Simplest)

```bash
cd /Users/timothydowler/Projects/JAMNet/PNBTR_JELLIE_DSP/standalone/training_testbed/PNBTR-JELLIE-TRAINER/build
open "PnbtrJellieTrainer_artefacts/Release/PNBTR+JELLIE Training Testbed.app"
```

## Option 3: Add a Shell Alias (One-time setup)

Add this to your `~/.zshrc` file:

```bash
alias launchpnbtr='cd /Users/timothydowler/Projects/JAMNet/PNBTR_JELLIE_DSP/standalone/training_testbed/PNBTR-JELLIE-TRAINER && ./launch_app.sh'
```

Then you can launch from anywhere with:

```bash
launchpnbtr
```

## Option 4: Double-click in Finder

Navigate to:

```
/Users/timothydowler/Projects/JAMNet/PNBTR_JELLIE_DSP/standalone/training_testbed/PNBTR-JELLIE-TRAINER/build/PnbtrJellieTrainer_artefacts/Release/
```

Then double-click: `PNBTR+JELLIE Training Testbed.app`

## 🎯 Current Status

The app launches successfully and shows:
✅ Core Audio bridge initializes
✅ Metal shaders compile
✅ GUI loads step-by-step
❌ AudioUnit device selection still fails (-10851)

## 🔧 Next Steps

The real issue is the AudioUnit device selection errors. The app GUI works, but audio processing is blocked by Core Audio device binding failures.
