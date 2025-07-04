## JVID + JAMCam Setup Instructions

This section outlines how to initialize and prepare the video stream layer for JAMNet using JVID and the JAMCam application module.

### Overview

- **JVID** is the video counterpart to JDAT.
- It streams compressed visual data as JSON objects, frame-by-frame, prioritizing ultra-low latency over quality.
- **JAMCam** is the app-layer interface responsible for capturing, encoding, decoding, and rendering video streams using JVID.

### 1. Requirements

- Node.js >= 18.x
- A machine with webcam support and access permissions
- WebRTC + WebSockets enabled in browser or Electron shell
- Access to the local TOAST transport (UDP)
- Optional: FFmpeg (for debugging frame encoding)

### 2. Project Structure

/jvid/
├── jamcam.js # Video capture + frame tokenizer
├── render.js # Receiver-side visual renderer
├── compressors/
│ ├── downscale.js # Resize input video to ~low res
│ └── jpegstrip.js # Reduce frame to essential pixels
├── stream/
│ ├── packetize.js # Wrap JVID packets for TOAST
│ └── depacketize.js # Unwrap on receiver side
└── utils/
└── timestamp.js # Frame sync with audio & MIDI

markdown
Copy
Edit

### 3. Setup Instructions

**a. Capture and encode:**

- `jamcam.js` uses `navigator.mediaDevices.getUserMedia()` to access webcam.
- Frames are downsampled to ~144p or lower using `downscale.js`.
- Each frame is stripped into base64-encoded JSON-friendly byte arrays.

**b. Send over TOAST:**

- `packetize.js` splits each frame into TOAST-compatible datagrams.
- Transmit using the same socket manager as JDAT/MIDI.

**c. Receive and decode:**

- On the receiving client, `depacketize.js` reassembles the chunks.
- `render.js` paints the received frames on a canvas with timestamp alignment via `timestamp.js`.

### 4. Notes

- JAMCam throttles framerate to match audio clock sync (default 10–20fps).
- Visual prediction or frame interpolation is handled by `pntbtr-core.js` if frames are lost.
- Latency budget for video = Audio latency target + ~10ms

### 5. Coming Soon

- JamCam mobile preview
- Face-framing + lighting normalization filters
- Edge-based caching for nearby JAMWAN video relays
