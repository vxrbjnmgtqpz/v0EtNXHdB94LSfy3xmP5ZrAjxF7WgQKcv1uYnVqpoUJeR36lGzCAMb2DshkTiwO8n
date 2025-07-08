const { ipcRenderer } = require("electron");

// UI Elements - Declare variables (will be initialized after DOM loads)
let transmitModeBtn,
  receiveModeBtn,
  transmitConfig,
  receiveConfig,
  discoveryPanel;
let inputSource,
  sourceDetails,
  sourceList,
  txMulticast,
  txPort,
  txSession,
  txFps,
  txResolution,
  txCodec,
  txBurst,
  txDiscovery,
  startTransmitBtn,
  stopTransmitBtn;
let rxMulticast,
  rxPort,
  rxDiscovery,
  rxPeerFilter,
  rxSession,
  startReceiveBtn,
  stopReceiveBtn;
let discoveredPeers, refreshDiscoveryBtn, connectPeerBtn;
let videoFrame, noVideo, statusIndicator, latencyDisplay, frameInfo;
let protocolStatus,
  activePeers,
  framesSent,
  framesReceived,
  bytesSent,
  bytesReceived,
  packetLoss,
  avgLatency,
  burstPackets,
  currentFps;

// Application state
let currentMode = "transmit";
let isTransmitting = false;
let isReceiving = false;
let frameCount = 0;
let lastFrameTime = Date.now();
let discoveredPeersList = new Map(); // Store discovered peers
let availableSources = [];
let selectedSource = null;
let captureInterval = null;

// Video elements
let videoElement;
let canvasElement;
let ctx;
let sourceSelect;
let codecSelect;
let compressionStatsDiv;
let videoProcessingStatsDiv;

// JVID processor for 60fps RGB matrix processing
let jvidProcessor;

// Performance monitoring for JVID
let performanceMonitor = {
  frameCount: 0,
  lastTime: 0,
  actualFPS: 0,
  processingTime: 0,
  networkStats: {
    rgbLanesSent: 0,
    pnbtrPredictions: 0,
    averageLatency: 0,
  },
};

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  console.log("DOM loaded, initializing JAMCam...");

  // Initialize all DOM element references after DOM is loaded
  initializeDOMElements();

  generateSessionId();
  setupEventListeners();

  // Set transmit mode as default (after DOM elements are initialized)
  switchMode("transmit");

  console.log("About to load available sources...");
  loadAvailableSources();

  // Initialize JVID processor
  initializeJVIDProcessor();

  // Start video processing at 60fps
  startVideoProcessing();
});

function initializeDOMElements() {
  // Initialize all DOM elements after DOM is loaded
  transmitModeBtn = document.getElementById("transmit-mode");
  receiveModeBtn = document.getElementById("receive-mode");
  transmitConfig = document.getElementById("transmit-config");
  receiveConfig = document.getElementById("receive-config");
  discoveryPanel = document.getElementById("discovery-panel");

  inputSource = document.getElementById("input-source");
  sourceDetails = document.getElementById("source-details");
  sourceList = document.getElementById("source-list");
  txMulticast = document.getElementById("tx-multicast");
  txPort = document.getElementById("tx-port");
  txSession = document.getElementById("tx-session");
  txFps = document.getElementById("tx-fps");
  txResolution = document.getElementById("tx-resolution");
  txCodec = document.getElementById("tx-codec");
  txBurst = document.getElementById("tx-burst");
  txDiscovery = document.getElementById("tx-discovery");
  startTransmitBtn = document.getElementById("start-transmit");
  stopTransmitBtn = document.getElementById("stop-transmit");

  rxMulticast = document.getElementById("rx-multicast");
  rxPort = document.getElementById("rx-port");
  rxDiscovery = document.getElementById("rx-discovery");
  rxPeerFilter = document.getElementById("rx-peer-filter");
  rxSession = document.getElementById("rx-session");
  startReceiveBtn = document.getElementById("start-receive");
  stopReceiveBtn = document.getElementById("stop-receive");

  discoveredPeers = document.getElementById("discovered-peers");
  refreshDiscoveryBtn = document.getElementById("refresh-discovery");
  connectPeerBtn = document.getElementById("connect-peer");

  videoFrame = document.getElementById("video-frame");
  noVideo = document.getElementById("no-video");
  statusIndicator = document.getElementById("status");
  latencyDisplay = document.getElementById("latency");
  frameInfo = document.getElementById("frame-info");

  protocolStatus = document.getElementById("protocol-status");
  activePeers = document.getElementById("active-peers");
  framesSent = document.getElementById("frames-sent");
  framesReceived = document.getElementById("frames-received");
  bytesSent = document.getElementById("bytes-sent");
  bytesReceived = document.getElementById("bytes-received");
  packetLoss = document.getElementById("packet-loss");
  avgLatency = document.getElementById("avg-latency");
  burstPackets = document.getElementById("burst-packets");
  currentFps = document.getElementById("current-fps");

  // Video elements for JVID processing
  videoElement = document.getElementById("video-frame");
  // Create canvas for JVID processing (not in DOM)
  canvasElement = document.createElement("canvas");
  ctx = canvasElement.getContext("2d");
}

function generateSessionId() {
  const sessionId = "jamcam_" + Date.now().toString(36);
  txSession.value = sessionId;
}

// Input source management
async function loadAvailableSources() {
  console.log("loadAvailableSources called!");

  // Safety check for inputSource element
  if (!inputSource) {
    console.error("inputSource element not found!");
    return;
  }

  try {
    inputSource.innerHTML = '<option value="">Loading sources...</option>';
    console.log("About to invoke get-available-sources IPC...");

    const sources = await ipcRenderer.invoke("get-available-sources");
    console.log("IPC response received:", sources);

    if (!sources || sources.length === 0) {
      console.warn("No sources received from IPC call");
      inputSource.innerHTML = '<option value="">No sources available</option>';
      return;
    }

    availableSources = sources;

    // Completely clear and rebuild the dropdown
    inputSource.innerHTML = "";

    // Add default option
    const defaultOption = document.createElement("option");
    defaultOption.value = "";
    defaultOption.textContent = "Select input source...";
    inputSource.appendChild(defaultOption);

    // Add screen sources
    const screenSources = sources.filter((source) => source.type === "screen");
    screenSources.forEach((source) => {
      const option = document.createElement("option");
      option.value = source.id;
      option.textContent = `📺 ${source.name}`;
      inputSource.appendChild(option);
    });

    // Add camera sources
    const cameraSources = sources.filter((source) => source.type === "camera");
    cameraSources.forEach((source) => {
      const option = document.createElement("option");
      option.value = source.id;
      option.textContent = `📷 ${source.name}`;
      inputSource.appendChild(option);
    });

    console.log(
      "Sources processed - Total:",
      sources.length,
      "Cameras:",
      cameraSources.length,
      "Screens:",
      screenSources.length
    );

    // Auto-select first camera in TX mode (prefer built-in camera)
    if (currentMode === "transmit" && cameraSources.length > 0) {
      // Look for built-in camera first (FaceTime HD Camera)
      let selectedCamera = cameraSources.find(
        (camera) =>
          camera.name.toLowerCase().includes("facetime") ||
          camera.name.toLowerCase().includes("built-in")
      );

      // Fallback to first camera if no built-in found
      if (!selectedCamera) {
        selectedCamera = cameraSources[0];
      }

      inputSource.value = selectedCamera.id;
      selectedSource = selectedCamera;
      console.log("Auto-selected camera:", selectedCamera.name);

      // Start preview immediately with a slight delay to ensure DOM is ready
      setTimeout(async () => {
        try {
          await startPreview();
        } catch (error) {
          console.error("Failed to start preview:", error);
        }
      }, 100);

      // Set JVID defaults: 60fps RAW PCM at 144p for optimal timing accuracy
      if (txFps) {
        txFps.value = "60";
      }
      if (txResolution) {
        txResolution.value = "LOW_144P";
      }
      if (txCodec) {
        txCodec.value = "raw";
      }
    }
  } catch (error) {
    console.error("Failed to load available sources:", error);
    inputSource.innerHTML = '<option value="">Error loading sources</option>';
  }
}

function populateSourceList(sourceType) {
  sourceList.innerHTML = '<option value="">Loading sources...</option>';

  const filteredSources = availableSources.filter((source) => {
    if (sourceType === "screen") return source.type === "screen";
    if (sourceType === "window") return source.type === "window";
    if (sourceType === "webcam") return source.type === "camera";
    return false;
  });

  if (filteredSources.length === 0) {
    sourceList.innerHTML = '<option value="">No sources available</option>';
    return;
  }

  sourceList.innerHTML = '<option value="">Select source...</option>';
  filteredSources.forEach((source) => {
    const option = document.createElement("option");
    option.value = source.id;
    option.textContent = source.name;
    sourceList.appendChild(option);
  });
}

function setupEventListeners() {
  // Mode switching
  transmitModeBtn.addEventListener("click", () => switchMode("transmit"));
  receiveModeBtn.addEventListener("click", () => switchMode("receive"));

  // Input source selection
  inputSource.addEventListener("change", async (e) => {
    const sourceId = e.target.value;
    if (sourceId) {
      selectedSource = availableSources.find(
        (source) => source.id === sourceId
      );
      console.log("Selected source:", selectedSource);

      // Hide the source details since we're not using the two-tier approach anymore
      sourceDetails.style.display = "none";

      // Start preview immediately for cameras
      if (selectedSource && selectedSource.type === "camera") {
        await startPreview();
      }
    } else {
      selectedSource = null;
      sourceDetails.style.display = "none";
      stopPreview();
    }
  });

  // Transmit controls
  startTransmitBtn.addEventListener("click", startStreaming);
  stopTransmitBtn.addEventListener("click", stopStreaming);

  // Receive controls
  startReceiveBtn.addEventListener("click", startReceiving);
  stopReceiveBtn.addEventListener("click", stopReceiving);

  // Discovery controls
  refreshDiscoveryBtn.addEventListener("click", refreshDiscovery);
  connectPeerBtn.addEventListener("click", connectToPeer);
}

function switchMode(mode) {
  currentMode = mode;

  // Safety checks for DOM elements
  if (
    !transmitModeBtn ||
    !receiveModeBtn ||
    !transmitConfig ||
    !receiveConfig
  ) {
    console.warn("DOM elements not ready for switchMode");
    return;
  }

  if (mode === "transmit") {
    transmitModeBtn.classList.add("active");
    receiveModeBtn.classList.remove("active");
    transmitConfig.style.display = "block";
    receiveConfig.style.display = "none";
  } else {
    transmitModeBtn.classList.remove("active");
    receiveModeBtn.classList.add("active");
    transmitConfig.style.display = "none";
    receiveConfig.style.display = "block";
  }

  resetVideoDisplay();
  resetProtocolStats();
}

function resetVideoDisplay() {
  // Safety checks for DOM elements
  if (!videoFrame || !noVideo || !frameInfo || !latencyDisplay) {
    console.warn("DOM elements not ready for resetVideoDisplay");
    return;
  }

  videoFrame.style.display = "none";
  noVideo.style.display = "block";
  frameInfo.style.display = "none";
  latencyDisplay.style.display = "none";
}

function resetProtocolStats() {
  // Safety checks for DOM elements
  if (
    !framesSent ||
    !bytesSent ||
    !currentFps ||
    !framesReceived ||
    !bytesReceived ||
    !packetLoss ||
    !activePeers ||
    !avgLatency ||
    !burstPackets
  ) {
    console.warn("DOM elements not ready for resetProtocolStats");
    return;
  }

  framesSent.textContent = "0";
  bytesSent.textContent = "0 B";
  currentFps.textContent = "0";
  framesReceived.textContent = "0";
  bytesReceived.textContent = "0 B";
  packetLoss.textContent = "0%";
  activePeers.textContent = "0";
  avgLatency.textContent = "0 ms";
  burstPackets.textContent = "0";
}

function updateStatus(type, text) {
  // Safety check for DOM element
  if (!statusIndicator) {
    console.warn("statusIndicator not ready for updateStatus");
    return;
  }

  statusIndicator.className = `status-indicator status-${type}`;
  statusIndicator.textContent = text;
}

function showStatus(message, type) {
  updateStatus(type, message);
}

// Preview functions for immediate video display
async function startPreview() {
  if (!selectedSource || selectedSource.type !== "camera") return;

  try {
    console.log("Starting preview for:", selectedSource.name);

    const deviceId =
      selectedSource.deviceId || selectedSource.id.replace("camera:", "");
    const constraints = {
      video: deviceId === "default" ? true : { deviceId: { exact: deviceId } },
      audio: false,
    };

    const stream = await navigator.mediaDevices.getUserMedia(constraints);

    // Set up video preview
    videoFrame.srcObject = stream;
    videoFrame.style.display = "block";
    noVideo.style.display = "none";
    frameInfo.style.display = "block";

    // Start JVID processing once video metadata is loaded
    videoFrame.addEventListener(
      "loadedmetadata",
      () => {
        if (canvasElement && ctx) {
          canvasElement.width = videoFrame.videoWidth || 640;
          canvasElement.height = videoFrame.videoHeight || 480;

          // Start JVID frame capture
          if (jvidProcessor && jvidProcessor.isProcessing) {
            startFrameCapture();
          }
        }
      },
      { once: true }
    );

    console.log("Preview started successfully with JVID processing");
  } catch (error) {
    console.error("Failed to start preview:", error);
    showStatus(`Preview failed: ${error.message}`, "error");
  }
}

function stopPreview() {
  if (videoFrame.srcObject) {
    videoFrame.srcObject.getTracks().forEach((track) => track.stop());
    videoFrame.srcObject = null;
  }
  resetVideoDisplay();
}

async function startStreaming() {
  const sourceId = inputSource.value;
  const codec = txCodec.value;

  if (!sourceId) {
    showStatus("Please select a source first", "error");
    return;
  }

  try {
    showStatus("Starting stream...", "info");

    // Find the selected source from available sources
    const selectedSource = availableSources.find(
      (source) => source.id === sourceId
    );

    if (!selectedSource) {
      throw new Error("Selected source not found");
    }

    let stream;

    // Handle camera sources differently from screen/window sources
    if (selectedSource.type === "camera") {
      // Reuse existing preview stream if available
      if (videoFrame.srcObject) {
        stream = videoFrame.srcObject;
        console.log("Reusing existing camera preview stream");
      } else {
        // Create new camera stream
        const deviceId =
          selectedSource.deviceId || selectedSource.id.replace("camera:", "");
        const constraints = {
          video:
            deviceId === "default" ? true : { deviceId: { exact: deviceId } },
          audio: false, // We'll handle audio separately if needed
        };

        stream = await navigator.mediaDevices.getUserMedia(constraints);
        console.log("Camera stream started:", stream);
      }
    } else {
      // Use desktopCapturer for screen/window sources
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          mandatory: {
            chromeMediaSource: "desktop",
            chromeMediaSourceId: selectedSource.id,
          },
        },
      });
      console.log("Screen/window stream started:", stream);
    }

    // Set up video preview
    const video = document.getElementById("video-frame");
    video.srcObject = stream;
    video.style.display = "block";
    noVideo.style.display = "none";
    frameInfo.style.display = "block";

    // Start TOAST transmission
    const config = {
      multicastAddress: txMulticast.value,
      port: parseInt(txPort.value),
      sessionId: txSession.value,
      fps: parseInt(txFps.value),
      resolution: txResolution.value,
      codec: codec,
      burst: txBurst.checked,
      discovery: txDiscovery.checked,
      source: selectedSource,
      stream: stream,
    };

    const result = await ipcRenderer.invoke("start-toast-transmitter", config);

    if (result.success) {
      isTransmitting = true;
      startTransmitBtn.disabled = true;
      stopTransmitBtn.disabled = false;
      updateStatus(
        "connected",
        `Streaming - Session: ${result.sessionId || txSession.value}`
      );

      // Disable input fields during transmission
      setTransmitInputsEnabled(false);

      // Start capture loop
      startCaptureLoop(config);
    } else {
      throw new Error(result.error || "Failed to start transmission");
    }
  } catch (error) {
    console.error("Failed to start streaming:", error);
    showStatus(`Failed to start streaming: ${error.message}`, "error");

    // Clean up on error
    const video = document.getElementById("video-frame");
    if (video.srcObject) {
      video.srcObject.getTracks().forEach((track) => track.stop());
      video.srcObject = null;
    }
  }
}

async function stopStreaming() {
  try {
    await ipcRenderer.invoke("stop-toast-transmitter");

    isTransmitting = false;
    startTransmitBtn.disabled = false;
    stopTransmitBtn.disabled = true;
    updateStatus("disconnected", "Transmitter stopped");

    // Re-enable input fields
    setTransmitInputsEnabled(true);

    // Stop capture loop
    stopCaptureLoop();

    resetVideoDisplay();
    resetProtocolStats();
  } catch (error) {
    console.error("Error stopping transmitter:", error);
  }
}

async function startReceiving() {
  try {
    const config = {
      multicastAddress: rxMulticast.value,
      port: parseInt(rxPort.value),
      discovery: rxDiscovery.checked,
      peerFilter: rxPeerFilter.value,
      sessionFilter: rxSession.value,
    };

    const result = await ipcRenderer.invoke("start-toast-receiver", config);

    if (result.success) {
      isReceiving = true;
      startReceiveBtn.disabled = true;
      stopReceiveBtn.disabled = false;
      updateStatus("connected", "Receiving");

      // Disable input fields during reception
      setReceiveInputsEnabled(false);
    } else {
      alert(`Failed to start receiver: ${result.error}`);
    }
  } catch (error) {
    console.error("Error starting receiver:", error);
    alert(`Error starting receiver: ${error.message}`);
  }
}

async function stopReceiving() {
  try {
    await ipcRenderer.invoke("stop-toast-receiver");

    isReceiving = false;
    startReceiveBtn.disabled = false;
    stopReceiveBtn.disabled = true;
    updateStatus("disconnected", "Receiver stopped");

    // Re-enable input fields and reset display
    setReceiveInputsEnabled(true);
    resetVideoDisplay();
    resetProtocolStats();
  } catch (error) {
    console.error("Error stopping receiver:", error);
  }
}

async function refreshDiscovery() {
  try {
    await ipcRenderer.invoke("send-toast-discovery");
    refreshDiscoveryBtn.textContent = "Refreshing...";
    setTimeout(() => {
      refreshDiscoveryBtn.textContent = "Refresh Discovery";
    }, 2000);
  } catch (error) {
    console.error("Error refreshing discovery:", error);
  }
}

async function connectToPeer() {
  const selectedPeer = discoveredPeers.value;
  if (!selectedPeer) return;

  try {
    const result = await ipcRenderer.invoke("connect-to-peer", selectedPeer);
    if (result.success) {
      updateStatus("connected", `Connected to ${selectedPeer}`);
    } else {
      alert(`Failed to connect to peer: ${result.error}`);
    }
  } catch (error) {
    console.error("Error connecting to peer:", error);
  }
}

// Video capture loop
function startCaptureLoop(config) {
  if (captureInterval) {
    clearInterval(captureInterval);
  }

  const frameInterval = 1000 / config.fps;

  captureInterval = setInterval(async () => {
    try {
      const captureConfig = {
        source: config.source,
        width: getResolutionWidth(config.resolution),
        height: getResolutionHeight(config.resolution),
        codec: config.codec,
        quality: 80,
      };

      await ipcRenderer.invoke("capture-screen", captureConfig);
    } catch (error) {
      console.error("Capture error:", error);
    }
  }, frameInterval);
}

function stopCaptureLoop() {
  if (captureInterval) {
    clearInterval(captureInterval);
    captureInterval = null;
  }
}

function getResolutionWidth(resolution) {
  const resolutions = {
    LOW_144P: 256,
    SD_240P: 426,
    SD_360P: 640,
    HD_720P: 1280,
  };
  return resolutions[resolution] || 640;
}

function getResolutionHeight(resolution) {
  const resolutions = {
    LOW_144P: 144,
    SD_240P: 240,
    SD_360P: 360,
    HD_720P: 720,
  };
  return resolutions[resolution] || 360;
}

// Helper functions
function setTransmitInputsEnabled(enabled) {
  inputSource.disabled = !enabled;
  sourceList.disabled = !enabled;
  txMulticast.disabled = !enabled;
  txPort.disabled = !enabled;
  txSession.disabled = !enabled;
  txFps.disabled = !enabled;
  txResolution.disabled = !enabled;
  txCodec.disabled = !enabled;
  txBurst.disabled = !enabled;
  txDiscovery.disabled = !enabled;
}

function setReceiveInputsEnabled(enabled) {
  rxMulticast.disabled = !enabled;
  rxPort.disabled = !enabled;
  rxDiscovery.disabled = !enabled;
  rxPeerFilter.disabled = !enabled;
  rxSession.disabled = !enabled;
}

// Zero-API JSON message routing - eliminates traditional callback APIs
ipcRenderer.on("json-message", (event, message) => {
  switch (message.type) {
    case "transmitter_started":
      console.log("TOAST transmitter started:", message);
      updateStatus(
        "transmitting",
        `Transmitting (Session: ${message.session_id})`
      );
      txSession.value = message.session_id;
      break;

    case "receiver_started":
      console.log("TOAST receiver started:", message);
      updateStatus(
        "connected",
        `Receiving on ${message.config.multicastAddress}:${message.config.port}`
      );
      break;

    case "transmitter_stopped":
      console.log("TOAST transmitter stopped");
      isTransmitting = false;
      startTransmitBtn.disabled = false;
      stopTransmitBtn.disabled = true;
      updateStatus("disconnected", "Transmitter stopped");
      setTransmitInputsEnabled(true);
      resetProtocolStats();
      break;

    case "receiver_stopped":
      console.log("TOAST receiver stopped");
      isReceiving = false;
      startReceiveBtn.disabled = false;
      stopReceiveBtn.disabled = true;
      updateStatus("disconnected", "Receiver stopped");
      setReceiveInputsEnabled(true);
      resetVideoDisplay();
      resetProtocolStats();
      break;

    case "transmitter_error":
      console.error("Transmitter error:", message.error);
      updateStatus("error", `Transmitter error: ${message.error}`);
      break;

    case "receiver_error":
      console.error("Receiver error:", message.error);
      updateStatus("error", `Receiver error: ${message.error}`);
      break;

    case "peer_discovered":
      console.log("Peer discovered:", message.peer);
      addPeerToDiscovery(message.peer);
      break;

    case "peer_heartbeat":
      console.log("Peer heartbeat:", message);
      updatePeerLastSeen(message);
      break;

    case "video_frame_received":
      console.log("Video frame received:", message);
      displayVideoFrame(message);
      break;

    case "video_frame_sent":
      console.log("Video frame sent:", message);
      updateTransmitterStats(message);
      break;

    case "discovery_sent":
      console.log("Discovery sent:", message);
      break;

    case "capture_error":
      console.error("Capture error:", message.error);
      updateStatus("error", `Capture error: ${message.error}`);
      break;

    case "sources_available":
      console.log("Sources available:", message.sources);
      availableSources = message.sources;
      if (inputSource.value) {
        populateSourceList(inputSource.value);
      }
      break;

    default:
      console.log("Unknown message type:", message.type, message);
  }
});

// Helper functions for JSON message processing
function addPeerToDiscovery(peer) {
  // Clear existing options except the placeholder
  discoveredPeers.innerHTML =
    '<option value="">Select a peer to connect</option>';

  const option = document.createElement("option");
  option.value = peer.session_id;
  option.textContent = `${peer.device_name || peer.session_id} (${
    peer.address
  }:${peer.port})`;
  discoveredPeers.appendChild(option);

  // Store peer info
  discoveredPeersList.set(peer.session_id, peer);

  // Enable connect button
  connectPeerBtn.disabled = false;
}

function updateTransmitterStats(message) {
  framesSent.textContent = (parseInt(framesSent.textContent) || 0) + 1;
  bytesSent.textContent = formatBytes(
    (parseInt(bytesSent.textContent.replace(/[^0-9]/g, "")) || 0) + message.size
  );

  // Update compression info if available
  if (message.compression_ratio && message.original_size) {
    const compressionPercent = ((1 - message.compression_ratio) * 100).toFixed(
      1
    );
    const codecInfo = message.codec || "raw";

    // Update frame info to show compression stats
    if (frameInfo) {
      frameInfo.textContent = `Codec: ${codecInfo} | Compression: ${compressionPercent}% | ${formatBytes(
        message.original_size
      )} → ${formatBytes(message.size)}`;
      frameInfo.style.display = "block";
    }
  }

  // Calculate FPS
  frameCount++;
  const now = Date.now();
  const timeDiff = (now - lastFrameTime) / 1000;
  if (timeDiff >= 1.0) {
    const fps = frameCount / timeDiff;
    currentFps.textContent = fps.toFixed(1);
    frameCount = 0;
    lastFrameTime = now;
  }
}

function displayVideoFrame(message) {
  // Display the received frame
  if (message.data) {
    videoFrame.src = `data:image/png;base64,${message.data}`;
    videoFrame.style.display = "block";
    noVideo.style.display = "none";

    // Update frame info with detailed information
    frameInfo.style.display = "block";
    let infoText = `Frame: ${message.width}x${message.height} | Format: ${message.format}`;

    // Add codec information if available
    if (message.codec) {
      infoText += ` | Codec: ${message.codec}`;
    }

    // Add compression information if available
    if (message.compression_ratio && message.original_size) {
      const compressionPercent = (
        (1 - message.compression_ratio) *
        100
      ).toFixed(1);
      infoText += ` | Compression: ${compressionPercent}%`;
    }

    frameInfo.textContent = infoText;

    // Update latency if available
    if (message.timestamp_us) {
      const latency = (Date.now() * 1000 - message.timestamp_us) / 1000;
      latencyDisplay.style.display = "block";
      latencyDisplay.textContent = `${latency.toFixed(1)} ms`;
    }

    // Update receive stats
    framesReceived.textContent =
      (parseInt(framesReceived.textContent) || 0) + 1;
    bytesReceived.textContent = formatBytes(
      (parseInt(bytesReceived.textContent.replace(/[^0-9]/g, "")) || 0) +
        message.data.length
    );
  }
}

function updatePeerLastSeen(message) {
  // Update peer last seen timestamp
  console.log("Updating peer last seen:", message.session_id);
}

function formatBytes(bytes) {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

// Initialize JVID processor
function initializeJVIDProcessor() {
  jvidProcessor = new JVIDProcessor();

  // Start JVID processing
  jvidProcessor.start();

  console.log("🎥 JVID Processor initialized for 60fps RGB matrix processing");
}

// Start video processing at 60fps
function startVideoProcessing() {
  if (!videoElement || !jvidProcessor) return;

  console.log(
    "🎥 Starting 60fps video processing with JVID RGB matrix splitting"
  );

  // Create canvas for frame capture
  if (!canvasElement) {
    canvasElement = document.createElement("canvas");
    ctx = canvasElement.getContext("2d");
  }

  // Set canvas size to match video
  canvasElement.width = videoElement.videoWidth || 640;
  canvasElement.height = videoElement.videoHeight || 480;

  // Start 60fps capture loop
  startFrameCapture();
}

function startFrameCapture() {
  let lastFrameTime = 0;
  const targetFPS = 60;
  const frameInterval = 1000 / targetFPS; // 16.67ms per frame

  function captureFrame(timestamp) {
    if (!jvidProcessor.isProcessing) return;

    // Throttle to 60fps
    if (timestamp - lastFrameTime >= frameInterval) {
      processVideoFrame(timestamp);
      lastFrameTime = timestamp;
    }

    // Continue capture loop
    requestAnimationFrame(captureFrame);
  }

  // Start the capture loop
  requestAnimationFrame(captureFrame);
}

function processVideoFrame(timestamp) {
  if (!videoElement || !canvasElement || !ctx) return;

  try {
    // Draw video frame to canvas
    ctx.drawImage(
      videoElement,
      0,
      0,
      canvasElement.width,
      canvasElement.height
    );

    // Get image data for RGB matrix processing
    const imageData = ctx.getImageData(
      0,
      0,
      canvasElement.width,
      canvasElement.height
    );

    // Convert timestamp to microseconds for TOAST clock
    const toastTimestamp = timestamp * 1000;

    // Process frame with JVID processor
    const jvidMessages = jvidProcessor.processFrame(imageData, toastTimestamp);

    if (jvidMessages && jvidMessages.length > 0) {
      // Send RGB lanes via TOAST protocol
      sendRGBLanes(jvidMessages);

      // Update performance metrics
      updatePerformanceMetrics(jvidMessages);
    }
  } catch (error) {
    console.error("Error processing video frame:", error);
  }
}

function sendRGBLanes(jvidMessages) {
  if (!window.electronAPI) return;

  // Send each RGB lane separately for parallel processing
  jvidMessages.forEach((message) => {
    window.electronAPI.sendMessage({
      type: "jvid_rgb_lane",
      data: message,
    });
  });

  // Update network statistics
  performanceMonitor.networkStats.rgbLanesSent += jvidMessages.length;
  performanceMonitor.networkStats.pnbtrPredictions += jvidMessages.filter(
    (m) => m.pnbtr.confidence > 0.7
  ).length;
}

function updatePerformanceMetrics(jvidMessages) {
  const currentTime = performance.now();
  performanceMonitor.frameCount++;

  // Calculate actual FPS
  if (currentTime - performanceMonitor.lastTime >= 1000) {
    performanceMonitor.actualFPS = performanceMonitor.frameCount;
    performanceMonitor.frameCount = 0;
    performanceMonitor.lastTime = currentTime;

    // Get JVID processor stats
    const jvidStats = jvidProcessor.getStats();

    // Update UI with performance metrics
    updatePerformanceUI(jvidStats);
  }
}

function updatePerformanceUI(jvidStats) {
  // Update compression stats
  if (compressionStatsDiv) {
    compressionStatsDiv.innerHTML = `
            <div style="font-family: monospace; font-size: 12px; color: #00ff00; background: rgba(0,0,0,0.7); padding: 10px; border-radius: 5px;">
                <strong>🎥 JVID Performance (60fps RGB Matrix Processing)</strong><br>
                <strong>Actual FPS:</strong> ${
                  performanceMonitor.actualFPS
                }/60<br>
                <strong>Frames Processed:</strong> ${
                  jvidStats.framesProcessed
                }<br>
                <strong>Avg Processing Time:</strong> ${jvidStats.averageProcessingTime.toFixed(
                  2
                )}ms<br>
                <strong>RGB Compression:</strong> ${jvidStats.rgbCompressionRatio.toFixed(
                  2
                )}x<br>
                <strong>RGB Lanes Sent:</strong> ${
                  performanceMonitor.networkStats.rgbLanesSent
                }<br>
                <strong>PNBTR Predictions:</strong> ${
                  performanceMonitor.networkStats.pnbtrPredictions
                }<br>
                <strong>Dropped Frames:</strong> ${jvidStats.droppedFrames}<br>
                <strong>PNBTR Accuracy:</strong> ${(
                  jvidStats.pnbtrPredictionAccuracy * 100
                ).toFixed(1)}%
            </div>
        `;
  }

  // Update video processing stats
  if (videoProcessingStatsDiv) {
    videoProcessingStatsDiv.innerHTML = `
            <div style="font-family: monospace; font-size: 11px; color: #00ccff; background: rgba(0,0,0,0.7); padding: 8px; border-radius: 4px;">
                <strong>📊 JVID RGB Matrix Details</strong><br>
                <strong>R Lane:</strong> ${Math.floor(
                  performanceMonitor.networkStats.rgbLanesSent / 3
                )} frames<br>
                <strong>G Lane:</strong> ${Math.floor(
                  performanceMonitor.networkStats.rgbLanesSent / 3
                )} frames<br>
                <strong>B Lane:</strong> ${Math.floor(
                  performanceMonitor.networkStats.rgbLanesSent / 3
                )} frames<br>
                <strong>Audio Clock Sync:</strong> 48kHz (${
                  jvidStats.framesProcessed * 800
                } samples)<br>
                <strong>TOAST Clock Drift:</strong> ${jvidProcessor.toastClock.clockDrift.toFixed(
                  2
                )}μs<br>
                <strong>Motion Vectors:</strong> ${
                  performanceMonitor.networkStats.pnbtrPredictions * 3
                } calculated<br>
                <strong>Prediction Confidence:</strong> ${(
                  jvidStats.pnbtrPredictionAccuracy * 100
                ).toFixed(1)}%
            </div>
        `;
  }
}

// Add JVID processor controls
function addJVIDControls() {
  const controlsDiv = document.createElement("div");
  controlsDiv.innerHTML = `
        <div style="margin-top: 10px; padding: 10px; background: rgba(0,0,0,0.8); border-radius: 5px;">
            <h3 style="color: #00ff00; margin: 0 0 10px 0;">🎥 JVID Controls (60fps RGB Matrix)</h3>
            <div>
                <label style="color: white; display: inline-block; width: 150px;">Target FPS:</label>
                <select id="jvid-fps-select" style="width: 100px;">
                    <option value="30">30 FPS</option>
                    <option value="60" selected>60 FPS</option>
                    <option value="120">120 FPS</option>
                </select>
            </div>
            <div style="margin-top: 5px;">
                <label style="color: white; display: inline-block; width: 150px;">PNBTR Confidence:</label>
                <input type="range" id="pnbtr-confidence" min="0" max="1" step="0.1" value="0.7" style="width: 100px;">
                <span id="confidence-value" style="color: white; margin-left: 5px;">0.7</span>
            </div>
            <div style="margin-top: 5px;">
                <label style="color: white; display: inline-block; width: 150px;">Prediction Horizon:</label>
                <input type="range" id="prediction-horizon" min="4" max="32" step="4" value="16" style="width: 100px;">
                <span id="horizon-value" style="color: white; margin-left: 5px;">16 frames</span>
            </div>
            <div style="margin-top: 10px;">
                <button id="reset-jvid-stats" style="background: #333; color: white; border: 1px solid #555; padding: 5px 10px; border-radius: 3px; cursor: pointer;">
                    Reset Statistics
                </button>
            </div>
        </div>
    `;

  document.body.appendChild(controlsDiv);

  // Add event listeners
  document.getElementById("jvid-fps-select").addEventListener("change", (e) => {
    const newFPS = parseInt(e.target.value);
    if (jvidProcessor) {
      jvidProcessor.videoFrameRate = newFPS;
      jvidProcessor.samplesPerFrame = jvidProcessor.audioSampleRate / newFPS;
      console.log(`🎥 JVID FPS changed to ${newFPS}`);
    }
  });

  document.getElementById("pnbtr-confidence").addEventListener("input", (e) => {
    const confidence = parseFloat(e.target.value);
    if (jvidProcessor) {
      jvidProcessor.confidenceThreshold = confidence;
    }
    document.getElementById("confidence-value").textContent =
      confidence.toFixed(1);
  });

  document
    .getElementById("prediction-horizon")
    .addEventListener("input", (e) => {
      const horizon = parseInt(e.target.value);
      if (jvidProcessor) {
        jvidProcessor.predictionHorizon = horizon;
      }
      document.getElementById(
        "horizon-value"
      ).textContent = `${horizon} frames`;
    });

  document.getElementById("reset-jvid-stats").addEventListener("click", () => {
    if (jvidProcessor) {
      jvidProcessor.stats = {
        framesProcessed: 0,
        averageProcessingTime: 0,
        rgbCompressionRatio: 0,
        pnbtrPredictionAccuracy: 0,
        droppedFrames: 0,
      };
      performanceMonitor.networkStats = {
        rgbLanesSent: 0,
        pnbtrPredictions: 0,
        averageLatency: 0,
      };
      console.log("🎥 JVID statistics reset");
    }
  });
}
