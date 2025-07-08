const { app, BrowserWindow, ipcMain, desktopCapturer } = require("electron");
const path = require("path");
const fs = require("fs");
const ffmpeg = require("fluent-ffmpeg");
const ffmpegStatic = require("ffmpeg-static");
const { TOASTv2Transport } = require("./toast-transport");

// Set FFmpeg binary path
ffmpeg.setFfmpegPath(ffmpegStatic);

let mainWindow;
let transmitter;
let receiver;

// Zero-API JSON message routing - eliminates traditional APIs
const messageRouter = {
  handlers: new Map(),

  route(message) {
    const handler = this.handlers.get(message.type);
    if (handler) {
      handler(message);
    }
  },

  on(messageType, handler) {
    this.handlers.set(messageType, handler);
  },

  send(message) {
    if (mainWindow && mainWindow.webContents) {
      mainWindow.webContents.send("json-message", message);
    }
  },
};

// Video processing utility functions
const VideoProcessor = {
  // Convert buffer to video using FFmpeg
  async processWithFFmpeg(inputBuffer, codec, options = {}) {
    return new Promise((resolve, reject) => {
      const tempInputPath = path.join(__dirname, "temp_input.png");
      const tempOutputPath = path.join(
        __dirname,
        `temp_output.${codec === "h264" ? "mp4" : "webm"}`
      );

      try {
        // Write input buffer to temporary file
        fs.writeFileSync(tempInputPath, inputBuffer);

        let command = ffmpeg(tempInputPath)
          .inputOptions(["-f", "image2"])
          .outputOptions([
            "-f",
            codec === "h264" ? "mp4" : "webm",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "15", // Frame rate
            "-t",
            "0.1", // Duration for single frame
          ]);

        if (codec === "h264") {
          command = command
            .videoCodec("libx264")
            .outputOptions([
              "-preset",
              "ultrafast",
              "-tune",
              "zerolatency",
              "-crf",
              options.quality || "23",
            ]);
        } else if (codec === "vp8") {
          command = command
            .videoCodec("libvpx")
            .outputOptions([
              "-deadline",
              "realtime",
              "-cpu-used",
              "8",
              "-crf",
              options.quality || "10",
            ]);
        }

        command
          .output(tempOutputPath)
          .on("end", () => {
            try {
              const outputBuffer = fs.readFileSync(tempOutputPath);

              // Cleanup temporary files
              fs.unlinkSync(tempInputPath);
              fs.unlinkSync(tempOutputPath);

              resolve(outputBuffer);
            } catch (error) {
              reject(new Error(`Failed to read output: ${error.message}`));
            }
          })
          .on("error", (error) => {
            // Cleanup on error
            try {
              if (fs.existsSync(tempInputPath)) fs.unlinkSync(tempInputPath);
              if (fs.existsSync(tempOutputPath)) fs.unlinkSync(tempOutputPath);
            } catch (cleanupError) {
              console.error("Cleanup error:", cleanupError);
            }
            reject(new Error(`FFmpeg processing failed: ${error.message}`));
          })
          .run();
      } catch (error) {
        reject(new Error(`Failed to process with FFmpeg: ${error.message}`));
      }
    });
  },

  // Resize image using FFmpeg
  async resizeImage(inputBuffer, width, height) {
    return new Promise((resolve, reject) => {
      const tempInputPath = path.join(__dirname, "temp_resize_input.png");
      const tempOutputPath = path.join(__dirname, "temp_resize_output.png");

      try {
        fs.writeFileSync(tempInputPath, inputBuffer);

        ffmpeg(tempInputPath)
          .outputOptions(["-vf", `scale=${width}:${height}`, "-f", "png"])
          .output(tempOutputPath)
          .on("end", () => {
            try {
              const outputBuffer = fs.readFileSync(tempOutputPath);
              fs.unlinkSync(tempInputPath);
              fs.unlinkSync(tempOutputPath);
              resolve(outputBuffer);
            } catch (error) {
              reject(error);
            }
          })
          .on("error", (error) => {
            try {
              if (fs.existsSync(tempInputPath)) fs.unlinkSync(tempInputPath);
              if (fs.existsSync(tempOutputPath)) fs.unlinkSync(tempOutputPath);
            } catch (cleanupError) {
              console.error("Cleanup error:", cleanupError);
            }
            reject(error);
          })
          .run();
      } catch (error) {
        reject(error);
      }
    });
  },
};

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      enableRemoteModule: true,
    },
    title: "JAMCam - JVID Video Streaming",
    icon: path.join(__dirname, "../assets/jamcam-icon.png"),
  });

  mainWindow.loadFile(path.join(__dirname, "renderer/index.html"));

  if (process.env.NODE_ENV === "development") {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

// JVID Message structure
class JVIDMessage {
  constructor() {
    this.timestamp_us = Date.now() * 1000; // Current time in microseconds
    this.sequence_number = 0;
    this.session_id = "";
    this.video_info = {
      resolution: "LOW_144P",
      quality: "FAST",
      format: "BASE64_JPEG",
      frame_width: 256,
      frame_height: 144,
      fps_target: 15,
      is_keyframe: false,
      stream_id: 0,
    };
    this.frame_data = {
      frame_base64: "",
      compressed_size: 0,
      original_size: 0,
      compression_ratio: 0.0,
    };
    this.timing_info = {
      capture_timestamp_us: 0,
      encode_timestamp_us: 0,
      send_timestamp_us: 0,
      encode_duration_us: 0,
      expected_decode_us: 200,
    };
    this.integrity = {
      checksum: 0,
      is_predicted: false,
      prediction_confidence: 0,
    };
  }

  toJSON() {
    return {
      t: "vid", // JVID message type
      id: "jvid",
      seq: this.sequence_number,
      sid: this.session_id,
      ts: this.timestamp_us,
      vi: this.video_info,
      fd: this.frame_data,
      ti: this.timing_info,
      in: this.integrity,
    };
  }

  static fromJSON(json) {
    const message = new JVIDMessage();
    message.sequence_number = json.seq || 0;
    message.session_id = json.sid || "";
    message.timestamp_us = json.ts || 0;
    message.video_info = json.vi || message.video_info;
    message.frame_data = json.fd || message.frame_data;
    message.timing_info = json.ti || message.timing_info;
    message.integrity = json.in || message.integrity;
    return message;
  }
}

// Video Transmitter (Encoder + TOAST UDP Send)
class JAMCamTransmitter {
  constructor() {
    this.isRunning = false;
    this.sequenceNumber = 0;
    this.sessionId = "jamcam_" + Date.now();
    this.targetFPS = 15;
    this.toastTransport = null;
    this.stats = {
      framesCaptured: 0,
      framesEncoded: 0,
      framesSent: 0,
      averageEncodeTime: 0,
      currentBitrate: 0,
    };
  }

  async start(multicastAddress = "239.255.77.77", port = 7777) {
    if (this.isRunning) return;

    this.isRunning = true;
    console.log("JAMCam Transmitter starting with TOAST UDP...");

    // Initialize TOAST v2 transport
    this.toastTransport = new TOASTv2Transport();

    // Set up error callback
    this.toastTransport.onError = (error) => {
      console.error("TOAST Transport Error:", error);
      mainWindow.webContents.send("transmitter-error", error);
    };

    // Initialize transport
    const sessionId = Math.floor(Math.random() * 0xffffffff);
    const success = await this.toastTransport.initialize(
      multicastAddress,
      port,
      sessionId
    );

    if (!success) {
      this.isRunning = false;
      throw new Error("Failed to initialize TOAST transport");
    }

    // Start video capture loop
    this.captureLoop();

    mainWindow.webContents.send("transmitter-started", {
      sessionId: this.sessionId,
      multicastAddress: multicastAddress,
      port: port,
    });
  }

  async captureLoop() {
    const frameInterval = 1000 / this.targetFPS;

    while (this.isRunning) {
      const startTime = Date.now();

      try {
        await this.captureAndSendFrame();
        this.stats.framesCaptured++;
      } catch (error) {
        console.error("Frame capture error:", error);
      }

      // Maintain target FPS
      const elapsedTime = Date.now() - startTime;
      const sleepTime = Math.max(0, frameInterval - elapsedTime);

      if (sleepTime > 0) {
        await new Promise((resolve) => setTimeout(resolve, sleepTime));
      }

      // Send stats update to renderer
      if (this.stats.framesCaptured % 30 === 0) {
        // Every 30 frames
        mainWindow.webContents.send("transmitter-stats", this.stats);
      }
    }
  }

  async captureAndSendFrame() {
    const encodeStartTime = Date.now() * 1000; // microseconds

    // Create JVID message
    const jvidMessage = new JVIDMessage();
    jvidMessage.sequence_number = this.sequenceNumber++;
    jvidMessage.session_id = this.sessionId;
    jvidMessage.timing_info.capture_timestamp_us = encodeStartTime;
    jvidMessage.timing_info.encode_timestamp_us = encodeStartTime;

    // Generate test frame (solid color with sequence number)
    const testFrame = await this.generateTestFrame(jvidMessage.sequence_number);
    jvidMessage.frame_data.frame_base64 = testFrame.base64;
    jvidMessage.frame_data.compressed_size = testFrame.size;
    jvidMessage.frame_data.original_size = 256 * 144 * 3; // RGB
    jvidMessage.frame_data.compression_ratio =
      jvidMessage.frame_data.original_size / testFrame.size;

    const encodeEndTime = Date.now() * 1000;
    jvidMessage.timing_info.encode_duration_us =
      encodeEndTime - encodeStartTime;
    jvidMessage.timing_info.send_timestamp_us = encodeEndTime;

    // Send via TOAST UDP transport
    if (this.toastTransport) {
      const success = this.toastTransport.sendJVIDFrame(jvidMessage);
      if (success) {
        this.stats.framesSent++;
      }
    }

    this.stats.framesEncoded++;
    this.stats.averageEncodeTime =
      jvidMessage.timing_info.encode_duration_us / 1000; // ms
  }

  async generateTestFrame(sequenceNumber) {
    // Generate a simple test pattern as base64 JPEG
    // For now, create a minimal JPEG-like structure or use a library

    const width = 256;
    const height = 144;

    // Create a simple pattern - alternating colors based on sequence
    const colorIndex = sequenceNumber % 8;
    const colors = [
      "#FF0000",
      "#00FF00",
      "#0000FF",
      "#FFFF00",
      "#FF00FF",
      "#00FFFF",
      "#FFFFFF",
      "#808080",
    ];

    // Generate SVG and convert to base64 (simpler than JPEG for test)
    const svgContent = `
            <svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
                <rect width="100%" height="100%" fill="${colors[colorIndex]}"/>
                <text x="10" y="30" font-family="Arial" font-size="20" fill="white">
                    Frame ${sequenceNumber}
                </text>
                <text x="10" y="60" font-family="Arial" font-size="16" fill="white">
                    ${new Date().toLocaleTimeString()}
                </text>
                <text x="10" y="90" font-family="Arial" font-size="14" fill="white">
                    TOAST UDP Test
                </text>
            </svg>
        `;

    // Convert SVG to base64 (browsers can display SVG as image)
    const base64 = Buffer.from(svgContent).toString("base64");

    return {
      base64: base64,
      size: base64.length,
    };
  }

  stop() {
    this.isRunning = false;
    if (this.toastTransport) {
      this.toastTransport.shutdown();
      this.toastTransport = null;
    }
    console.log("JAMCam Transmitter stopped");
    mainWindow.webContents.send("transmitter-stopped");
  }
}

// Video Receiver (TOAST UDP Receive + Decoder)
class JAMCamReceiver {
  constructor() {
    this.isRunning = false;
    this.toastTransport = null;
    this.stats = {
      framesReceived: 0,
      framesDecoded: 0,
      averageDecodeTime: 0,
      packetLossRate: 0,
      lastSequenceNumber: -1,
      discoveredPeers: 0,
    };
  }

  async start(multicastAddress = "239.255.77.77", port = 7777) {
    if (this.isRunning) return;

    this.isRunning = true;
    console.log("JAMCam Receiver starting with TOAST UDP...");

    // Initialize TOAST v2 transport
    this.toastTransport = new TOASTv2Transport();

    // Set up callbacks
    this.toastTransport.onVideoFrame = (jvidData, rinfo) => {
      this.handleIncomingFrame(jvidData, rinfo);
    };

    this.toastTransport.onDiscovery = (discoveryData, rinfo) => {
      console.log("Discovered peer:", discoveryData, "from", rinfo.address);
      this.stats.discoveredPeers++;
      mainWindow.webContents.send("peer-discovered", {
        peerId: discoveryData.session_id,
        address: rinfo.address,
        port: rinfo.port,
        capabilities: discoveryData.capabilities,
      });
    };

    this.toastTransport.onHeartbeat = (heartbeatData, rinfo) => {
      console.log("Received heartbeat from peer:", heartbeatData.session_id);
      mainWindow.webContents.send("peer-heartbeat", {
        peerId: heartbeatData.session_id,
        address: rinfo.address,
      });
    };

    this.toastTransport.onError = (error) => {
      console.error("TOAST Transport Error:", error);
      mainWindow.webContents.send("receiver-error", error);
    };

    // Initialize transport
    const sessionId = Math.floor(Math.random() * 0xffffffff);
    const success = await this.toastTransport.initialize(
      multicastAddress,
      port,
      sessionId
    );

    if (!success) {
      throw new Error("Failed to initialize TOAST transport");
    }

    console.log(`JAMCam Receiver listening on ${multicastAddress}:${port}`);
    mainWindow.webContents.send("receiver-connected", {
      multicastAddress: multicastAddress,
      port: port,
    });
  }

  async handleIncomingFrame(jvidData, rinfo) {
    const decodeStartTime = Date.now() * 1000; // microseconds

    try {
      // Process JVID message
      if (jvidData.t === "vid" && jvidData.id === "jvid") {
        const jvidMessage = JVIDMessage.fromJSON(jvidData);

        this.stats.framesReceived++;

        // Check for packet loss
        if (this.stats.lastSequenceNumber >= 0) {
          const expectedSeq = this.stats.lastSequenceNumber + 1;
          if (jvidMessage.sequence_number > expectedSeq) {
            const lostFrames = jvidMessage.sequence_number - expectedSeq;
            console.log(`Packet loss detected: ${lostFrames} frames lost`);
            this.stats.packetLossRate =
              lostFrames / jvidMessage.sequence_number;
          }
        }
        this.stats.lastSequenceNumber = jvidMessage.sequence_number;

        // Decode frame
        await this.decodeAndDisplayFrame(jvidMessage);

        const decodeEndTime = Date.now() * 1000;
        const decodeTime = (decodeEndTime - decodeStartTime) / 1000; // ms
        this.stats.averageDecodeTime = decodeTime;
        this.stats.framesDecoded++;

        // Send frame to renderer for display
        mainWindow.webContents.send("frame-received", {
          sequenceNumber: jvidMessage.sequence_number,
          base64Image:
            "data:image/jpeg;base64," + jvidMessage.frame_data.frame_base64,
          timing: jvidMessage.timing_info,
          stats: this.stats,
          source: {
            address: rinfo.address,
            port: rinfo.port,
          },
        });
      }
    } catch (error) {
      console.error("Frame decode error:", error);
    }
  }

  async decodeAndDisplayFrame(jvidMessage) {
    // In a real implementation, this would decode the frame
    // For now, we just pass the base64 data to the renderer
    // The browser can handle base64 JPEG directly

    // Calculate end-to-end latency
    const currentTime = Date.now() * 1000;
    const endToEndLatency =
      currentTime - jvidMessage.timing_info.capture_timestamp_us;

    console.log(
      `Frame ${jvidMessage.sequence_number}: ${(endToEndLatency / 1000).toFixed(
        2
      )}ms latency`
    );
  }

  stop() {
    this.isRunning = false;
    if (this.toastTransport) {
      this.toastTransport.shutdown();
      this.toastTransport = null;
    }
    console.log("JAMCam Receiver stopped");
    mainWindow.webContents.send("receiver-stopped");
  }
}

// Initialize transmitter and receiver
transmitter = new TOASTv2Transport();
receiver = new TOASTv2Transport();

// Zero-API JSON message routing for transmitter
transmitter.on("discovery_announce", (message) => {
  messageRouter.send({
    type: "peer_discovered",
    peer: {
      device_name: message.device_name,
      session_id: message.session_id,
      capabilities: message.capabilities,
      timestamp: message.timestamp_us,
    },
  });
});

transmitter.on("heartbeat", (message) => {
  messageRouter.send({
    type: "peer_heartbeat",
    session_id: message.session_id,
    timestamp: message.timestamp_us,
  });
});

// Zero-API JSON message routing for receiver
receiver.on("jvid_frame", (message) => {
  messageRouter.send({
    type: "video_frame_received",
    timestamp: message.timestamp_us,
    width: message.width,
    height: message.height,
    format: message.format,
    data: message.data,
  });
});

receiver.on("discovery_announce", (message) => {
  messageRouter.send({
    type: "peer_discovered",
    peer: {
      device_name: message.device_name,
      session_id: message.session_id,
      capabilities: message.capabilities,
      timestamp: message.timestamp_us,
    },
  });
});

// IPC handlers using zero-API JSON message routing
ipcMain.handle("start-toast-transmitter", async (event, config) => {
  try {
    await transmitter.start(config.multicastAddress, config.port);

    // Send JSON message instead of traditional callback
    messageRouter.send({
      type: "transmitter_started",
      config: config,
      session_id: transmitter.getSessionId(),
      timestamp: Date.now(),
    });

    return { success: true, sessionId: transmitter.getSessionId() };
  } catch (error) {
    messageRouter.send({
      type: "transmitter_error",
      error: error.message,
      timestamp: Date.now(),
    });
    return { success: false, error: error.message };
  }
});

ipcMain.handle("stop-toast-transmitter", async () => {
  try {
    transmitter.stop();

    messageRouter.send({
      type: "transmitter_stopped",
      timestamp: Date.now(),
    });

    return { success: true };
  } catch (error) {
    messageRouter.send({
      type: "transmitter_error",
      error: error.message,
      timestamp: Date.now(),
    });
    return { success: false, error: error.message };
  }
});

ipcMain.handle("start-toast-receiver", async (event, config) => {
  try {
    await receiver.start(config.multicastAddress, config.port);

    messageRouter.send({
      type: "receiver_started",
      config: config,
      session_id: receiver.getSessionId(),
      timestamp: Date.now(),
    });

    return { success: true, sessionId: receiver.getSessionId() };
  } catch (error) {
    messageRouter.send({
      type: "receiver_error",
      error: error.message,
      timestamp: Date.now(),
    });
    return { success: false, error: error.message };
  }
});

ipcMain.handle("stop-toast-receiver", async () => {
  try {
    receiver.stop();

    messageRouter.send({
      type: "receiver_stopped",
      timestamp: Date.now(),
    });

    return { success: true };
  } catch (error) {
    messageRouter.send({
      type: "receiver_error",
      error: error.message,
      timestamp: Date.now(),
    });
    return { success: false, error: error.message };
  }
});

ipcMain.handle("get-toast-stats", () => {
  return {
    transmitter: transmitter.getStats(),
    receiver: receiver.getStats(),
  };
});

ipcMain.handle("send-toast-discovery", async () => {
  try {
    const success = transmitter.sendDiscovery();

    messageRouter.send({
      type: "discovery_sent",
      success: success,
      timestamp: Date.now(),
    });

    return { success };
  } catch (error) {
    messageRouter.send({
      type: "discovery_error",
      error: error.message,
      timestamp: Date.now(),
    });
    return { success: false, error: error.message };
  }
});

// IPC handler for getting available sources
ipcMain.handle("get-available-sources", async () => {
  console.log("get-available-sources IPC handler called!");
  try {
    const formattedSources = [];

    // Only add primary screen - no individual windows/documents
    const screenSources = await desktopCapturer.getSources({
      types: ["screen"],
      thumbnailSize: { width: 150, height: 150 },
    });

    if (screenSources.length > 0) {
      formattedSources.push({
        id: screenSources[0].id,
        name: "Primary Screen",
        type: "screen",
        thumbnail: screenSources[0].thumbnail
          ? screenSources[0].thumbnail.toDataURL()
          : null,
      });
    }

    // Get real camera devices using webContents to access navigator.mediaDevices
    if (mainWindow && mainWindow.webContents) {
      try {
        const cameraDevices = await mainWindow.webContents.executeJavaScript(`
          (async () => {
            try {
              console.log('Requesting camera permission...');
              // Request camera permissions first
              const stream = await navigator.mediaDevices.getUserMedia({ video: true });
              console.log('Camera permission granted, enumerating devices...');
              // Stop the stream immediately, we just needed permissions
              stream.getTracks().forEach(track => track.stop());
              
              // Now enumerate devices
              const devices = await navigator.mediaDevices.enumerateDevices();
              const cameras = devices
                .filter(device => device.kind === 'videoinput')
                .map(device => ({
                  id: device.deviceId,
                  name: device.label || 'Camera ' + device.deviceId.slice(0, 8),
                  type: 'camera',
                  groupId: device.groupId
                }));
              console.log('Found cameras:', cameras);
              return cameras;
            } catch (error) {
              console.error('Camera enumeration error:', error);
              return [];
            }
          })()
        `);

        console.log(
          `Camera enumeration completed. Found ${cameraDevices.length} cameras:`,
          cameraDevices
        );

        // Add real camera devices to the sources
        cameraDevices.forEach((camera) => {
          formattedSources.push({
            id: `camera:${camera.id}`,
            name: camera.name,
            type: "camera",
            thumbnail: null,
            deviceId: camera.id,
            groupId: camera.groupId,
          });
        });
      } catch (cameraError) {
        console.warn("Failed to enumerate camera devices:", cameraError);
        // Add fallback camera option
        formattedSources.push({
          id: "camera:default",
          name: "Default Camera (Permission Required)",
          type: "camera",
          thumbnail: null,
        });
      }
    } else {
      console.warn("mainWindow not available for camera enumeration");
    }

    console.log(
      `Returning ${formattedSources.length} total sources:`,
      formattedSources.map((s) => s.name)
    );

    messageRouter.send({
      type: "sources_available",
      sources: formattedSources,
      timestamp: Date.now(),
    });

    return formattedSources;
  } catch (error) {
    console.error("Failed to get available sources:", error);
    messageRouter.send({
      type: "sources_error",
      error: error.message,
      timestamp: Date.now(),
    });
    return [];
  }
});

// Enhanced video capture handler with FFmpeg integration
ipcMain.handle("capture-screen", async (event, config) => {
  try {
    let frameData;
    let width, height;

    if (config.source) {
      if (config.source.type === "camera") {
        // Handle webcam capture (placeholder for now)
        throw new Error(
          "Webcam capture not yet implemented - requires additional camera API integration"
        );
      } else {
        // Handle screen/window capture
        const sources = await desktopCapturer.getSources({
          types: [config.source.type],
          thumbnailSize: {
            width: config.width || 1920,
            height: config.height || 1080,
          },
        });

        const selectedSource = sources.find(
          (source) => source.id === config.source.id
        );
        if (!selectedSource) {
          throw new Error(`Source ${config.source.id} not found`);
        }

        frameData = selectedSource.thumbnail.toPNG();
        const size = selectedSource.thumbnail.getSize();
        width = size.width;
        height = size.height;
      }
    } else {
      // Fallback to primary screen
      const sources = await desktopCapturer.getSources({
        types: ["screen"],
        thumbnailSize: {
          width: config.width || 1920,
          height: config.height || 1080,
        },
      });

      if (sources.length === 0) {
        throw new Error("No screen sources available");
      }

      const source = sources[0];
      frameData = source.thumbnail.toPNG();
      const size = source.thumbnail.getSize();
      width = size.width;
      height = size.height;
    }

    // Resize if needed
    if (
      config.width &&
      config.height &&
      (width !== config.width || height !== config.height)
    ) {
      console.log(
        `Resizing from ${width}x${height} to ${config.width}x${config.height}`
      );
      frameData = await VideoProcessor.resizeImage(
        frameData,
        config.width,
        config.height
      );
      width = config.width;
      height = config.height;
    }

    // Apply codec processing if specified
    let processedData = frameData;
    let codecInfo = "raw";

    if (config.codec && config.codec !== "raw") {
      console.log(`Processing with codec: ${config.codec}`);
      const codecOptions = {
        quality: config.quality || 23,
      };

      try {
        processedData = await VideoProcessor.processWithFFmpeg(
          frameData,
          config.codec,
          codecOptions
        );
        codecInfo = `${config.codec} (FFmpeg)`;
        console.log(
          `FFmpeg processing complete: ${frameData.length} -> ${processedData.length} bytes`
        );
      } catch (codecError) {
        console.warn(
          `FFmpeg processing failed, using raw: ${codecError.message}`
        );
        // Fall back to raw if codec processing fails
        codecInfo = "raw (fallback)";
      }
    }

    // Send video frame using zero-API JSON message routing
    const success = transmitter.sendVideoFrame(
      processedData,
      Date.now() * 1000, // microseconds
      width,
      height
    );

    messageRouter.send({
      type: "video_frame_sent",
      success: success,
      timestamp: Date.now(),
      width: width,
      height: height,
      size: processedData.length,
      original_size: frameData.length,
      compression_ratio:
        frameData.length > 0 ? processedData.length / frameData.length : 1,
      source: config.source,
      codec: codecInfo,
    });

    return {
      success: success,
      width: width,
      height: height,
      size: processedData.length,
      original_size: frameData.length,
      compression_ratio:
        frameData.length > 0 ? processedData.length / frameData.length : 1,
      codec: codecInfo,
    };
  } catch (error) {
    console.error("Capture error:", error);
    messageRouter.send({
      type: "capture_error",
      error: error.message,
      timestamp: Date.now(),
    });
    return { success: false, error: error.message };
  }
});

// JVID RGB matrix processing and TOAST v2 integration
ipcMain.handle("jvid_rgb_lane", async (event, message) => {
  try {
    console.log(
      `📡 Processing JVID RGB lane: ${message.data.channel} (Lane ${message.data.lane})`
    );

    // Create TOAST v2 frame for RGB lane
    const toastFrame = createTOASTv2Frame(message.data);

    // Send via UDP multicast
    if (toastTransport) {
      await toastTransport.sendRGBLane(toastFrame);

      // Update statistics
      updateJVIDStats(message.data);

      return {
        success: true,
        lane: message.data.lane,
        channel: message.data.channel,
      };
    } else {
      console.warn("⚠️ TOAST transport not initialized for JVID RGB lane");
      return { success: false, error: "TOAST transport not available" };
    }
  } catch (error) {
    console.error("❌ Error processing JVID RGB lane:", error);
    return { success: false, error: error.message };
  }
});

/**
 * Create TOAST v2 frame for JVID RGB lane with PNBTR data
 * @param {Object} jvidMessage - JVID message with RGB lane data
 * @returns {Object} TOAST v2 frame
 */
function createTOASTv2Frame(jvidMessage) {
  const frameType = getJVIDFrameType(jvidMessage.channel);

  return {
    // TOAST v2 header (32 bytes)
    header: {
      magic: 0x544f4153, // "TOAS" in ASCII
      version: 2, // TOAST v2
      frame_type: frameType, // JVID_R, JVID_G, JVID_B
      session_id: hashString(sessionId || "jvid-60fps-001"),
      timestamp_us: jvidMessage.ts,
      sequence_num: jvidMessage.seq,
      payload_format: 0x0002, // Compact JSONL
      payload_length: JSON.stringify(jvidMessage).length,
      checksum: calculateCRC32(JSON.stringify(jvidMessage)),
      reserved: 0,
    },

    // JVID RGB payload
    payload: {
      // Core JVID data
      t: jvidMessage.t,
      id: jvidMessage.id,
      seq: jvidMessage.seq,
      ts: jvidMessage.ts,
      fps: jvidMessage.fps,
      w: jvidMessage.w,
      h: jvidMessage.h,

      // RGB lane specifics
      lane: jvidMessage.lane,
      channel: jvidMessage.channel,
      fmt: jvidMessage.fmt,
      d: jvidMessage.d, // Direct pixel data (no base64)

      // PNBTR prediction data
      pnbtr: {
        motion: jvidMessage.pnbtr.motion,
        confidence: jvidMessage.pnbtr.confidence,
        predicted: jvidMessage.pnbtr.predicted,
        method: "waveform_autocorrelation",
        horizon_frames: 16,
      },

      // TOAST synchronization
      toast: {
        audio_sample: jvidMessage.toast.audio_sample,
        frame_offset: jvidMessage.toast.frame_offset,
        clock_drift: jvidMessage.toast.clock_drift,
        sync_quality: calculateSyncQuality(jvidMessage),
      },
    },
  };
}

/**
 * Get TOAST frame type for RGB channel
 * @param {string} channel - R, G, or B
 * @returns {number} Frame type
 */
function getJVIDFrameType(channel) {
  const frameTypes = {
    R: 0x1001, // JVID_R
    G: 0x1002, // JVID_G
    B: 0x1003, // JVID_B
  };
  return frameTypes[channel] || 0x1000;
}

/**
 * Calculate synchronization quality
 * @param {Object} jvidMessage - JVID message
 * @returns {number} Sync quality (0-1)
 */
function calculateSyncQuality(jvidMessage) {
  const audioSample = jvidMessage.toast.audio_sample;
  const expectedSample = Math.floor((jvidMessage.ts * 48000) / 1000000);
  const drift = Math.abs(audioSample - expectedSample);

  // Good sync if within 1ms (48 samples at 48kHz)
  return Math.max(0, 1 - drift / 48);
}

// JVID statistics tracking
let jvidStats = {
  rgbLanesSent: { R: 0, G: 0, B: 0 },
  totalFrames: 0,
  pnbtrPredictions: 0,
  averageConfidence: 0,
  clockDriftUs: 0,
  processingTimeMs: 0,
  bandwidthMbps: 0,
  lastUpdate: Date.now(),
};

/**
 * Update JVID processing statistics
 * @param {Object} jvidMessage - JVID message
 */
function updateJVIDStats(jvidMessage) {
  // Update RGB lane counters
  jvidStats.rgbLanesSent[jvidMessage.channel]++;
  jvidStats.totalFrames++;

  // Update PNBTR statistics
  if (jvidMessage.pnbtr.confidence > 0.7) {
    jvidStats.pnbtrPredictions++;
  }

  // Calculate rolling average confidence
  const alpha = 0.1;
  jvidStats.averageConfidence =
    (1 - alpha) * jvidStats.averageConfidence +
    alpha * jvidMessage.pnbtr.confidence;

  // Update clock drift
  jvidStats.clockDriftUs = jvidMessage.toast.clock_drift;

  // Calculate bandwidth (every 60 frames = 1 second at 60fps)
  if (jvidStats.totalFrames % 180 === 0) {
    // 3 RGB lanes * 60 frames
    const now = Date.now();
    const timeDelta = (now - jvidStats.lastUpdate) / 1000; // seconds
    const bytesPerFrame = JSON.stringify(jvidMessage).length;
    const bytesPerSecond = (180 * bytesPerFrame) / timeDelta;
    jvidStats.bandwidthMbps = (bytesPerSecond * 8) / (1024 * 1024); // Convert to Mbps
    jvidStats.lastUpdate = now;

    console.log(
      `📊 JVID Stats: ${
        jvidStats.totalFrames / 3
      } frames, ${jvidStats.bandwidthMbps.toFixed(2)} Mbps, ${(
        jvidStats.averageConfidence * 100
      ).toFixed(1)}% PNBTR confidence`
    );
  }
}

// RGB lane functionality is now integrated into the existing TOASTv2Transport class in toast-transport.js

// App lifecycle
app.whenReady().then(() => {
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

// Cleanup on app quit
app.on("before-quit", () => {
  if (transmitter) {
    transmitter.stop();
  }
  if (receiver) {
    receiver.stop();
  }

  // Cleanup any temporary files
  try {
    const tempFiles = [
      "temp_input.png",
      "temp_output.mp4",
      "temp_output.webm",
      "temp_resize_input.png",
      "temp_resize_output.png",
    ];
    tempFiles.forEach((file) => {
      const filePath = path.join(__dirname, file);
      if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
      }
    });
  } catch (error) {
    console.error("Error cleaning up temporary files:", error);
  }
});

// Utility functions for TOAST v2 protocol
/**
 * Calculate simple hash of a string
 * @param {string} str - String to hash
 * @returns {number} 32-bit hash
 */
function hashString(str) {
  let hash = 0;
  if (str.length === 0) return hash;

  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & 0xffffffff; // Convert to 32-bit integer
  }

  return Math.abs(hash);
}

/**
 * Calculate CRC32 checksum
 * @param {string} data - Data to calculate checksum for
 * @returns {number} CRC32 checksum
 */
function calculateCRC32(data) {
  const crcTable = [];
  for (let i = 0; i < 256; i++) {
    let crc = i;
    for (let j = 0; j < 8; j++) {
      if (crc & 1) {
        crc = (crc >>> 1) ^ 0xedb88320;
      } else {
        crc = crc >>> 1;
      }
    }
    crcTable[i] = crc;
  }

  let crc = 0 ^ -1;
  for (let i = 0; i < data.length; i++) {
    crc = (crc >>> 8) ^ crcTable[(crc ^ data.charCodeAt(i)) & 0xff];
  }

  return (crc ^ -1) >>> 0; // Convert to unsigned 32-bit
}
