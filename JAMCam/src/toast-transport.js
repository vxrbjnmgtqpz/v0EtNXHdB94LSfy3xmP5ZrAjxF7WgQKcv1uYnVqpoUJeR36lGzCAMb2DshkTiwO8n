const dgram = require("dgram");
const { Buffer } = require("buffer");
const crc = require("crc");

/**
 * TOAST v2 Protocol Constants (from JAM_Framework_v2/include/jam_toast.h)
 */
const TOAST_MAGIC = 0x54534f54; // "TOST"
const TOAST_VERSION = 2;

/**
 * TOAST v2 Frame Types (matching JAM Framework v2)
 */
const TOASTFrameType = {
  MIDI: 0x01,
  AUDIO: 0x02,
  VIDEO: 0x03, // JVID video frames
  SYNC: 0x04,
  TRANSPORT: 0x05,
  DISCOVERY: 0x06,
  HEARTBEAT: 0x07,
  BURST_HEADER: 0x08,
};

/**
 * TOAST v2 Frame Header (32 bytes fixed)
 * Matches C++ struct from jam_toast.h
 */
class TOASTFrameHeader {
  constructor() {
    this.magic = TOAST_MAGIC;
    this.version = TOAST_VERSION;
    this.frame_type = 0;
    this.flags = 0;
    this.sequence_number = 0;
    this.timestamp_us = 0;
    this.payload_size = 0;
    this.burst_id = 0;
    this.burst_index = 0;
    this.burst_total = 1;
    this.checksum = 0;
    this.session_id = 0;
  }

  serialize() {
    const buffer = Buffer.alloc(32);
    let offset = 0;

    buffer.writeUInt32BE(this.magic, offset);
    offset += 4;
    buffer.writeUInt8(this.version, offset);
    offset += 1;
    buffer.writeUInt8(this.frame_type, offset);
    offset += 1;
    buffer.writeUInt16BE(this.flags, offset);
    offset += 2;
    buffer.writeUInt32BE(this.sequence_number, offset);
    offset += 4;
    buffer.writeUInt32BE(this.timestamp_us, offset);
    offset += 4;
    buffer.writeUInt32BE(this.payload_size, offset);
    offset += 4;
    buffer.writeUInt32BE(this.burst_id, offset);
    offset += 4;
    buffer.writeUInt8(this.burst_index, offset);
    offset += 1;
    buffer.writeUInt8(this.burst_total, offset);
    offset += 1;
    buffer.writeUInt16BE(this.checksum, offset);
    offset += 2;
    buffer.writeUInt32BE(this.session_id, offset);
    offset += 4;

    return buffer;
  }

  static deserialize(buffer) {
    const header = new TOASTFrameHeader();
    let offset = 0;

    header.magic = buffer.readUInt32BE(offset);
    offset += 4;
    header.version = buffer.readUInt8(offset);
    offset += 1;
    header.frame_type = buffer.readUInt8(offset);
    offset += 1;
    header.flags = buffer.readUInt16BE(offset);
    offset += 2;
    header.sequence_number = buffer.readUInt32BE(offset);
    offset += 4;
    header.timestamp_us = buffer.readUInt32BE(offset);
    offset += 4;
    header.payload_size = buffer.readUInt32BE(offset);
    offset += 4;
    header.burst_id = buffer.readUInt32BE(offset);
    offset += 4;
    header.burst_index = buffer.readUInt8(offset);
    offset += 1;
    header.burst_total = buffer.readUInt8(offset);
    offset += 1;
    header.checksum = buffer.readUInt16BE(offset);
    offset += 2;
    header.session_id = buffer.readUInt32BE(offset);
    offset += 4;

    return header;
  }

  validate() {
    return this.magic === TOAST_MAGIC && this.version === TOAST_VERSION;
  }
}

/**
 * TOAST v2 Frame (complete frame with header and payload)
 */
class TOASTFrame {
  constructor(frameType, payload = Buffer.alloc(0)) {
    this.header = new TOASTFrameHeader();
    this.header.frame_type = frameType;
    this.header.payload_size = payload.length;
    this.payload = payload;
  }

  // Calculate CRC16 checksum (matching C++ implementation)
  calculateChecksum() {
    this.header.checksum = 0; // Reset checksum

    const headerBuffer = this.header.serialize();
    const fullBuffer = Buffer.concat([headerBuffer, this.payload]);

    // Calculate CRC16 (using same algorithm as C++)
    this.header.checksum = crc.crc16(fullBuffer) & 0xffff;
  }

  // Validate checksum
  validateChecksum() {
    const storedChecksum = this.header.checksum;
    this.calculateChecksum();
    const calculatedChecksum = this.header.checksum;
    this.header.checksum = storedChecksum; // Restore original

    return storedChecksum === calculatedChecksum;
  }

  // Serialize complete frame
  serialize() {
    this.header.payload_size = this.payload.length;
    this.calculateChecksum();

    const headerBuffer = this.header.serialize();
    return Buffer.concat([headerBuffer, this.payload]);
  }

  // Deserialize complete frame
  static deserialize(buffer) {
    if (buffer.length < 32) return null;

    const header = TOASTFrameHeader.deserialize(buffer.slice(0, 32));
    if (!header.validate()) return null;

    const payload = buffer.slice(32, 32 + header.payload_size);
    const frame = new TOASTFrame(header.frame_type, payload);
    frame.header = header;

    // Validate checksum
    if (!frame.validateChecksum()) {
      console.warn("TOAST frame checksum validation failed");
      return null;
    }

    return frame;
  }

  // Create JVID video frame
  static createVideoFrame(sessionId, frameData, timestamp, width, height) {
    const jsonPayload = {
      type: "jvid_frame",
      timestamp_us: timestamp,
      width: width,
      height: height,
      format: "rgb24",
      data: frameData.toString("base64"),
    };

    const payload = Buffer.from(JSON.stringify(jsonPayload));
    const frame = new TOASTFrame(TOASTFrameType.VIDEO, payload);
    frame.header.session_id = sessionId;
    frame.header.timestamp_us = timestamp;

    return frame;
  }

  // Create discovery frame
  static createDiscoveryFrame(sessionId, deviceName = "JAMCam") {
    const jsonPayload = {
      type: "discovery_announce",
      timestamp_us: Date.now() * 1000,
      device_name: deviceName,
      capabilities: ["jvid_transmit", "jvid_receive"],
      session_id: sessionId,
    };

    const payload = Buffer.from(JSON.stringify(jsonPayload));
    const frame = new TOASTFrame(TOASTFrameType.DISCOVERY, payload);
    frame.header.session_id = sessionId;
    frame.header.timestamp_us = Date.now() * 1000;

    return frame;
  }

  // Create heartbeat frame
  static createHeartbeatFrame(sessionId) {
    const jsonPayload = {
      type: "heartbeat",
      timestamp_us: Date.now() * 1000,
      session_id: sessionId,
    };

    const payload = Buffer.from(JSON.stringify(jsonPayload));
    const frame = new TOASTFrame(TOASTFrameType.HEARTBEAT, payload);
    frame.header.session_id = sessionId;
    frame.header.timestamp_us = Date.now() * 1000;

    return frame;
  }
}

/**
 * TOAST v2 Transport - Zero-API JSON Message Routing
 * Implements the revolutionary JAMNet paradigm where JSON messages replace APIs
 * Now enhanced with JVID RGB lane processing and PNBTR integration
 */
class TOASTv2Transport {
  constructor() {
    this.socket = null;
    this.isRunning = false;
    this.sessionId = Math.floor(Math.random() * 0xffffffff);
    this.sequenceNumber = 0;
    this.multicastAddress = null;
    this.port = null;
    this.discoveryInterval = null;
    this.heartbeatInterval = null;

    // JVID RGB lane buffers for PNBTR processing
    this.rgbLaneBuffers = { R: [], G: [], B: [] };
    this.frameReassembly = new Map();

    // Zero-API JSON message handlers
    this.messageHandlers = new Map();
    this.stats = {
      frames_sent: 0,
      frames_received: 0,
      bytes_sent: 0,
      bytes_received: 0,
      active_peers: 0,
      average_latency_us: 0,
      // JVID-specific stats
      rgb_lanes_sent: { R: 0, G: 0, B: 0 },
      pnbtr_predictions: 0,
      frame_reconstructions: 0,
      prediction_accuracy: 0,
    };
  }

  // Universal JSON message routing (eliminates traditional APIs)
  routeMessage(jsonMessage) {
    try {
      const message = JSON.parse(jsonMessage);
      const handler = this.messageHandlers.get(message.type);
      if (handler) {
        handler(message);
      }
    } catch (error) {
      console.error("JSON message routing error:", error);
    }
  }

  // Register JSON message handler (replaces traditional callback APIs)
  on(messageType, handler) {
    this.messageHandlers.set(messageType, handler);
  }

  async start(multicastAddress, port) {
    this.multicastAddress = multicastAddress;
    this.port = port;

    return new Promise((resolve, reject) => {
      this.socket = dgram.createSocket("udp4");

      this.socket.on("message", (msg, rinfo) => {
        this.handleIncomingFrame(msg, rinfo);
      });

      this.socket.on("error", (error) => {
        console.error("TOAST socket error:", error);
        reject(error);
      });

      this.socket.bind(() => {
        this.socket.addMembership(this.multicastAddress);
        this.isRunning = true;

        // Start zero-API discovery protocol
        this.startDiscovery();
        this.startHeartbeat();

        resolve();
      });
    });
  }

  stop() {
    this.isRunning = false;

    if (this.discoveryInterval) {
      clearInterval(this.discoveryInterval);
      this.discoveryInterval = null;
    }

    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }

    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  }

  // Zero-API video transmission
  sendVideoFrame(frameData, timestamp, width, height) {
    if (!this.isRunning) return false;

    const frame = TOASTFrame.createVideoFrame(
      this.sessionId,
      frameData,
      timestamp,
      width,
      height
    );
    frame.header.sequence_number = this.sequenceNumber++;

    return this.sendFrame(frame);
  }

  // Zero-API discovery (JSON message routing)
  sendDiscovery() {
    if (!this.isRunning) return false;

    const frame = TOASTFrame.createDiscoveryFrame(this.sessionId);
    frame.header.sequence_number = this.sequenceNumber++;

    return this.sendFrame(frame);
  }

  // Internal frame transmission
  sendFrame(frame) {
    if (!this.socket || !this.isRunning) return false;

    const frameBuffer = frame.serialize();

    this.socket.send(frameBuffer, this.port, this.multicastAddress, (error) => {
      if (error) {
        console.error("TOAST send error:", error);
      } else {
        this.stats.frames_sent++;
        this.stats.bytes_sent += frameBuffer.length;
      }
    });

    return true;
  }

  // Zero-API frame processing (JSON message routing)
  handleIncomingFrame(buffer, rinfo) {
    const frame = TOASTFrame.deserialize(buffer);
    if (!frame) return;

    this.stats.frames_received++;
    this.stats.bytes_received += buffer.length;

    // Route frame to appropriate JSON message handler
    try {
      const jsonMessage = frame.payload.toString("utf8");
      this.routeMessage(jsonMessage);
    } catch (error) {
      console.error("Frame processing error:", error);
    }
  }

  // Zero-API discovery protocol
  startDiscovery() {
    if (this.discoveryInterval) return;

    const sendDiscovery = () => {
      if (this.isRunning) {
        this.sendDiscovery();
      }
    };

    sendDiscovery(); // Send immediately
    this.discoveryInterval = setInterval(sendDiscovery, 10000); // Every 10 seconds
  }

  // Zero-API heartbeat protocol
  startHeartbeat() {
    if (this.heartbeatInterval) return;

    const sendHeartbeat = () => {
      if (this.isRunning) {
        const frame = TOASTFrame.createHeartbeatFrame(this.sessionId);
        frame.header.sequence_number = this.sequenceNumber++;
        this.sendFrame(frame);
      }
    };

    sendHeartbeat(); // Send immediately
    this.heartbeatInterval = setInterval(sendHeartbeat, 5000); // Every 5 seconds
  }

  getStats() {
    return this.stats;
  }

  getSessionId() {
    return this.sessionId;
  }

  /**
   * Send RGB lane via TOAST v2 protocol for JVID processing
   * @param {Object} toastFrame - TOAST v2 frame with RGB data
   */
  async sendRGBLane(toastFrame) {
    try {
      // Serialize frame to compact binary format
      const frameBuffer = this.serializeTOASTFrame(toastFrame);

      // Send via UDP multicast
      await this.sendUDPMulticast(frameBuffer);

      // Buffer for redundancy (PNBTR-style)
      this.bufferRGBLane(toastFrame);

      // Update statistics
      this.stats.rgb_lanes_sent[toastFrame.payload.channel]++;
    } catch (error) {
      console.error("❌ Error sending RGB lane:", error);
      throw error;
    }
  }

  /**
   * Send data via UDP multicast
   * @param {Buffer} data - Data to send
   */
  async sendUDPMulticast(data) {
    return new Promise((resolve, reject) => {
      if (!this.socket || !this.isRunning) {
        reject(new Error("Socket not available"));
        return;
      }

      this.socket.send(data, this.port, this.multicastAddress, (error) => {
        if (error) {
          reject(error);
        } else {
          this.stats.frames_sent++;
          this.stats.bytes_sent += data.length;
          resolve();
        }
      });
    });
  }

  /**
   * Serialize TOAST v2 frame to binary with JVID RGB data
   * @param {Object} frame - TOAST frame
   * @returns {Buffer} Serialized frame
   */
  serializeTOASTFrame(frame) {
    const header = Buffer.alloc(32);
    let offset = 0;

    // Write header fields
    header.writeUInt32BE(frame.header.magic, offset);
    offset += 4;
    header.writeUInt16BE(frame.header.version, offset);
    offset += 2;
    header.writeUInt16BE(frame.header.frame_type, offset);
    offset += 2;
    header.writeUInt32BE(frame.header.session_id, offset);
    offset += 4;
    header.writeBigUInt64BE(BigInt(frame.header.timestamp_us), offset);
    offset += 8;
    header.writeUInt32BE(frame.header.sequence_num, offset);
    offset += 4;
    header.writeUInt16BE(frame.header.payload_format, offset);
    offset += 2;
    header.writeUInt16BE(frame.header.payload_length, offset);
    offset += 2;
    header.writeUInt32BE(frame.header.checksum, offset);
    offset += 4;

    // Serialize payload as compact JSONL
    const payloadJSON = JSON.stringify(frame.payload);
    const payloadBuffer = Buffer.from(payloadJSON, "utf8");

    // Combine header and payload
    return Buffer.concat([header, payloadBuffer]);
  }

  /**
   * Buffer RGB lane for redundancy and PNBTR prediction
   * @param {Object} frame - TOAST frame
   */
  bufferRGBLane(frame) {
    const channel = frame.payload.channel;
    const buffer = this.rgbLaneBuffers[channel];

    // Add to buffer
    buffer.push(frame);

    // Keep only recent frames (8 frames for PNBTR prediction)
    if (buffer.length > 8) {
      buffer.shift();
    }
  }

  /**
   * Process received RGB lane and attempt frame reconstruction
   * @param {Buffer} frameBuffer - Received frame buffer
   */
  processReceivedRGBLane(frameBuffer) {
    try {
      const frame = this.deserializeTOASTFrame(frameBuffer);
      const seq = frame.payload.seq;
      const channel = frame.payload.channel;

      // Store in reassembly buffer
      if (!this.frameReassembly.has(seq)) {
        this.frameReassembly.set(seq, {
          R: null,
          G: null,
          B: null,
          timestamp: Date.now(),
        });
      }

      this.frameReassembly.get(seq)[channel] = frame;

      // Check if we have all RGB channels for this frame
      const frameData = this.frameReassembly.get(seq);
      if (frameData.R && frameData.G && frameData.B) {
        this.reconstructCompleteFrame(seq, frameData);
        this.frameReassembly.delete(seq);
        this.stats.frame_reconstructions++;
      }

      // Clean up old incomplete frames
      this.cleanupFrameReassembly();
    } catch (error) {
      console.error("❌ Error processing received RGB lane:", error);
    }
  }

  /**
   * Deserialize TOAST v2 frame from binary
   * @param {Buffer} buffer - Serialized frame
   * @returns {Object} Deserialized frame
   */
  deserializeTOASTFrame(buffer) {
    if (buffer.length < 32) {
      throw new Error("Invalid frame: too short");
    }

    let offset = 0;
    const header = {};

    // Read header fields
    header.magic = buffer.readUInt32BE(offset);
    offset += 4;
    header.version = buffer.readUInt16BE(offset);
    offset += 2;
    header.frame_type = buffer.readUInt16BE(offset);
    offset += 2;
    header.session_id = buffer.readUInt32BE(offset);
    offset += 4;
    header.timestamp_us = buffer.readBigUInt64BE(offset);
    offset += 8;
    header.sequence_num = buffer.readUInt32BE(offset);
    offset += 4;
    header.payload_format = buffer.readUInt16BE(offset);
    offset += 2;
    header.payload_length = buffer.readUInt16BE(offset);
    offset += 2;
    header.checksum = buffer.readUInt32BE(offset);
    offset += 4;

    // Read payload
    const payloadBuffer = buffer.slice(offset, offset + header.payload_length);
    const payload = JSON.parse(payloadBuffer.toString("utf8"));

    return { header, payload };
  }

  /**
   * Reconstruct complete frame from RGB lanes
   * @param {number} seq - Sequence number
   * @param {Object} frameData - RGB lane data
   */
  reconstructCompleteFrame(seq, frameData) {
    console.log(`🎥 Reconstructing frame ${seq} from RGB lanes`);

    // Merge RGB channels back into image data
    const width = frameData.R.payload.w;
    const height = frameData.G.payload.h;
    const pixelCount = width * height;

    const imageData = new Uint8ClampedArray(pixelCount * 4);

    for (let i = 0; i < pixelCount; i++) {
      imageData[i * 4] = frameData.R.payload.d[i]; // Red
      imageData[i * 4 + 1] = frameData.G.payload.d[i]; // Green
      imageData[i * 4 + 2] = frameData.B.payload.d[i]; // Blue
      imageData[i * 4 + 3] = 255; // Alpha
    }

    // Emit reconstructed frame via zero-API JSON message routing
    this.routeMessage(
      JSON.stringify({
        type: "jvid_frame_reconstructed",
        seq,
        width,
        height,
        imageData: Array.from(imageData),
        pnbtr: {
          confidence:
            (frameData.R.payload.pnbtr.confidence +
              frameData.G.payload.pnbtr.confidence +
              frameData.B.payload.pnbtr.confidence) /
            3,
          motion: frameData.R.payload.pnbtr.motion,
        },
      })
    );
  }

  /**
   * Clean up old incomplete frames from reassembly buffer
   */
  cleanupFrameReassembly() {
    const now = Date.now();
    const timeout = 100; // 100ms timeout for frame completion

    for (const [seq, frameData] of this.frameReassembly.entries()) {
      if (now - frameData.timestamp > timeout) {
        console.warn(
          `⚠️ Frame ${seq} incomplete after timeout, applying PNBTR reconstruction`
        );
        this.applyPNBTRReconstruction(seq, frameData);
        this.frameReassembly.delete(seq);
      }
    }
  }

  /**
   * Apply PNBTR reconstruction for missing RGB lanes
   * @param {number} seq - Sequence number
   * @param {Object} frameData - Partial frame data
   */
  applyPNBTRReconstruction(seq, frameData) {
    // Use PNBTR prediction to fill missing channels
    const channels = ["R", "G", "B"];
    const availableChannels = channels.filter((ch) => frameData[ch]);

    if (availableChannels.length === 0) return; // No data to work with

    // Use available channel as reference for prediction
    const referenceChannel = frameData[availableChannels[0]];

    channels.forEach((channel) => {
      if (!frameData[channel]) {
        console.log(
          `🔮 PNBTR predicting missing ${channel} channel for frame ${seq}`
        );
        frameData[channel] = this.predictMissingChannel(
          referenceChannel,
          channel
        );
        this.stats.pnbtr_predictions++;
      }
    });

    // Reconstruct with predicted data
    this.reconstructCompleteFrame(seq, frameData);
  }

  /**
   * Predict missing RGB channel using PNBTR methodology
   * @param {Object} referenceFrame - Available channel frame
   * @param {string} targetChannel - Channel to predict
   * @returns {Object} Predicted channel frame
   */
  predictMissingChannel(referenceFrame, targetChannel) {
    const predicted = JSON.parse(JSON.stringify(referenceFrame.payload));
    predicted.channel = targetChannel;

    // Apply channel-specific PNBTR prediction
    const channelOffset = { R: 0, G: 85, B: 170 }[targetChannel];

    predicted.d = predicted.d.map((pixel) => {
      // Simplified PNBTR-style prediction based on reference channel
      const predictedValue = Math.max(
        0,
        Math.min(255, pixel + channelOffset - 85)
      );
      return predictedValue;
    });

    // Update PNBTR metadata
    predicted.pnbtr.confidence *= 0.7; // Reduce confidence for predicted data
    predicted.pnbtr.predicted = true;
    predicted.pnbtr.method = "channel_extrapolation";

    return { payload: predicted };
  }
}

module.exports = {
  TOASTv2Transport,
  TOASTFrame,
  TOASTFrameType,
  TOASTFrameHeader,
};
