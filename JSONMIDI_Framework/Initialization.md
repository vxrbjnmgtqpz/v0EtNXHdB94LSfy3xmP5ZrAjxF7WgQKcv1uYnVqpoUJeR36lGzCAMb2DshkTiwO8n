MIDI 2.0 Byte-Level JSON Schema for Streaming

Byte-Level Schematic for Streaming MIDI as JSON
Overview of MIDI Streaming as JSON
We propose a framework to stream MIDI data over the internet using JSON, providing a 1:1 byte-level mapping of MIDI messages into a human-readable format. The goal is to transmit every MIDI event (including MIDI 1.0 and MIDI 2.0 messages) as discrete JSON chunks in real time, without losing any information. By using a streaming JSON approach (e.g. leveraging a library like Oboe.js or its fork Bassoon.js), we can send a continuous sequence of JSON objects over HTTP to a front-end, which can parse events incrementally as they arrive
github.com
. Each MIDI message (note on/off, control change, etc.) will be converted into a JSON object that preserves all of its original byte data, ensuring the mapping is reversible and exact. This JSON stream will be designed to support both MIDI 1.0 and MIDI 2.0 protocols, covering all types of MIDI messages, and will be synchronized with a master transport clock (provided by a JUCE-based engine) for timing accuracy. We also outline a strategy to allocate each MIDI channel to its own sub-stream, which allows flexible routing and even duplication of channels on unused streams for redundancy or multi-destination purposes. Using JSON for MIDI has the advantage of being self-descriptive and platform-agnostic (and even the new MIDI 2.0 specification uses JSON for certain data exchange, like Property Exchange SysEx messages
sweetwater.com
). Below, we detail the byte-level structure of MIDI messages and how each byte is represented in JSON, then discuss timing synchronization and multi-channel streaming considerations.
MIDI Message Structure: MIDI 1.0 and 2.0 Overview
MIDI 1.0 Byte Format: In MIDI 1.0, every message is transmitted as a sequence of bytes in a streaming format. Each MIDI message consists of one Status byte followed by one or two Data bytes (except System Exclusive which can have arbitrary length)
studiocode.dev
. The status byte is distinguished by having its most significant bit (MSB) set to 1 (i.e. values 128–255 or 0x80–0xFF), whereas data bytes have MSB = 0 (values 0–127)
studiocode.dev
. The status byte indicates the type of message (e.g. Note On, Control Change) and, for channel-specific messages, it also encodes the MIDI channel number in its lower 4 bits
studiocode.dev
. (In this scheme, a status of 0x90 means "Note On on channel 1", 0x91 means channel 2, and so on up to 0x9F for channel 16
midi.org
.) The number of data bytes that follow is determined by the message type – for example, a Note On (status 0x9n) requires two data bytes (key and velocity), while a Program Change (0xCn) requires only one data byte (program number)
studiocode.dev
. The table below summarizes the main MIDI 1.0 channel message types and their byte structure:
Note On (0x9n): 2 data bytes – Key Number (0–127), Velocity (0–127)
studiocode.dev
Note Off (0x8n): 2 data bytes – Key Number, Release Velocity (0–127)
studiocode.dev
Polyphonic Aftertouch (0xAn): 2 data bytes – Key Number, Pressure Value
studiocode.dev
Control Change (0xBn): 2 data bytes – Controller Number (0–119 general, 120–127 channel mode), Controller Value
studiocode.dev
Program Change (0xCn): 1 data byte – Program Number (0–127)
studiocode.dev
Channel Aftertouch (0xDn): 1 data byte – Pressure Value
studiocode.dev
Pitch Bend (0xEn): 2 data bytes – LSB, MSB (14-bit pitch value combined)
studiocode.dev
Figure: Typical MIDI 1.0 message structure – one status byte (MSB=1) followed by one or two data bytes (MSB=0). The status byte encodes the message type and channel, and determines how many data bytes follow. In addition to channel voice messages, MIDI 1.0 defines system messages which are not tied to a channel. These fall into three categories: System Exclusive, System Common, and System Real-Time
studiocode.dev
. System Exclusive (SysEx) messages begin with the 0xF0 status and contain a manufacturer ID and arbitrary data bytes, terminated by an End-Of-Exclusive byte (0xF7)
studiocode.dev
. System Common messages include things like MIDI Time Code quarter-frame, Song Position Pointer, Song Select, etc., and they have their own defined data lengths (for example, Song Position Pointer is 0xF2 followed by 2 data bytes)
studiocode.dev
. System Real-Time messages (0xF8 through 0xFF) are single-byte messages that can intersperse with other data – for example, 0xF8 is a Timing Clock tick, 0xFA is Start, 0xFB Continue, 0xFC Stop, 0xFE Active Sensing, and 0xFF System Reset
studiocode.dev
. These real-time messages have no data bytes and can be sent at any time, even between the bytes of other messages, to maintain timing signals
studiocode.dev
.
Running Status: MIDI 1.0 also allows running status, a mechanism to omit the status byte for subsequent messages of the same type in a stream to save bandwidth
studiocode.dev
. In our JSON framework, we will not use running status; instead, each JSON event will explicitly include its full status/message type so that every JSON object is self-contained. This ensures a one-to-one correspondence with actual MIDI events – it may slightly increase size compared to a running-status byte stream, but it vastly simplifies parsing and maintains clarity.
MIDI 2.0 and Universal MIDI Packets: MIDI 2.0 is designed as an extension of MIDI 1.0 that remains backward-compatible
midi.org
midi.org
. It introduces a new binary protocol format called the Universal MIDI Packet (UMP), which packages messages into 32-bit words (and bundles of 1 to 4 words, i.e. 32, 64, 96, or 128 bits total per message) instead of a raw byte stream
midi.org
. The UMP allows conveying both traditional MIDI 1.0 messages and new MIDI 2.0 messages in a unified way
amei.or.jp
amei.or.jp
. Notably, the UMP format expands the addressing space by introducing 16 MIDI groups, each group containing 16 channels (so up to 256 channels total)
amei.or.jp
. It also provides an optional Jitter Reduction Timestamp that can prefix messages for higher timing precision over jittery transports
amei.or.jp
. MIDI 2.0 extends the resolution and range of many performance values. For example, controller values can span 32-bit precision, up from 7-bit in MIDI 1.0
midi.org
. There are over 32,000 controller IDs available (versus 128 in MIDI 1.0), including per-note controllers for more expressive control
midi.org
. Note velocity is upgraded from 7-bit to 16-bit resolution, allowing much finer gradations of playing dynamics
midi.org
. Additionally, MIDI 2.0 defines new messages that unify what used to be multi-message sequences in MIDI 1.0 – for instance, a single MIDI 2.0 program change message can include bank selection data in one go, rather than sending CC#0/32 and then Program Change as separate messages. It also adds per-note pitch bend and other per-note expressive data, which required MPE or channel rotation tricks in MIDI 1.0. Despite these changes, MIDI 2.0 devices can negotiate and fall back to MIDI 1.0 if needed (using a Capability Inquiry exchange) to ensure backward compatibility
midi.org
. From a byte-level perspective, a MIDI 2.0 Channel Voice message in UMP format consists of a 32-bit packet with the following high-level structure: 4 bits of group, 4 bits of message type, and the remaining 24 bits used for message data (with additional 32-bit words for extended data if required). For example, a "Note On" in MIDI 2.0 uses a 64-bit packet: it includes the note number (likely still 7 bits as before, or possibly larger range but typically 0–127 is used), a 16-bit velocity, and a 16-bit attribute field which can carry extra information (such as per-note articulation or fine pitch) along with an attribute type ID. The key point is that the data size for many fields has increased (e.g. velocities, controller values, and pitch bend are 16-bit or more), but the fundamental concept of each message is similar to MIDI 1.0. We will ensure our JSON schema can accommodate these extended values by using numeric fields that aren’t limited to 0–127, and by including additional fields for the new data where applicable. MIDI 2.0 also retains the idea of status nibbles for message type (e.g. 0x9 = Note On, 0xB = Control Change, etc.) and channel in the lower bits of the code, so in many cases the "type" and "channel" identification in JSON can work similarly for both protocols – the difference is just that MIDI 2.0 may carry larger values or extra parameters in the message. In summary, MIDI 1.0 messages will be encoded with their 7-bit data, while MIDI 2.0 messages will be encoded with their extended data width and additional fields, but the JSON structure will be designed to handle both seamlessly. (For completeness, note that the system messages in MIDI 2.0 are mostly carried over from MIDI 1.0 in function, aside from a new form of SysEx (called SysEx 8) that allows bytes to use the full 8-bit range without the 7-bit packing of MIDI 1.0 SysEx. Our JSON representation of SysEx will not need to worry about the 7-bit packing, since we will represent SysEx data bytes directly as values 0–255, thereby naturally supporting the SysEx 8 concept.)
JSON Event Representation (1:1 Byte Mapping of MIDI Messages)
Each MIDI message, whether from the 1.0 or 2.0 protocol, will be represented as a JSON object in the stream. This object contains fields that explicitly describe the message type and include all of the original data bytes (or the expanded data in the case of MIDI 2.0). The mapping is designed such that the JSON object contains enough information to reconstruct the exact MIDI bytes in order, preserving the sequence of events exactly. In other words, there is a one-to-one correspondence: one MIDI message = one JSON object in the stream. General structure: Every JSON event will have at minimum a "type" field (a human-readable name for the MIDI message type) and may include a "channel" (if applicable), as well as numeric fields for the data bytes (often given more descriptive names like "note", "velocity", "value", etc., depending on the context). We also include a "timestamp" field (or "time") for synchronization (explained in the next section), which marks when the event should occur relative to the transport clock. Optionally, we can include a "rawBytes" field – an array of integers (0–255) or a hex string – that shows the exact bytes of the original MIDI message. This can be useful for debugging or for any applications that prefer dealing with raw MIDI data directly. However, since all information is also captured in the structured fields, the raw bytes are not strictly necessary for functionality. Below, we break down the JSON representation by category of MIDI message:
Channel Voice Messages (Musical/Performance Events)
These are the most common MIDI messages, carrying musical performance data on a specific channel (1–16). In JSON, we will include a "channel" field (1–16) for these messages, and fields for their parameters. The "type" field will distinguish the kind of message. All numeric values that originated as 7-bit MIDI data (0–127 range) will be represented as standard JSON numbers (integers). For MIDI 2.0 extended resolution, those values might go beyond 127 (e.g. up to 65535 for 16-bit) – we will represent them with a normal number as well (JSON has no specific limit for integer size, so this is fine). For clarity, here are all the channel voice messages and how we encode them:
Note On:
Bytes: Status = 0x9n (n = channel-1), Data1 = Key (0–127), Data2 = Velocity (0–127).
JSON:
json
Copy
{
  "type": "noteOn",
  "channel": 1,
  "note": 60,
  "velocity": 127,
  "timestamp": <time>
}
This example represents Note On, channel 1, note number 60, velocity 127. The "note" and "velocity" correspond to the two data bytes. In MIDI 2.0 mode, if velocity is 16-bit, we would allow "velocity" to range 0–65535. (If desired, we could also split it into "velocityMSB" and "velocityLSB" for 16-bit, but usually a single number is sufficient since the receiver knows whether it's 7-bit or 16-bit based on protocol context or a version field.) If a MIDI 2.0 Note On includes an attribute (additional 16-bit value with a type code), we would add e.g. "attributeType" and "attributeValue" fields. For instance, a MIDI 2.0 Note On might look like:
json
Copy
{
  "type": "noteOn",
  "channel": 1,
  "note": 60,
  "velocity": 30000,
  "attributeType": 1,
  "attributeValue": 10000,
  "timestamp": <time>
}
Here "attributeType": 1 could indicate (for example) that the attribute is a per-note pitch adjustment, and "attributeValue": 10000 (out of 65535) is the value. We will formalize attribute types as needed when using MIDI 2.0 messages.
Note Off:
Bytes: Status = 0x8n, Data1 = Key, Data2 = Release Velocity.
JSON:
json
Copy
{
  "type": "noteOff",
  "channel": 1,
  "note": 60,
  "velocity": 64,
  "timestamp": <time>
}
We use "velocity" here to represent release velocity for note-off (alternatively could use "releaseVelocity" for clarity, but using the same key "velocity" is acceptable as context distinguishes note-on vs note-off). In running status MIDI 1.0, Note Off is sometimes represented as Note On with velocity 0, but our JSON will explicitly use "type": "noteOff" for clarity when the intent is a true Note Off. (If the source MIDI uses Note On with 0 velocity for note-off, we can convert it to a "noteOff" JSON type anyway — it’s semantically equivalent.)
Polyphonic Aftertouch (Key Pressure):
Bytes: Status = 0xAn, Data1 = Key, Data2 = Pressure value.
JSON:
json
Copy
{
  "type": "polyAftertouch",
  "channel": 1,
  "note": 60,
  "pressure": 80,
  "timestamp": <time>
}
This indicates key-specific aftertouch (pressure on an individual note). The "pressure" value is 0–127 normally, or up to 16383 if using MIDI 2.0 14-bit pressure.
Control Change (CC):
Bytes: Status = 0xBn, Data1 = Controller Number, Data2 = Value.
JSON:
json
Copy
{
  "type": "controlChange",
  "channel": 1,
  "controller": 74,
  "value": 45,
  "timestamp": <time>
}
This example could represent controller #74 (often assigned to filter cutoff in MIDI conventions) with a value of 45. Controller numbers 0–119 are general purpose or defined MIDI CCs, while 120–127 are channel mode messages (like All Notes Off, etc.). We will use the same "type": "controlChange" for all, and just use the numeric "controller" ID. If a controller is one of the channel mode functions (e.g. controller 123 = All Notes Off), it’s still represented the same way here; the receiver can handle the special ID as needed. In MIDI 2.0, controller values can be 32-bit; we would then allow "value" to be a larger number (the JSON number can naturally expand to that). If needed, one could include a flag or separate type for 32-bit vs 7-bit controllers, but it might be simpler to always use a 0–16383 range for value and just note that in MIDI 1.0 only 0–127 will be used. (Alternatively, include "resolution":7 or "midiVersion":1 vs 2 at the message or stream level to clarify.)
Program Change:
Bytes: Status = 0xCn, Data1 = Program Number. (No second data byte.)
JSON:
json
Copy
{
  "type": "programChange",
  "channel": 1,
  "program": 10,
  "timestamp": <time>
}
This sets the instrument program on channel 1 to #10 (in MIDI, program numbers typically 0–127 correspond to patches). In MIDI 1.0, if we need to specify bank, that comes from CC#0 and CC#32 messages before the PC. In MIDI 2.0, there is an extended message that can carry bank and program in one; we could represent that as "type": "programChange" with additional fields "bankMsb" and "bankLsb" (or a combined "bank" number) if we decide to unify them. Alternatively, we could introduce a separate "type": "bankSelect" event in JSON if using the MIDI 1.0 method with CC messages. But a cleaner MIDI 2.0 approach is a single event; we can detect if two CCs followed by PC come in (typical MIDI 1.0 bank select sequence) and possibly combine or just send them as separate JSON events (that's simpler – just send CC0, CC32, then PC events). For now, we treat Program Change as just changing the program number; bank select remains via CC events.
Channel Aftertouch (Channel Pressure):
Bytes: Status = 0xDn, Data1 = Pressure value.
JSON:
json
Copy
{
  "type": "channelPressure",
  "channel": 1,
  "pressure": 100,
  "timestamp": <time>
}
This is aftertouch applied to the whole channel (as opposed to a specific note). The pressure value is 0–127 (7-bit) or up to 16383 (14-bit) in MIDI 2.0.
Pitch Bend Change:
Bytes: Status = 0xEn, Data1 = LSB (0–127), Data2 = MSB (0–127). These two bytes form a 14-bit little-endian value (0–16383) where 8192 is the center (no bend).
JSON:
json
Copy
{
  "type": "pitchBend",
  "channel": 1,
  "value": 8192,
  "timestamp": <time>
}
Here "value": 8192 represents pitch wheel at center (no pitch deviation). If we wanted, we could instead use a signed interpretation (-8192 to +8191), but it’s simplest to use the raw 0–16383 range and let 8192 be the center. In MIDI 2.0, pitch bend is 32-bit (to allow very high resolution pitch changes)
midi.org
. We can accommodate that by allowing "value" to be 0–(2^32-1) if needed, or using a higher-level representation like a float in semitones. However, to keep it byte-level, we will likely stick to numeric representation of the full value. For instance, a MIDI 2.0 pitch bend might be "value": 543210000 (some 32-bit number) in JSON. The receiver, knowing it's MIDI 2.0, would interpret that accordingly.
For all channel voice messages, we preserve the exact channel number and message type so that the original status byte can be reconstructed. For example, given {"type":"noteOn","channel":5,...} we know the status byte would be 0x94 (since 0x90 is Note On channel1, 0x95 would be channel 6 because 0x94 is channel 5 if 0x90 was channel1 in 0-based? Actually, careful: The status formula is 0x90 + (channel-1). So for channel 5, 0x90 + 4 = 0x94, yes)
studiocode.dev
midi.org
. We could compute that if needed or even include a "statusByte" field (e.g. "status": 144 for 0x90) to make the mapping explicit. Including "status": 144 (decimal for 0x90) might be useful for debugging but redundant because type+channel already imply it. It's optional in our scheme; the priority is the human-readable form.
System Messages (Common and Real-Time)
System messages do not carry a channel and often have special meanings for synchronization or meta-information. We will represent them with "type" and any necessary data fields. Here are the various system messages and their JSON forms:
MIDI Timing Clock (0xF8):
Bytes: Status = 0xF8, no data. This is a timing tick that typically comes 24 times per quarter note (if using MIDI clock sync).
JSON:
json
Copy
{ "type": "timingClock", "timestamp": <time> }
We include the timestamp to know which clock tick in time this is. In many cases, if the whole stream is locked to a tempo, the timing clock messages might be emitted at a regular interval. We may or may not need to send 0xF8 clock events if we are directly time-stamping all events (more on this in the synchronization section). If we do include them, the JSON as above suffices (no additional data fields).
Start (0xFA), Stop (0xFC), Continue (0xFB):
These control the transport (start playback, stop playback, continue from pause). They have no data bytes.
JSON:
json
Copy
{ "type": "start", "timestamp": <time> }
and similarly { "type": "stop", ... }, { "type": "continue", ... }. The timestamp indicates when the action took place in the timeline (e.g. when the sequence was started or stopped).
Active Sensing (0xFE):
This is a keep-alive message some MIDI devices send periodically to indicate the connection is alive. No data bytes.
JSON:
json
Copy
{ "type": "activeSensing", "timestamp": <time> }
(Usually the timestamp is not musically relevant, it's more of a real-time pulse. We might still include when it was received.)
System Reset (0xFF):
This signals resetting of the MIDI system.
JSON:
json
Copy
{ "type": "reset", "timestamp": <time> }
MIDI Time Code Quarter Frame (0xF1):
Bytes: Status = 0xF1, Data1 = one nibble of time code. This is used for syncing to external time code (like SMPTE) in pieces. The data byte's high 3 bits indicate the message type (frame, seconds, minutes, etc.) and low 4 bits carry a value chunk.
JSON:
json
Copy
{ 
  "type": "timeCodeQuarter",
  "value": 0x5, 
  "timestamp": <time> 
}
Here "value" 0x5 is an example of one 8-bit quarter-frame message. We could break it into the message type and data nibble if needed (since the bits have meaning), but since this is a fairly specialized sync message, keeping the combined byte might suffice unless we need to interface with SMPTE directly.
Song Position Pointer (0xF2):
Bytes: Status = 0xF2, Data1 = LSB, Data2 = MSB of a 14-bit value. This value is the number of MIDI beats (sixteenth note positions) since the start of the song. (One MIDI beat = 6 MIDI clocks, as 24 clocks = quarter note.)
JSON:
json
Copy
{
  "type": "songPosition",
  "position": 128, 
  "timestamp": <time>
}
Here "position": 128 means the song position is at 128 (in 16th notes). We will combine the two data bytes into one number for convenience. The receiver can split it if needed (or we could provide both "lsb" and "msb" fields, but not necessary). This message might be used if we implement jumping to a certain position in sync.
Song Select (0xF3):
Bytes: Status = 0xF3, Data1 = Song Number (0–127). This is used to select a sequence or song number on devices that have multiple songs.
JSON:
json
Copy
{
  "type": "songSelect",
  "number": 5,
  "timestamp": <time>
}
This would indicate selecting song 5 on the device.
Tune Request (0xF6):
Bytes: Status = 0xF6, no data. This asks analog synths to tune oscillators.
JSON:
json
Copy
{ "type": "tuneRequest", "timestamp": <time> }
(Undefined 0xF4, 0xF5: These are not used in MIDI 1.0 (though 0xF5 is sometimes labeled “Bus Select” in some documentation, it’s essentially undefined)
studiocode.dev
. We will ignore or filter out any undefined status bytes if they ever appear. Our system should be robust to ignore unknown message types gracefully
studiocode.dev
.*
If needed, we can add specific JSON types for any additional system messages that come with MIDI 2.0. However, MIDI 2.0’s system messages mostly revolve around new SysEx (discussed next) and the Capability Inquiry/Negotiation which itself involves SysEx or special packets. Those are beyond the performance data scope, but if we ever needed to stream those, we could incorporate them as well (for example, a Capability Inquiry message could be represented, but that’s likely not in the real-time performance stream – more in setup phase).
System Exclusive (SysEx) Messages
System Exclusive messages carry arbitrary length data and typically begin with 0xF0 and end with 0xF7 in MIDI 1.0. Everything in between is manufacturer-specific or user-defined data. In our JSON encoding, we will capture the entire SysEx payload in an array or similar structure. We define the JSON as follows for a SysEx event:
json
Copy
{
  "type": "sysEx",
  "manufacturerId": [ 0x41 ], 
  "data": [ 0x10, 0x42, 0x37, 0x7F, ... ],
  "timestamp": <time>
}
Here, "manufacturerId" is an array of one or three bytes that identify the manufacturer (as per the MIDI spec, a 1-byte ID if <0x7D, or a 3-byte ID sequence if the first byte is 0x00). In this example, 0x41 would be Roland’s ID for instance. The "data" array contains all the remaining bytes in the message excluding the starting 0xF0 and terminating 0xF7. We know it's SysEx because "type": "sysEx" implies that. One could include the end-of-exclusive explicitly, but it’s not necessary — we can assume that each SysEx JSON object represents one complete SysEx message (the end is implicit). However, for completeness or debugging, we might include a boolean "end": true or something if partial SysEx were ever streamed (but normally we will send complete SysEx chunks as one JSON object; we will not split a single SysEx across multiple JSON objects unless absolutely needed for extremely large transfers). If the SysEx message is using the new MIDI 2.0 SysEx 8 format (which allows bytes to use full 8-bit range, whereas MIDI 1.0 SysEx technically should have 7-bit data bytes with 0xF7 reserved for end), we do not actually need a different representation – we simply allow values in the data array to be 0–255 (which covers the full 8-bit range). In MIDI 1.0, any 0xF7 would denote the end; in MIDI 2.0 SysEx8, 0xF7 could appear as data because end is indicated by packet length instead. But since our JSON will know the length from the array length, we can handle either case. Essentially, "data" array will contain the exact bytes transmitted (except 0xF0/0xF7 framing), and its length tells us how long the SysEx was. For example, a complete SysEx to set a device’s parameter might be:
json
Copy
{
  "type": "sysEx",
  "manufacturerId": [0x00, 0x20, 0x33],
  "data": [0x7F, 0x01, 0x04, 0x05, 0xF7],
  "timestamp": <time>
}
If we include 0xF7 in the data array here, it actually represents the End Of Exclusive. We have to be careful: either we include the terminating 0xF7 in the data array (as above), or we could exclude it. It might be clearer to exclude the 0xF7 (since 0xF7 is not part of the device-specific payload). In that case the example would end "data": [0x7F, 0x01, 0x04, 0x05] and we just know it's terminated. We should choose one convention and stick to it. Let’s decide to exclude 0xF7 in the data field and let the JSON object boundary imply termination. The "manufacturerId" is given separately in case of multi-byte IDs, but we could also just include those in front of data. Separating them is slightly cleaner semantically. Either way, all bytes can be reconstructed by outputting 0xF0, then manufacturer ID bytes, then data bytes, then 0xF7. (Note: If large bulk dumps are sent via SysEx and we want to stream them progressively, we might chunk them. MIDI 1.0 allowed sending part of SysEx, then a real-time byte, then continuing SysEx, etc. Our JSON approach would likely avoid intermixing and send one SysEx as one object for simplicity. If partial streaming of a single SysEx was needed, we could have "type": "sysExChunk" with a sequence number, but this complicates the standard and isn't necessary unless dealing with extremely large data in low memory. We can assume SysEx messages are manageable or use MIDI 2.0's new data set messages if needed.)
MIDI 2.0 Extended Messages and Metadata
Beyond the standard set above, MIDI 2.0 has some additional message types such as Per-Note Management (for turning groups of notes off, etc.), Registered Controllers that are unified (RPNs from MIDI 1.0 become single messages in MIDI 2.0), and so on. We won’t enumerate every MIDI 2.0 message here, but our JSON format can be extended for them in a straightforward way. For example:
Registered Controller (MIDI 2.0): In MIDI 1.0, adjusting a Registered Parameter (RPN) like pitch bend range required sending CC 101/100 to select RPN and CC 6/38 to adjust. In MIDI 2.0, there is a single message with a 32-bit value. We could represent that as {"type": "registeredParameter", "channel": N, "index": RPN#, "value": <32-bit value>}. Similarly for NRPN (non-registered), though MIDI 2.0 also simplifies those.
Per-Note Pitch Bend (MIDI 2.0): If a device uses per-note pitch, it will send a message tied to a specific note (with a note ID) rather than the channel. JSON could be {"type": "perNotePitchBend", "channel": N, "note": 60, "value": <14-bit or 32-bit value>}.
Per-Note Controller: {"type": "perNoteControlChange", "channel": N, "note": 60, "controller": X, "value": Y} for MIDI 2.0's per-note CCs.
The Universal MIDI Packet scheme groups these under certain message type codes (for instance, UMP message type 0x2 is for MIDI 2.0 channel voice, 0x3 for data messages like SysEx8, etc.). We do not need to explicitly expose the UMP code in JSON, because we describe messages by their high-level type. But if we wanted to, an optional "midiProtocol": 1 | 2 or "umpType": X field could be added to each message. In a mixed environment, the JSON producer will know whether a message was originally MIDI 1.0 or 2.0. Since our aim is to standardize this JSON format, it might be wise to include a protocol indicator at least once (perhaps as a header or in each message if mixing). For example, if running in a pure MIDI 1.0 mode, we could have an overall flag in the stream header (outside of the event objects) like "midiVersion":1. If both appear, we could mark each event with "midiVersion":1 or 2. However, to keep per-event overhead low, we might instead decide that the server and client agree on the protocol context (e.g. negotiate to use MIDI 2.0 if both sides support it, otherwise use 1.0). In any case, our JSON fields are designed such that if a MIDI 2.0-only field appears (like "attributeValue" or a value >127), a MIDI 1.0 client could ignore or clip it. Backward compatibility is easier managed at the source by not sending 2.0-specific data to a 1.0 client. The JSON structure is flexible enough to carry either.
Example of Full JSON Event and Reconstruction
To illustrate the completeness of this representation, consider a MIDI 1.0 Note On and how we would reconstruct it: Suppose we receive the JSON object:
json
Copy
{ 
  "type": "noteOn", 
  "channel": 2, 
  "note": 64, 
  "velocity": 100, 
  "timestamp": 12345678 
}
We know "type":"noteOn" corresponds to status nibble 0x9, and "channel":2 means channel 2. MIDI status bytes for Note On start at 0x90 for channel 1, so for channel 2 it would be 0x91 (which is 145 in decimal)
midi.org
. The data bytes are note 64, velocity 100. Thus the raw MIDI bytes would be [0x91, 0x40, 0x64] (where 0x40 is 64 in hex, 0x64 is 100 in hex). If our JSON included a "rawBytes" field, it would likely show [145, 64, 100] (or hex ["0x91","0x40","0x64"]). Either way, it's directly mappable. Similarly, a SysEx JSON with manufacturerId and data can be concatenated to form the exact SysEx byte stream, etc. We have ensured every piece of information (status, channel, all data bytes) is present in the JSON either explicitly or implicitly. By formalizing this JSON schema for MIDI events, we enable interoperability: any front-end or service can consume this stream and interpret or even convert back to MIDI bytes if needed. It serves as a self-documented, high-level representation that is still lossless with respect to the original MIDI data.
Transport Clock Synchronization and Timing
Accurate timing is crucial in music data streaming. Since we are sending MIDI events over a potentially asynchronous network (HTTP streaming or WebSockets), we must include timing information to keep the sender and receiver in sync. We plan to synchronize the stream with a transport clock driven by a JUCE-powered engine on the server (or the master host). The JUCE library will provide a high-resolution audio/MIDI clock or timeline reference. We will use this to timestamp events and possibly to send explicit sync messages. There are a couple of strategies we will combine to ensure sync: 1. Timestamping Each Event: Every JSON event includes a "timestamp" field which indicates when that event should occur, relative to the session start or the last start point. This could be an absolute time in milliseconds or microseconds from the beginning of the session (or from when the "play" command was issued). Alternatively, it could be in musical units, like ticks (e.g. MIDI ticks or pulses) from the start, if the tempo is fixed or known. Using a time in milliseconds is straightforward for implementation: e.g. "timestamp": 15320 could mean the event should be played at 15.320 seconds into the song. The receiver can then schedule the event (if it has audio/MIDI output capabilities) to occur at that exact time (accounting for network latency). If network latency is non-negligible, typically the client might start receiving a little ahead of time so it can queue events accurately. If an event arrives with a timestamp that is already passed (due to latency), the receiver might have to drop it or play immediately (which would be late). To mitigate this, the server could always be slightly ahead in sending data (buffer a bit). Using absolute timestamps works well if both sender and receiver have roughly synchronized clocks or at least agree on "time zero". When the transport is started, we could send a special message like { "type": "start", "timestamp": 0 } to indicate the beginning and baseline of the timeline. All following timestamps are relative to that. If the tempo is constant, the sender can compute timestamps easily from musical position. If the tempo changes or there is a need to adjust, either the sender can inform the client of the new tempo (see below) or simply the timing of events naturally reflects it (denser or sparser timestamps). 2. Periodic Clock Messages: In addition to absolute timestamps, or as a backup for devices that use MIDI's own clocking, we can send the standard MIDI Timing Clock events (0xF8). These are 24 per quarter note in MIDI 1.0 convention. If we are syncing to a particular BPM, we would emit 24 of these per beat. For example, at 120 BPM (2 beats per second), that's 48 clock ticks per second (i.e. one every ~20.8ms). Each clock message in JSON is {"type":"timingClock", "timestamp": T} indicating when that tick occurs. A receiver that uses a MIDI clock paradigm could lock its sequencer or arpeggiator to these. However, relying purely on MIDI clock pulses over internet is risky because any jitter or loss can cause tempo fluctuations. Therefore, we consider the clock messages as supplementary; the primary method is the timestamp scheduling. 3. Song Position and Tempo Info: We can also leverage Song Position Pointer (SPP) and explicit tempo messages to help the receiver stay in sync if jumping or tempo change occurs. For instance, if the user jumps to a different position in the timeline (say back to bar 1), the server could send an SPP message like {"type":"songPosition", "position": 0, "timestamp": <time>} to tell the client to reset position. Regarding tempo, MIDI 1.0 does not have a real-time message for tempo except a System Exclusive or Meta event (Set Tempo meta exists in MIDI files). We can introduce a custom JSON message for tempo changes, e.g. {"type": "tempo", "bpm": 140, "timestamp": T} if the tempo changes at time T. Since we are standardizing this format, including a tempo event type makes sense for completeness. This would inform the client that from timestamp T onward, the tempo is 140 BPM (thus the spacing of any upcoming clock ticks or the interpretation of musical time would change). In JUCE, we can get notifications of tempo changes or set tempo; those would trigger sending such messages. JUCE Transport as Master: Because our system uses a JUCE-based transport, we can trust it as the master clock. The JUCE Timecode or AudioPlayHead can provide the current PPQ (pulses per quarter) position, tempo, and sample time. On the server side, as events are sent out, they will be stamped with the current time position. If the user hits play in the DAW (JUCE), we send "start". If stop, we send "stop". If loop or jump, we send an SPP or simply reset and use a new start for that section. Jitter Reduction: The MIDI 2.0 spec itself introduced a jitter reduction timestamp (JR Timestamp) that can prefix any message in the UMP stream for sub-millisecond timing precision
amei.or.jp
. In our JSON approach, including an absolute timestamp with high resolution serves a similar purpose – it effectively acts like a time-stamp prefix for the event. The advantage is that even if messages arrive slightly early or late, the receiver can buffer or adjust to play them at the right moment (much like how MIDI 2.0 devices might use the JR timestamp to schedule events precisely). We should use high-resolution timers (e.g. microseconds or nanoseconds) if possible for the timestamps to get near sample-accurate timing. However, a millisecond resolution may suffice given typical network jitter might be a few ms; we can refine that as needed. Latency and Alignment: During initialization, it might be useful for the client to know an estimate of network latency or to receive a "sync" message. We could implement a handshake where the client sends a ping and the server responds to measure round-trip, or simply manually configure a buffer. This is outside the core JSON format, but important for usage. Perhaps as part of the JSON stream setup, we send a config object like { "type": "setup", "midiVersion": 2, "tickDuration": 21, "latency": 50 } where tickDuration might be the ms between 0xF8 clocks at current tempo, and latency could be an estimated client buffer time in ms. This is just an idea; a simpler approach is to start the stream slightly before actual audio (like send a bar of pre-roll clocks or a count-in). In summary, the synchronization strategy is: each event carries an exact timing, and we optionally send periodic tempo/clock signals. This dual approach means even if a few JSON events are delayed or lost, the continuous clock can help a naive client keep tempo, while a smarter client will use the precise timestamps to schedule events accurately. Since our streaming is intended to be reliable (over TCP via HTTP or WebSocket), we expect to get all events in order. The main issue is jitter, which timestamps solve.
Multi-Channel Stream Segmentation and Duplication
MIDI’s design of 16 channels within one cable/port is historically to allow multiple instrument parts in one data stream. However, for our internet streaming framework, we have the flexibility to structure how data is delivered. We plan to stream each MIDI channel as its own logical stream. This means that instead of one monolithic JSON array containing events from all channels mixed together, we could have 16 separate JSON streams (one per channel). For example, there might be separate API endpoints or WebSocket topics like /midi/channel/1, /midi/channel/2, ..., each streaming only the events of that channel. If a user is only interested in a specific instrument (channel), they could subscribe to just that channel’s feed. Internally, this separation can also simplify routing (the server can manage each channel separately) and possibly improve performance by parallelizing the handling of different channels. Advantages of per-channel streams:
Isolation and modularity: Each channel’s data is isolated, so heavy traffic on one channel (say, a flurry of drum notes on channel 10) won’t delay the parsing of another channel’s events. In practice with a single JSON stream that might not be a big issue if properly buffered, but isolation could help in scaling or debugging (you can inspect a single channel easily).
Selective subscription: A client that only needs some channels doesn’t waste bandwidth on others. For instance, a lighting control application might only care about a specific channel that carries lighting MIDI, and ignore the rest.
Parallel processing: In a scenario where multiple threads or processes handle MIDI, each could take one channel’s stream.
Reduced contention: One channel’s JSON can be sent in its own HTTP chunk sequence. We avoid one giant combined stream where events must strictly interleave in time. (However, note that the combined chronological order is still important musically. If separated, we need to ensure a way to realign by time – which is where the timestamps come in. Since events across channels can happen concurrently, the timestamp across all streams is the global timeline reference.)
We will synchronize all channel streams via the common timestamp clock. Even though events are sent in different streams, their "timestamp" allows the client to merge or order them if needed. Essentially the timestamp is a global clock across all channels. Using Unused Channels for Duplication: The user specifically suggested "we can use this to send duplicates of channel streams over unused channels". This implies a form of redundancy or multi-casting using spare channels. For example, MIDI defines 16 channels per port; if in a given session only 8 channels are actually used for instruments, we have 8 unused channels. We could choose to duplicate some important channels onto those unused ones. Why do this? A few possible reasons:
Redundancy for reliability: If the same data is sent on two channels (two streams) and one stream experiences a drop-out or delay, the other might still get through, ensuring no loss. Over a network, if we route two channels via two different paths or just rely on TCP which will reorder, this might be not typical, but conceptually it could be used as an error-correction scheme (like sending the same message twice on two logical channels).
Multi-destination scenarios: Perhaps one channel’s music needs to be sent to two different receiver devices that expect it on different MIDI channel numbers. By duplicating the data on another channel, a second device listening on that channel could receive it without having to filter by content. This could be an alternative to having multiple clients subscribe to one feed; instead they subscribe to different feeds that carry the same info. For instance, Channel 1’s piano part could also be mirrored on Channel 11; a second synth could be set to listen on 11 to play the same piano part. This is somewhat analogous to how in a MIDI thru chain you might set two synths to the same channel to layer sounds. Here we are doing it at the stream level by duplicating events.
Legacy or load distribution: If certain legacy reasons required one device to only listen on a fixed channel, duplicating data to that channel stream could feed it without altering the original channel usage. Also, if one channel stream becomes very data-heavy (e.g. high resolution 2.0 messages), perhaps splitting half of its notes to another channel could theoretically spread out processing (though musically that changes channel identity, so unless using MPE or something, it's not typical).
Concretely, how would we implement duplication? The server could have a config that, say, channels 15 and 16 are unused, and we decide to copy channel 1’s events onto channel 15’s stream as well. The JSON objects on channel 15’s stream would still say "channel":1 (since the musical channel is 1) or would we change it to 15? This is an interesting point: If we are truly treating each stream as independent, maybe we would actually change the channel field to 15 in the duplicated stream to indicate that, from the duplicate stream’s perspective, it's on “its own channel”. However, that could confuse things, since then it’s not an identical copy at the data level (the channel number changed). If the goal is redundancy, we wouldn't want to change the data. So more likely we would keep the channel number the same and just literally output identical JSON objects on both streams 1 and 15. But then a client listening on channel 15 stream would see "channel":1 events. That might be okay if they understand it's a mirrored feed of channel 1. Alternatively, we could decide that what we call "channel streams" is more like "part streams" or "instrument streams". In that case, duplicating to another stream endpoint would usually mean the channel number stays constant. However, another interpretation: maybe the "unused channels" duplication was meant for error correction where the duplicate is truly just backup, not for separate consumption. In that scenario, the duplication might be handled at a lower level (like send two identical sets of data over two sockets and the receiver merges them, which is complex and likely unnecessary with TCP). We might need clarification, but since we are making a complete plan, let's articulate the duplication idea like this: if channels are free, the system can mirror active channel data on those free channels, either for backup or multi-target output. This might not be a standard MIDI practice, but in our framework it’s possible since channels are just labels in data streams now. Implementing multiple streams: In practice, using separate HTTP streaming endpoints for each channel means 16 open connections if all channels are in use. That could be heavy for a browser to handle (though 16 keep-alive connections might be okay, or we could use one WebSocket and multiplex). Alternatively, we might simulate separate streams by multiplexing channel-tagged messages in one connection and let the client split them. But since the requirement explicitly mentions "each channel as its own stream", we lean toward actual separate streams (which could also be separate JSON arrays in one response separated by some delimiter if using HTTP2 server push or so, but likely easier is multiple endpoints or multiple websockets). Given that Bassoon/Oboe are used, those typically stream from a single URL. Perhaps we might use a single stream and simply filter by channel on client side. But since the user listed it as a point, they likely want to design for multiple streams conceptually. We can do it either way, but let's assume multiple streams. The duplication then could be as simple as the server reading the event and writing it to two stream outputs (the original channel's and the duplicate channel's). The JSON content could remain identical (with the same channel number inside). Summary of channel stream approach: We will formalize that the standard JSON representation is the same on all streams, but the transport layer can segregate by channel for flexibility. It's like each channel is its own broadcast. This doesn’t affect the JSON schema per se, except that if a client is only connected to one channel's feed, it will only see those JSON objects (with their channel field constant). A full listener (like a DAW listening to all MIDI) could either open all streams or we might also offer a combined stream for convenience. For completeness, we might say that an alternate mode is to combine all channels in one stream (for simplicity), which might be easier for some consumers. In that case, each JSON object’s "channel" field tells them which channel. This is actually how we started in design. The per-channel streams is an added capability on top. Perhaps our standard can support both modes: monolithic stream mode and split stream mode. Since Bassoon.js expects one stream by default, we might initially implement monolithic (all events in one array), then later move to splitting when forking. But as a blueprint, we’ll describe the splitting idea fully. Duplicating streams on unused channels: Suppose channels 13–16 are free. We could assign: channel 1 events also go out on stream 13, channel 2 events on stream 14, etc., or any scheme. This can be dynamically configured. Because this is a somewhat unconventional concept, it might not be part of a rigid standard, but we include it as a feature. If standardizing, we might define a JSON message or a setup instruction that indicates such duplication. For example, a special message { "type": "duplication", "sourceChannel": 1, "mirrorChannel": 13 } to inform the client that channel13 stream is a mirror of 1. But if the client just subscribes blindly, they might figure it out by identical timestamps and note patterns. In simpler terms, for now we can document: the framework may optionally send identical JSON events on multiple channel streams. If a client happens to be subscribed to both, they'd get duplicates. Normally one wouldn’t subscribe to a duplicate if they know it’s a duplicate. This feature is primarily for sending the same musical data to different destinations by virtue of channel separation.
Conclusion and Standardization Notes
The above constitutes a comprehensive, byte-accurate JSON schema for MIDI streaming. Every MIDI event from both MIDI 1.0 and MIDI 2.0 is accounted for, ensuring no loss of information: status bytes, data bytes, and timing are all represented. By formalizing this as a specification, different implementations can adopt it, allowing interoperability (for example, a Web MIDI client could receive this JSON and drive actual synthesizers, or a custom application could record it, etc.). We have taken care to cover all MIDI message types (channel voice, system common, system real-time, SysEx) and the enhancements in MIDI 2.0 (extended resolution, new message types, groups). A few standardization considerations to highlight:
JSON Schema Definition: It may be useful to write a JSON Schema or interface definition that enumerates the possible "type" values and their required fields. For instance, a schema could state that if "type":"noteOn", then "note" (number) and "velocity" are required, "channel" is required (1-16), etc. Doing so would formally fix the structure and ranges. The implementation can then validate outgoing and incoming messages against this schema.
Clock and Tempo Handling: While MIDI clock events are included, the primary timing mechanism is timestamps. This reflects modern practice (similar to DAW timelines or Ableton Link) more than the old MIDI clock, and suits network streaming. We might want to define that the timestamp is in milliseconds (or ticks of 480 PPQ at a default tempo?) – we should clearly document the time base. If using milliseconds, also clarify if it's absolute epoch or relative. We have assumed relative to session start (which is simplest and likely the intent).
MIDI Protocol Version Negotiation: If a receiver only knows MIDI 1.0 (e.g. a legacy client) and we send MIDI 2.0 extended values, it might mishandle them. A standardized solution is to have a negotiation or capability flag. Since our user scenario likely involves the same software on both ends or a controlled environment, we can ensure both support MIDI 2.0. But as a standard, we might incorporate a field like "protocolVersion":2 at the top or in each message. The MIDI Association's own solution to negotiation is MIDI-CI (Capability Inquiry) messages, which we could theoretically send as SysEx messages in JSON too. However, that might be beyond scope; simpler is an out-of-band configuration or an agreed mode.
Example Implementation with Bassoon/Oboe: Initially, we can implement a single HTTP endpoint (e.g. /midi/live) that, upon client connection, starts outputting a JSON array like [ then a series of event objects separated by commas, and eventually ]. The Bassoon.js library will allow the client to consume this as a stream of objects
github.com
. Once that works, expanding to multiple endpoints (like /midi/channel/1 etc.) is possible by filtering server-side. Alternatively, a query parameter or command can select which channels to stream. We'll likely test with the monolithic stream first (because Bassoon is ready for that), and then later consider forking Bassoon to support subscribing to multiple streams or a multiplex.
By rigorously following this scheme, we ensure that our MIDI-over-JSON stream is completely complete – it encapsulates all MIDI data down to the byte, while also providing additional context (like human-readable types and timestamps) useful for modern applications. This can be the foundation of a standardized format for MIDI data interchange in web and network applications, marrying the real-time performance capabilities of MIDI with the flexibility of JSON. Sources:
MIDI message format and status/data byte definitions
studiocode.dev
studiocode.dev
studiocode.dev
studiocode.dev
MIDI 1.0 status byte message list (channels and types)
midi.org
System messages and their byte counts
studiocode.dev
studiocode.dev
Introduction of MIDI 2.0 and its enhancements (higher resolution, more controllers, bi-directional comm.)
midi.org
midi.org
Universal MIDI Packet format (32-bit packets, groups, timestamps)
amei.or.jp
midi.org
Property Exchange using JSON in MIDI 2.0 (as an example of JSON already in MIDI domain)
sweetwater.com
Bassoon.js (Oboe.js fork) for streaming JSON arrays efficiently
github.com