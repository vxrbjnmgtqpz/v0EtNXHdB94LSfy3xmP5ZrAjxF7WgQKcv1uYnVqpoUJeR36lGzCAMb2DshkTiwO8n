# PNBTR+JELLIE Training Testbed — Current GUI Layout (Text Mock-up)

```
+-----------------------------------------------------------------------------------+
| [Transport Bar: ProfessionalTransportController]                                  |
|   [Play] [Pause] [Stop] [Record]  SESSION TIME: 00:00:00.000.000  BARS: 1.1.1     |
|   BPM: 120.0 [slider]    :: Packet Loss [%] [slider]  Jitter [ms] [slider]        |
+-----------------------------------------------------------------------------------+
| [Oscilloscope Row: OscilloscopeRow]                                               |
|   Input      |TOAST Network|  Log/Status        |  Output                         |
|   [osc]      |  [osc]      |  [log/status box]  |  [osc]                          |
+-----------------------------------------------------------------------------------+
|  [Audio Track with Spectral Analysis]                                             |
|                                                                                   |
|   JELLIE Track (Recorded Input)                                                   |
|                                                                                   |
+-----------------------------------------------------------------------------------+
| [Audio Track with Spectral Analysis]                                              |
|                                                                                   |
|   PNBTR Track (Reconstructed Output)                                              |
|                                                                                   |
+-----------------------------------------------------------------------------------+
| [Metrics Dashboard: MetricsDashboard]                                             |
|   [metrics: SNR, latency, packet loss, etc.]                                      |
+-----------------------------------------------------------------------------------+
```
