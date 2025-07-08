const { _electron: electron } = require("playwright");

async function testJAMCam() {
  console.log("Starting JAMCam test...");

  // Launch the Electron app
  const electronApp = await electron.launch({
    args: ["src/main.js"],
    timeout: 10000,
  });

  // Get the first window
  const window = await electronApp.firstWindow();

  // Wait for the app to load
  await window.waitForLoadState("domcontentloaded");
  await window.waitForTimeout(3000); // Give time for initialization

  console.log("App loaded, checking initial state...");

  // Check if transmit mode is active by default
  const transmitBtn = await window.locator("#transmit-mode");
  const isTransmitActive = await transmitBtn.evaluate((el) =>
    el.classList.contains("active")
  );
  console.log("Transmit mode active:", isTransmitActive);

  // Check input source dropdown
  const inputSource = await window.locator("#input-source");
  const selectedValue = await inputSource.inputValue();
  const options = await inputSource.locator("option").allTextContents();
  console.log("Input source options:", options);
  console.log("Selected value:", selectedValue);

  // Check if video is showing
  const noVideoElement = await window.locator("#no-video");
  const videoElement = await window.locator("#video-frame");

  const noVideoVisible = await noVideoElement.isVisible();
  const videoVisible = await videoElement.isVisible();

  console.log("No Video message visible:", noVideoVisible);
  console.log("Video element visible:", videoVisible);

  // Check if video has a source
  const videoSrcObject = await videoElement.evaluate(
    (el) => el.srcObject !== null
  );
  console.log("Video has source:", videoSrcObject);

  // Check console logs for any errors
  const logs = [];
  window.on("console", (msg) => {
    logs.push(`${msg.type()}: ${msg.text()}`);
  });

  // Wait a bit more to see if video starts
  await window.waitForTimeout(2000);

  console.log("Console logs:", logs);

  // Test result
  const success =
    isTransmitActive &&
    !noVideoVisible &&
    videoVisible &&
    videoSrcObject &&
    selectedValue !== "";
  console.log("TEST RESULT:", success ? "PASS" : "FAIL");

  if (!success) {
    console.log("ISSUES FOUND:");
    if (!isTransmitActive) console.log("- Transmit mode not active by default");
    if (noVideoVisible) console.log('- "No Video" message is showing');
    if (!videoVisible) console.log("- Video element not visible");
    if (!videoSrcObject) console.log("- Video element has no source");
    if (selectedValue === "") console.log("- No camera selected by default");
  }

  // Take a screenshot for debugging
  await window.screenshot({ path: "jamcam-test.png" });
  console.log("Screenshot saved as jamcam-test.png");

  // Close the app
  await electronApp.close();

  return success;
}

// Run the test
testJAMCam().catch(console.error);
