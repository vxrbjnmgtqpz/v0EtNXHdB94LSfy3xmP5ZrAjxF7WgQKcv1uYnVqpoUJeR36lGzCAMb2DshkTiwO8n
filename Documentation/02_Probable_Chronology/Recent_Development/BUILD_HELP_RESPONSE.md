Understood — I’ll perform a full diagnostic sweep of the TOASTer app in the MIDIp2p repo to identify legacy code issues, missing symbols, and anything broken during recent API dependency removal or refactor.

I’ll return with a precise, step-by-step recovery plan aligned with Pre-Phase 4 build architecture.


# Restoring the TOASTer Build (MIDIp2p Project)

## Overview of the Build Issue and Removed Dependencies

The **TOASTer** application in the MIDIp2p project is failing to produce a runnable executable even though compilation succeeds. After recent refactoring (Pre-Phase 4), all legacy frameworks (e.g. **JAM\_Framework\_v2** and **JMID\_Framework**) were removed. The source now relies only on JUCE modules (mainly `juce_gui_basics` and other essential JUCE modules). The CMake configuration completes without error and the code compiles, and an app bundle (`TOASTer.app`) is generated – but **the `Contents/MacOS` folder inside the bundle is empty**, meaning no actual executable was linked in.

This suggests a **linking or configuration issue**: the build likely fails at the final link stage or doesn’t properly copy the binary into the bundle. Common symptoms include missing symbols (if some functions from the removed frameworks are still referenced), misconfigured build targets, or an entry-point issue (no `main()` defined for the GUI app). Below, we identify the root causes and then present a step-by-step plan to fix the build, modernize the code, and realign with the intended architecture.

## Root Causes Identified

**1. Broken Dependencies:** With JAM\_Framework\_v2 and JMID\_Framework eliminated, any references to classes or functions they provided could cause unresolved symbols at link time. Even if you removed `#include` directives and usage in code, there may be *implicit* dependencies left in the build system (e.g. CMake still trying to link against their libraries or looking for their headers). If the code still declares something provided by those frameworks (perhaps via a forward declaration or leftover object files), the linker would fail to find the implementation. This can lead to a silent build failure where the `.app` is created but the binary is never linked in.

**2. Legacy Code Fragments:** Some code may still be assuming the presence of the old frameworks. For example, the `MainComponent` or other UI code might contain remnants (initialization calls, callbacks, or classes) related to the old APIs. Such fragments can either cause compile warnings or simply do nothing now, but could also result in logical mismatches. They need to be either removed or updated to use JUCE equivalents. Since the code compiles “clean,” it implies most obvious remnants have been removed, but subtle ones (like default parameters, #defines, or resource references) could remain.

**3. Missing Entry Point or Symbols:** A very common cause of a GUI app bundle with no executable is a missing **`main()`** symbol. In a JUCE GUI application, the `main` function is usually provided by the `START_JUCE_APPLICATION` macro. If this macro isn’t being used (or was accidentally removed during refactoring), then the app’s entry point might not be linked, causing the linker to produce no output. Another possibility is that the project’s CMake target is misconfigured such that it doesn’t treat the target as an *OS X GUI app* with a proper main. On macOS, if the target is a **Console** app instead of a **GUI bundle**, the bundle might be assembled incorrectly. In one similar case on the JUCE forum, a user built a JUCE example and found the app’s `Contents/MacOS` empty – it turned out the build setup was wrong for a GUI app (the target was treated like a plugin or had no `START_JUCE_APPLICATION`, leading to no actual binary).

**4. CMake Target Mismatch:** The CMake configuration was “simplified to minimal JUCE dependencies”, which is good, but we must ensure it still properly defines the app target. If the project switched from using a custom framework to JUCE’s CMake, it should use JUCE’s CMake helpers. For example, using `juce_add_gui_app` is the recommended way to set up a GUI application target in JUCE 8. This function automatically sets the target as a macOS app bundle and attaches the necessary JUCE frameworks. A misconfigured CMakeLists (for instance, using `add_executable` without the `MACOSX_BUNDLE` property or not linking all required JUCE modules) could result in an app bundle shell without an executable. In summary, the build system changes might have left out critical settings, causing the linking phase to be skipped or fail quietly.

**5. Mismatches from Removing External APIs:** The removal of external APIs might have changed how different parts of the system interact. For example, if JMID was handling MIDI I/O or timing (perhaps PNBTR is related to network/MIDI timing), the TOASTer app might have had code expecting those services. Now that those are gone, certain functions may effectively be no-ops. This can be acceptable for a build (since they’re removed), but if any were only partially removed (say, declared but not defined anymore), that’s a link error. It’s also possible that the **architectural “Phase 4”** expects certain modules (PNBTR, JMID) to be in place of the old frameworks. Until those are fully integrated, we might need temporary stubs to satisfy the linker and allow the app to run for testing.

With these causes in mind, the focus is to fix the build configuration and any lingering code issues. The goal is a working **TOASTer GUI application** using JUCE 8, with only minimal necessary dependencies, so that PNBTR and JMID functionality can be tested going forward.

## Plan to Restore and Modernize the TOASTer Application

Below is a detailed, actionable plan addressing each aspect of the fix:

### 1. Fixing the Build Configuration (JUCE/CMake) to Produce an Executable

* **Use JUCE’s CMake Functions:** Update the TOASTer `CMakeLists.txt` to use the JUCE CMake API for GUI apps. The simplest way is to use the `juce_add_gui_app` function provided by JUCE. This will create an executable target marked as a GUI app bundle. For example:

  ```cmake
  juce_add_gui_app(TOASTer 
      PRODUCT_NAME "TOASTer" 
      BUNDLE_ID com.yourdomain.toaster
      COMPANY_NAME "Your Name/Company")
  ```

  This sets up the **TOASTer** target as a macOS app bundle with the given product name and bundle ID. It will also auto-generate an Info.plist and include the `START_JUCE_APPLICATION` boilerplate if needed. You can refer to JUCE’s own example CMake for a GUI app: it uses `juce_add_gui_app` to add the target and mark it as an app bundle.

* **Attach Source Files and Headers:** Ensure all your source files (e.g. `Main.cpp`, `MainComponent.cpp`, etc.) are added to the target. In CMake, after creating the target, use `target_sources`:

  ```cmake
  target_sources(TOASTer PRIVATE 
      Source/Main.cpp 
      Source/MainComponent.cpp 
      Source/MainComponent.h)
  ```

  Double-check that **Main.cpp** is included – if it was omitted, the app would have no `main()` and thus no executable code to link.

* **Verify the Entry Point:** Open `Source/Main.cpp` and confirm it contains the macro to start the app. It should have something like:

  ```cpp
  START_JUCE_APPLICATION (ToasterApplication) 
  ```

  where `ToasterApplication` is your subclass of `juce::JUCEApplication` or `juce::JUCEApplicationBase`. This macro generates the required `main()` entry point on each platform. If this line is missing (perhaps removed inadvertently), add it back referencing the proper Application class name. Without this, the linker would have no `main()` for the GUI app (leading to a missing executable). On Windows, a similar issue occurs if the subsystem is wrong (no WinMain), but on macOS it’s typically the missing `main` symbol. Ensuring the macro is present fixes that. (In a JUCE *plugin* project you wouldn’t use this macro, but in a standalone GUI app you **must** have it.)

* **Link Against the Correct JUCE Modules:** With JUCE 8, modules are individually linked. Since you want minimal dependencies, start by linking only what’s needed:

  * At minimum, a GUI app needs **juce\_gui\_basics** (for basic GUI components) plus the modules it depends on (JUCE’s CMake will pull those in automatically if you use the highest-level module). Often, using **juce\_gui\_extra** is convenient because it includes juce\_gui\_basics and other core modules transitively. For example, the JUCE GUI app example links against `juce::juce_gui_extra` and notes that *“inter-module dependencies are resolved automatically, so juce\_core, juce\_events, etc. will also be linked”*.

  In your `CMakeLists.txt`, do:

  ```cmake
  target_link_libraries(TOASTer PRIVATE 
      juce::juce_gui_extra
      juce::juce_recommended_config_flags
      juce::juce_recommended_lto_flags
      juce::juce_recommended_warning_flags)
  ```

  This will link the GUI extra module (which covers GUI basics, graphics, core, etc.) and also apply JUCE’s recommended compiler/linker settings for warnings, LTO, and config. Using `juce_gui_extra` is optional if you truly only need `juce_gui_basics`, but note that if you *only* link `juce::juce_gui_basics`, you should also link the modules it depends on (like `juce::juce_graphics`, `juce::juce_core`, `juce::juce_events`, etc.). Linking `juce_gui_extra` is a convenient way to get them all at once.

  *Verify that no references to the removed JAM/JMID libraries remain in the link libraries.* Your CMake should **not** be linking against any old framework or library that no longer exists. Removing those from `target_link_libraries` (or old `find_package` calls) was likely done, but double-check.

* **Enable Verbose Linking Output:** As a diagnostic step, run the build with verbose output (e.g. `make VERBOSE=1` or the equivalent in your build system) to see the linker command. This will show if the linker is being invoked at all, and if so, what libraries and objects it’s using. If you see references to any missing libraries or undefined symbols, you can act accordingly:

  * Undefined symbol errors will be shown here if any. For example, if something from JMID was still being referenced, the verbose output might show an unresolved symbol name. That would confirm a **missing symbol** cause.
  * If the verbose output shows **no linker command at all** for the app target, it means CMake possibly thought the target was up-to-date or there was nothing to link. This could happen if no source files were tied to the executable target or if the target was never properly defined as an executable. In that case, revisiting the `juce_add_gui_app`/`add_executable` setup as above is critical.

* **Mac Bundle Settings:** Normally, `juce_add_gui_app` will take care of marking the target as a macOS bundle and generating Info.plist. If you opt to not use `juce_add_gui_app` and instead use raw CMake commands, ensure you set:

  ```cmake
  add_executable(TOASTer MACOSX_BUNDLE ...)
  ```

  and also set the **MACOSX\_BUNDLE** properties (bundle name, Info.plist). However, since JUCE 8.0.4 is in use, it's simpler to leverage JUCE’s built-in CMake integration to avoid mistakes. The JUCE CMake API will handle the bundle creation, Info.plist, and copying the compiled binary into the bundle automatically (when properly configured).

* **Alternative Approach (Xcode project generation):** If after the above changes the CLI build still fails silently, try generating an Xcode project with CMake and building in Xcode:

  ```bash
  cmake -B build_xcode -G Xcode .
  ```

  Then open the `.xcodeproj` and build. Xcode’s UI might show any linker errors more visibly or at least confirm whether the app binary is being produced. In some cases, building via Xcode can sidestep subtle issues (like environment path or toolchain differences). If the Xcode build succeeds (producing a working `TOASTer.app` with a binary), it indicates something was off in the Makefile build configuration, which the above steps should resolve. (For instance, Xcode might automatically find libraries or set RPATHs that Make might not without explicit config.)

* **Common JUCE App Bundle Issues:** Be aware of a couple of common issues that can result in an empty app bundle:

  * If a post-build step fails (like code-signing or copying resources), the build might not error out but the binary might be deleted or not moved. For initial testing, disable any code-signing or packaging steps. In JUCE’s CMake, there isn’t usually an automatic signing step unless you added one. Just make sure no custom CMake script is removing or moving the executable.
  * If you’re building a **Universal Binary** (arm64 + x86\_64) on Apple Silicon, ensure all linked libraries (including JUCE compiled objects) are available for both architectures. A lopsided universal build could cause the linker to drop the output. If not needed, you can configure CMake to build only native arch (e.g., set `CMAKE_OSX_ARCHITECTURES=arm64` for Apple Silicon only) to simplify debugging. You can add this in your CMakeLists or when calling CMake.
  * Ensure the deployment target is set to a reasonable value (e.g. macOS 10.13 or higher, as JUCE suggests). While an outdated deployment target likely wouldn’t cause a missing binary, mismatched SDK settings could cause linking issues on newer macOS. Setting `-DCMAKE_OSX_DEPLOYMENT_TARGET=10.13` (as recommended by JUCE) is a good practice.

By addressing the above, the **expected outcome** is that a successful build will produce `TOASTer.app` **with a `Contents/MacOS/TOASTer` executable inside**. At that point, running the app (e.g. `open TOASTer.app` or `./TOASTer.app/Contents/MacOS/TOASTer` from Terminal) should at least launch the application (even if it’s just a blank window or basic UI for now).

### 2. Eliminating or Modernizing Outdated Code Fragments

With the build system fixed, turn to the code itself to remove any vestiges of the old frameworks and update the app to modern JUCE practices:

* **Purge References to Removed APIs:** Search the codebase (especially in `MainComponent.*`, `Main.cpp`, and any module related to JMID or JAM) for any lingering mentions of `JAM_` or `JMID_` classes, functions, or macros. These should either be deleted or replaced. For example, if `JMID_Framework` provided a MIDI device list or a MIDI message class and you still have code trying to use it, switch to JUCE’s classes (`juce::MidiMessage`, `juce::MidiOutput`, etc.) or stub that functionality out for now. Since the removal was already done and *“source files: clean, no compilation errors”*, you likely have already commented out or removed these references. Just double-check that none remain in comments, unused variables, or inadvertently included headers.

* **Remove Legacy UI Constructs:** If the JAM framework was responsible for any UI components or helpers (for example, custom knobs, windows, or an application manager), replace those with JUCE equivalents:

  * Ensure that `MainComponent` is a subclass of `juce::Component` (or `juce::AudioAppComponent` if audio is involved) and not any custom base class from the old framework.
  * If the old framework had its own event loop or initialization calls (common in frameworks), remove them – JUCE’s `JUCEApplication` will handle the event loop and app init/shutdown via `initialise` and `shutdown` callbacks in your Application class.
  * Remove any redundant code that was working around the old frameworks. For example, if there were global singletons or managers from JAM/JMID, those are likely gone. If the app needs certain singletons (like a MIDI device manager), consider using JUCE’s `juce::MidiKeyboardState` or `juce::AudioDeviceManager` as needed.

* **Modernize API Usage:** Since JUCE 8.0.4 is in use, ensure that your code uses updated JUCE APIs:

  * Replace any deprecated calls (check JUCE 8 release notes if you used older JUCE before). For instance, if `MainComponent` was using older classes or methods that JUCE 8 changed, update those. One example might be how audio/MIDI is initialized or how `AudioProcessorValueTreeState` is used – if any such outdated patterns exist, update them to the current recommendations.
  * Remove any `JUCE_LEAK_DETECTOR` macros pointing to removed classes, or any `ScopedPointer` usage (modern JUCE uses `std::unique_ptr`).
  * If you had custom `std::thread` or other concurrency from old frameworks, consider using JUCE’s `juce::Thread` or `juce::ThreadPool` if appropriate (not mandatory, but for consistency).

* **Minimize Dependencies:** You mentioned the code now only depends on `juce_gui_basics` and essential modules. Double-check that in your AppConfig or Prefix header you’re not accidentally including unnecessary JUCE modules (e.g., `juce_audio_plugin_client` or any plugin-related stuff if this is now a standalone app). Keep the JUCE module list lean – this reduces the chance of needing additional frameworks (for example, if you don’t need web browsing or audio, you can disable those JUCE features to avoid linking WebKit or Apple AudioToolbox, etc.). In the JUCE CMake example, they explicitly disable the WebBrowser and CURL features for a minimal app. You can do similarly by adding in CMake:

  ```cmake
  target_compile_definitions(TOASTer PRIVATE 
      JUCE_WEB_BROWSER=0 
      JUCE_USE_CURL=0)
  ```

  This ensures you don’t accidentally pull in WebKit or libcurl as dependencies, which aligns with keeping the app lightweight.

* **Test a Basic GUI:** Since even a basic "Hello World" GUI was failing before, after the above build fixes, try running a trivial UI: e.g., have `MainComponent` just draw a background and a label. If that runs, you know the JUCE GUI pipeline is functioning. At that point, any further crashes or issues would be due to logic, not the build. This step is more about verification than code change – it ensures that nothing in the remaining code (like callbacks or threads from old frameworks) is causing runtime issues. If something does crash or misbehave, strip the `MainComponent` down to basics and add pieces back until you find the culprit.

### 3. Ensuring Alignment with **Pre-Phase 4 Architecture**

“Pre-Phase 4” likely refers to the state of the system after removing external dependencies but before introducing whatever new Phase 4 features are planned. To align with this architecture:

* **Modularize PNBTR and JMID:** If PNBTR and JMID are now intended to be internal modules or libraries (instead of external frameworks), treat them as such in the project structure. For example, you might have `PNBTR` and `JMID` as separate directories or components in the repository. Make sure their code is integrated:

  * If they are header-only or source modules, include their source files in the build (via `target_sources` or as static libraries).
  * If they were meant to be static libs, you can create static library targets for them in CMake (e.g., `add_library(PNBTR STATIC ...)` and similarly for JMID) and link those to TOASTer.

* **Decouple UI from Backend:** The TOASTer GUI should interface with PNBTR/JMID through clean APIs. Ensure that removing the old frameworks hasn’t left the UI without a way to call into the MIDI or network functionality. For now, if PNBTR or JMID code is incomplete due to the refactor, consider **stubbing** their interfaces:

  * For instance, if `PNBTR` is supposed to provide a tempo or beat-synchronization service (just as an example), and that was previously handled by JAM, create a placeholder `PNBTRManager` class with the same interface but minimal implementation. The UI can call this without crashing, even if it just logs or returns dummy values. This will allow the app to run and the interactions to be tested at least at a high level.
  * Similarly, if `JMID` was a MIDI I/O layer, ensure that you have something in place to list MIDI devices or send a MIDI message, even if it’s just using JUCE’s MIDI classes directly for now. The key is that **the absence of the old framework does not break the architectural flow** – each piece has some implementation, even if simplified.

* **Update Documentation/Comments:** As part of aligning with the new architecture, update any file headers, README notes, or comments that referred to the old frameworks. This helps prevent confusion. For example, if `MainComponent.h` had a comment “Uses JMID for MIDI messaging – TODO Phase 4: replace with internal implementation,” update it to reflect the current status (perhaps “JMID integration removed; now uses JUCE MIDI classes directly.”). This is a minor step but keeps the project coherent.

* **Architectural Consistency:** Ensure that **Phase 4** plans (if known) are accounted for. For example, if Phase 4 will introduce a new networking API to replace JAM, make sure nothing in the current codebase would conflict with that (e.g., no leftover hard-coded ports or singletons). The project at this stage should be cleanly separated: the UI (TOASTer) is just a GUI and maybe logic orchestrator, PNBTR handles networking/timing, and JMID handles MIDI. The build should reflect that modularity (separate CMake targets or at least separate source groups).

In summary, at Pre-Phase 4 completion, TOASTer should be a **lean JUCE-based application**, free of external cruft, ready to accept new Phase 4 components. By removing outdated fragments and stubbing where necessary, we prevent crashes and keep the testability of the app.

### 4. Files, Headers, and CMake Targets Needing Attention

From the description, key files to review were:

* **`TOASTer/CMakeLists.txt`:** This is the most critical. Apply the changes discussed: use `juce_add_gui_app`, link proper modules, remove any references to `JAM_Framework_v2` or `JMID_Framework`. Also, ensure any **CMake targets** added for those frameworks are removed. For example, if there were lines like `add_subdirectory(JAM_Framework_v2)` or `find_package(JMID ...)`, delete them. The only dependencies should be JUCE and possibly your own internal modules (PNBTR/JMID as internal targets if they exist). After modifications, the CMakeLists should resemble a standard JUCE GUI app setup (similar to JUCE’s examples) and include only the essential libraries.

  Also verify the **target name vs product name** in CMake: If your target is named "TOASTer", the product (executable) name can be set to "TOASTer" as well (or something user-friendly). Mismatches here can sometimes be confusing (e.g., a target named in all caps vs an actual binary name in Info.plist). Using `PRODUCT_NAME "TOASTer"` in `juce_add_gui_app` ensures consistency.

* **`Source/Main.cpp`:** Check for the `START_JUCE_APPLICATION` macro as noted. Additionally, confirm that `Main.cpp` isn’t defining its own `main()` manually (sometimes console apps do this). In a JUCE GUI app, you should **not** manually define `int main()`, because the JUCE macro does it (and on macOS it may need to be a special signature for GUI apps). If you have leftover console `main()` code (perhaps from testing the console version), guard it or remove it for the GUI target. For example, you might have had something like:

  ```cpp
  #ifdef TOASTER_CONSOLE_BUILD
  int main() { ... }
  #endif
  ```

  Make sure that for the GUI build, that segment is excluded and the JUCE app startup is used instead.

* **`Source/MainComponent.h/cpp`:** Review this for any includes of removed frameworks or usage of now-nonexistent classes. Ensure it only includes `JuceHeader.h` (which in JUCE 8 is typically generated or you can include `<JUCE/juce.h>` depending on setup). Remove any `using namespace XYZ` that came from old frameworks. If there are UI elements that were placeholders for old functionality (e.g., a button that was supposed to trigger a JAM framework action), you can leave the UI element but update its callback to either do nothing or use a new mechanism. For instance, if there was a “Connect” button that used JAM’s networking, you could temporarily disable it or have it call into PNBTR’s stub (logging “Not implemented” for now).

* **PNBTR / JMID source files:** If these exist as part of the project, check their includes and symbols. For example, if `PNBTR.cpp` still includes `<JAM_Network.h>`, that’s wrong now – it should either include a new internal header or `<JuceHeader.h>` if it uses JUCE networking (JUCE’s `StreamingSocket`, etc., could replace some network functionality). If JMID was a framework, perhaps now you have a `MIDIManager` class – ensure it doesn’t rely on any external API and uses JUCE’s MIDI I/O (`juce::MidiInput::getAvailableDevices()`, etc.). Update the CMakeLists for these if needed (e.g., add `target_link_libraries(PNBTR PRIVATE juce::juce_core juce::juce_events)` if PNBTR uses JUCE, or mark them INTERFACE if they’re header-only modules).

* **Headers:** Check any global header or precompiled header file (sometimes projects have a `AppConfig.h` or similar from Projucer days). Remove any defines related to the old frameworks. Also confirm that JUCE module configuration is correct (e.g., if you manually define `JUCE_MODULE_AVAILABLE_*` or other flags, update those).

* **CMake Targets:** Besides the main TOASTer target, ensure no orphan targets remain. For instance, if there was a `JMID_Library` target that’s now deleted, make sure it’s fully removed from the build scripts. On the flip side, if PNBTR and JMID are needed, add targets for them:

  * You might create `add_library(PNBTR STATIC ...sources...)` and then `target_link_libraries(TOASTer PRIVATE PNBTR)`.
  * Or, if their code is tiny, just compile them as part of TOASTer (add to `target_sources`).

  The choice depends on how you want to structure it. The key is that **all needed code gets compiled and linked**, and nothing is left out. An empty app bundle could result if, say, the TOASTer target had no source files of its own and you accidentally only built static libs without linking them – but the earlier steps ensure TOASTer has sources. Still, linking the internal libs is important so their symbols are included in the final binary.

* **Resource Files and Others:** If TOASTer had resources (images, MIDI files, etc.), ensure they are handled after the refactor. The CMake example for JUCE shows how to use `juce_add_binary_data` if needed. If you removed JAM/JMID, maybe you also removed some resource loading those frameworks did. Verify that if the app expects certain files (like a default config or preset), those are still being copied or included. Missing resources won’t usually stop the build, but could cause runtime issues.

In summary, **focus on the CMakeLists and Main.cpp** for the build fix, and on MainComponent plus any new PNBTR/JMID classes for code fix. Once these files are corrected, the project structure will be sound.

### 5. Step-by-Step Recovery Instructions

Finally, here is a step-by-step guide to implement the above and recover the build:

1. **Backup Current State:** Before heavy changes, commit your current project or make a backup. This way you can always compare or rollback if needed.

2. **Integrate JUCE CMake Updates:** Edit `TOASTer/CMakeLists.txt`:

   * Add the JUCE project setup (e.g., `find_package(JUCE REQUIRED)` or `add_subdirectory(JUCE)` if JUCE is a submodule).
   * Use `juce_add_gui_app` for the TOASTer target with the appropriate name, product name, company, etc.
   * Add `target_sources(TOASTer PRIVATE ... )` for all source files.
   * Add `target_compile_definitions(TOASTer PRIVATE JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0)` to disable unneeded features (optional but recommended for minimal dependencies).
   * Add `target_link_libraries(TOASTer PRIVATE juce::juce_gui_extra juce::juce_recommended_config_flags juce::juce_recommended_lto_flags juce::juce_recommended_warning_flags)`. This links JUCE modules and sets good defaults.
   * Remove or comment out anything related to the old frameworks (library links, include paths, etc.).
   * If PNBTR and JMID are separate components now, add them accordingly (e.g., if you have `PNBTR/CMakeLists.txt`, include it or directly compile those files).
   * Set CMake properties like `CMAKE_OSX_DEPLOYMENT_TARGET` if not set (e.g., add `set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13" CACHE STRING "" FORCE)` for compatibility).

3. **Ensure `Main.cpp` is Correct:** Open `Main.cpp` and verify it contains your `JUCEApplication` subclass and the `START_JUCE_APPLICATION` macro call. It should look roughly like:

   ```cpp
   class ToasterApplication  : public juce::JUCEApplication { ... };

   START_JUCE_APPLICATION (ToasterApplication)
   ```

   If anything else (like an old `main()` or references to a console mode) is present, remove those for the GUI build configuration. This step ensures the app has an entry point.

4. **Clean and Rebuild:** It’s a good idea to do a clean build after these changes. Delete the build output directory (e.g., `rm -rf build` or use a fresh build folder). Run CMake configure and build again:

   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=Debug .
   cmake --build build -- VERBOSE=1
   ```

   Watch the output. You should see compiler commands for each source file (Main.cpp, MainComponent.cpp, etc.), and then a **linker command** that creates an executable (likely something involving `c++ ... -o TOASTer.app/Contents/MacOS/TOASTer ...`). If the linker command runs without errors, and especially if you see `Linking CXX executable TOASTer.app/Contents/MacOS/TOASTer`, that’s a great sign. If there are errors, address them:

   * Undefined references: figure out which symbol – if it looks like a JUCE symbol, you might have missed a module (add the module in `target_link_libraries`; e.g., undefined `juce::WebInputStream` means you need `juce_networking` or enabling CURL). If it’s something from your code (e.g., a function from PNBTR), ensure that object file is linked or the function is implemented.
   * Missing frameworks: if the linker says it can’t find a framework (e.g., WebKit), disable that feature or add the proper flags (but since we disabled WebBrowser, you should be fine).

5. **Run the App:** Once built, try launching the app. On macOS, you can do `open build/Debug/TOASTer.app` or navigate to the .app in Finder. If it doesn’t open due to unidentified developer, you can right-click > Open, or run the binary from Terminal. The app should at least show its main window (even if it’s blank or a placeholder UI).

   * If the app **does not launch** or crashes immediately, run it from Terminal to see logs. Common issues could be a null pointer in `MainComponent` (maybe expecting something from JMID that’s null). Use `DBG` logging (JUCE’s `DBG()` macro) or std::cout to trace where it crashes and fix that code.
   * If the app launches and shows UI, test basic interactions that don’t require the missing pieces. This validates that the removal of JAM/JMID hasn’t broken the core event loop or GUI.

6. **Implement/Stubs for Critical Missing Functionality:** Now that the app runs, you can begin restoring **minimal functionality** for PNBTR and JMID so that testing can proceed:

   * For example, if the app was supposed to show a list of MIDI devices (via JMID before), use JUCE’s `MidiInput::getAvailableDevices()` and populate the UI list. If implementing fully is complex, even stubbing one device name in the list is fine, just to ensure the UI element works.
   * If PNBTR was supposed to sync to a network clock or send network messages, perhaps implement a simple timer or use JUCE’s `juce::Time` class to simulate a clock. The goal is not full functionality (since Phase 4 might handle that properly) but rather to avoid leaving parts of the app totally broken. This way, testers can at least navigate the UI and trigger actions without the app crashing or hanging.

7. **Eliminate Any Remaining Dead Code:** As you go through the above, you might find portions of code that are no longer used (e.g., classes that were only relevant when JAM was around). Remove these from the project to reduce confusion. Also update the `CMakeLists.txt` to stop compiling any .cpp files that are obsolete.

8. **Thorough Testing:** Now test the **build process** on a clean system if possible:

   * Try the release build (`cmake --build build --config Release`) to ensure optimizations don’t introduce any issues and that the app still is produced in the bundle.
   * If you have access to an Intel Mac or want to test universal binary, try setting `CMAKE_OSX_ARCHITECTURES="arm64;x86_64"` and building to see if any architecture-specific issues pop up. This might not be necessary for your immediate needs, but it’s a check before distributing the app to others.

9. **Document and Next Steps:** After achieving a successful build and a running app, document what was changed (this helps anyone else on the project). You’ve effectively migrated the app to a pure-JUCE basis, which is a big architectural step. This sets the stage for Phase 4 development – which presumably will implement new networking (PNBTR) and MIDI (JMID) logic on top of this foundation. Make sure the team knows that any testing of PNBTR/JMID will be done with this GUI and that any missing pieces are currently stubbed.

Throughout this process, keep in mind that similar issues have been encountered by others when setting up JUCE apps via CMake. The solution almost always comes down to **proper CMake target configuration** for the app and ensuring all needed symbols are linked in. By following JUCE’s recommended patterns (as we’ve done above) and scrubbing out legacy dependencies, you’ll achieve a clean build.

## Conclusion

After applying these fixes, the TOASTer application should compile and link successfully under JUCE 8, producing a valid `.app` bundle with an executable. All outdated code tied to the removed JAM/JMID frameworks will be gone or replaced with modern equivalents, aligning the project with the intended Pre-Phase 4 architecture. The **CMakeLists** will be streamlined to include only JUCE and relevant in-project modules, and key source files like **Main.cpp** and **MainComponent.cpp** will be updated to use JUCE’s application model.

You will have a working GUI application with the minimal necessary dependencies, suitable for testing the core functionality of **PNBTR** and **JMID** going forward. From here, the team can incrementally flesh out the stubs with real implementations in Phase 4, confident that the foundation (build system and app structure) is solid.

**Sources:**

* JUCE CMake GUI App Example – demonstrates setting up a GUI app target and linking JUCE modules.
* TOASTer Build Issue Summary – provided context on the missing executable and removed frameworks.
* JUCE Documentation and Forum insights on common macOS build issues (empty app bundle, missing entry point).

