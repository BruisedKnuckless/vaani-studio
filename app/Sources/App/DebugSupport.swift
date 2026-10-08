#if DEBUG
import AppKit
import Foundation

/// Development helpers driven by environment variables (Debug builds only):
///   VAANI_DEBUG_SCREEN=enhance|transcribe|dub|jobs|models|settings   start on that screen
///   VAANI_DEBUG_FILE=/path/to/audio                                   preselect a file
///   VAANI_DEBUG_SNAPSHOT=/path/to/dir                                 save window snapshots every 3 s
///   VAANI_DEBUG_AUTOSTART=1                                           press Start once the plan is ready
///   VAANI_DEBUG_SIZE=1100x1600                                        resize the window (for full-page snapshots)
///   VAANI_DEBUG_SETUP=1                                               press "Install the Engine" automatically
@MainActor
enum DebugSupport {
    static let env = ProcessInfo.processInfo.environment

    static var screen: Screen? { env["VAANI_DEBUG_SCREEN"].flatMap(Screen.init(rawValue:)) }
    static var file: URL? { env["VAANI_DEBUG_FILE"].map { URL(fileURLWithPath: $0) } }
    static var autostart: Bool { env["VAANI_DEBUG_AUTOSTART"] == "1" }
    static var autoSetup: Bool { env["VAANI_DEBUG_SETUP"] == "1" }

    static func startSnapshots() {
        guard let dir = env["VAANI_DEBUG_SNAPSHOT"] else { return }
        let folder = URL(fileURLWithPath: dir)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        var index = 0
        var resized = false
        Timer.scheduledTimer(withTimeInterval: 3, repeats: true) { _ in
            MainActor.assumeIsolated {
                guard let window = NSApp.windows.first(where: { $0.isVisible && $0.contentView != nil }) else { return }
                if !resized, let size = env["VAANI_DEBUG_SIZE"]?.split(separator: "x").compactMap({ Double($0) }),
                   size.count == 2 {
                    resized = true
                    window.setFrame(NSRect(x: 0, y: 0, width: size[0], height: size[1]), display: true)
                }
                guard
                      let view = window.contentView?.superview ?? window.contentView,
                      let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { return }
                view.cacheDisplay(in: view.bounds, to: rep)
                if let data = rep.representation(using: .png, properties: [:]) {
                    index += 1
                    try? data.write(to: folder.appending(path: String(format: "snap-%03d.png", index)))
                }
            }
        }
    }
}
#endif
