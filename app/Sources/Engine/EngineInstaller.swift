import Foundation
import Observation

/// First-launch setup for shared builds: the app bundle carries the engine's source and
/// lockfiles plus the `uv` installer; this installs Python 3.11 and the engine's packages
/// into ~/Library/Application Support/VaaniStudio. Re-runs (quickly) after an app update.
@MainActor
@Observable
final class EngineInstaller {
    enum Phase: Equatable {
        case idle
        case running(step: Int, title: String)
        case failed(String)
        case done
    }

    private(set) var phase: Phase = .idle
    private(set) var log: [String] = []
    private var process: Process?
    private var cancelled = false

    static let steps = ["Copying the engine", "Installing Python 3.11", "Installing the AI engine",
                        "Installing the voice tools", "Finishing up"]

    static var supportDir: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appending(path: "VaaniStudio")
    }
    static var engineDir: URL { supportDir.appending(path: "engine") }
    private static var marker: URL { engineDir.appending(path: ".installed-version") }
    private static var bundledEngine: URL? {
        guard let url = Bundle.main.resourceURL?.appending(path: "engine"),
              FileManager.default.fileExists(atPath: url.appending(path: "pyproject.toml").path) else { return nil }
        return url
    }
    private static var bundledUV: URL? { Bundle.main.resourceURL?.appending(path: "bin/uv") }

    /// The version of the engine shipped inside this app (nil for development builds).
    static var bundledVersion: String? {
        guard let engine = bundledEngine else { return nil }
        return try? String(contentsOf: engine.appending(path: "VERSION"), encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// True when this is a shared build whose engine hasn't been installed (or was installed by an older version).
    static var needsInstall: Bool {
        guard let bundled = bundledVersion else { return false }
        let installed = try? String(contentsOf: marker, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
        let python = engineDir.appending(path: ".venv/bin/python").path
        return installed != bundled || !FileManager.default.fileExists(atPath: python)
    }

    var isRunning: Bool {
        if case .running = phase { return true }
        return false
    }

    func install() async -> Bool {
        guard let source = Self.bundledEngine, let uv = Self.bundledUV, let version = Self.bundledVersion else {
            phase = .failed("This copy of Vaani doesn't include the engine installer.")
            return false
        }
        cancelled = false
        log.removeAll()
        let engine = Self.engineDir
        let fm = FileManager.default
        try? fm.createDirectory(at: engine, withIntermediateDirectories: true)

        let commands: [(Int, URL, [String], URL)] = [
            (0, URL(fileURLWithPath: "/usr/bin/rsync"),
             ["-a", "--delete", "--exclude", ".venv", "--exclude", "__pycache__", "--exclude", ".installed-version",
              source.path + "/", engine.path + "/"], Self.supportDir),
            (1, uv, ["python", "install", "3.11"], engine),
            (2, uv, ["sync", "--frozen", "--no-dev", "--extra", "ml", "--python", "3.11"], engine),
            (3, uv, ["sync", "--frozen", "--no-dev", "--python", "3.11"], engine.appending(path: "runtimes/mlxvoice")),
            (3, uv, ["sync", "--frozen", "--no-dev", "--python", "3.11"], engine.appending(path: "runtimes/voice")),
        ]
        for (step, tool, arguments, directory) in commands {
            phase = .running(step: step, title: Self.steps[step])
            let status = await run(tool, arguments, in: directory)
            if cancelled {
                phase = .failed("Setup was cancelled. You can start it again any time.")
                return false
            }
            if status != 0 {
                let tail = log.suffix(12).joined(separator: "\n")
                phase = .failed("\(Self.steps[step]) failed. Check your internet connection and try again.\n\n\(tail)")
                return false
            }
        }
        phase = .running(step: 4, title: Self.steps[4])
        try? version.write(to: Self.marker, atomically: true, encoding: .utf8)
        phase = .done
        return true
    }

    func cancel() {
        cancelled = true
        process?.terminate()
    }

    private func run(_ tool: URL, _ arguments: [String], in directory: URL) async -> Int32 {
        let process = Process()
        process.executableURL = tool
        process.arguments = arguments
        process.currentDirectoryURL = directory
        var env = ProcessInfo.processInfo.environment
        env["UV_PYTHON_INSTALL_DIR"] = Self.supportDir.appending(path: "python").path
        env["UV_CACHE_DIR"] = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appending(path: "VaaniStudio/uv").path
        env["UV_PYTHON_PREFERENCE"] = "only-managed"   // never pick up a Homebrew or system Python
        env["UV_NO_PROGRESS"] = "1"
        env["NO_COLOR"] = "1"
        process.environment = env
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        let (lines, continuation) = AsyncStream<String>.makeStream(bufferingPolicy: .bufferingNewest(500))
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty {
                handle.readabilityHandler = nil
                continuation.finish()
                return
            }
            for line in String(decoding: data, as: UTF8.self).split(whereSeparator: \.isNewline) {
                continuation.yield(String(line))
            }
        }
        do {
            try process.run()
        } catch {
            log.append("Couldn't start \(tool.lastPathComponent): \(error.localizedDescription)")
            return -1
        }
        self.process = process
        for await line in lines {
            log.append(line)
            if log.count > 400 { log.removeFirst(log.count - 400) }
        }
        process.waitUntilExit()
        self.process = nil
        return process.terminationStatus
    }
}
