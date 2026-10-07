import Foundation

/// Launches the Python engine, waits for its `READY port=N` line, keeps its
/// output for the log view, and stops it when the app quits.
@MainActor
final class EngineProcess {
    private(set) var log: [String] = []
    private var process: Process?
    private var readyContinuation: CheckedContinuation<Int, Error>?
    private var readerTask: Task<Void, Never>?
    var onExit: ((Int32) -> Void)?

    var isRunning: Bool { process?.isRunning == true }

    /// The engine folder: an explicit override, the development checkout baked
    /// in at build time, or the installed copy in Application Support.
    static func engineDirectory() -> URL? {
        var candidates: [URL] = []
        if let custom = UserDefaults.standard.string(forKey: "engineDir"), !custom.isEmpty {
            candidates.append(URL(fileURLWithPath: custom))
        }
        if let baked = Bundle.main.object(forInfoDictionaryKey: "VaaniEngineDir") as? String,
           !baked.isEmpty, !baked.contains("$(") {
            candidates.append(URL(fileURLWithPath: baked).standardizedFileURL)
        }
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        candidates.append(support.appending(path: "VaaniStudio/engine"))
        return candidates.first { FileManager.default.fileExists(atPath: $0.appending(path: ".venv/bin/python").path) }
    }

    func launch(token: String, environment extra: [String: String]) async throws -> Int {
        guard let engineDir = Self.engineDirectory() else {
            throw EngineError(message: "The Vaani engine isn't installed. Run scripts/bootstrap.sh in the VaaniStudio folder.")
        }
        stop()
        log.removeAll()

        let process = Process()
        process.executableURL = engineDir.appending(path: ".venv/bin/python")
        process.arguments = ["-W", "ignore", "-m", "vaani_engine.server", "--port", "0"]
        process.currentDirectoryURL = engineDir

        var env = ProcessInfo.processInfo.environment
        let path = env["PATH"] ?? "/usr/bin:/bin"
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + path   // ffmpeg etc.; Finder apps get a bare PATH
        env["PYTHONUNBUFFERED"] = "1"
        env["VAANI_TOKEN"] = token
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        for (key, value) in extra { env[key] = value }
        process.environment = env

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        let (lines, continuation) = AsyncStream<String>.makeStream(bufferingPolicy: .bufferingNewest(2000))
        let splitter = LineSplitter()
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty {
                handle.readabilityHandler = nil
                continuation.finish()
                return
            }
            for line in splitter.append(data) { continuation.yield(line) }
        }
        process.terminationHandler = { [weak self] proc in
            let status = proc.terminationStatus
            Task { @MainActor in self?.handleExit(status) }
        }

        try process.run()
        self.process = process

        readerTask = Task { [weak self] in
            for await line in lines {
                self?.handle(line)
            }
        }

        let timeout = Task { [weak self] in
            try? await Task.sleep(for: .seconds(90))
            guard !Task.isCancelled else { return }
            self?.failReady(EngineError(message: "The engine didn't start within 90 seconds."))
        }
        defer { timeout.cancel() }
        return try await withCheckedThrowingContinuation { continuation in
            readyContinuation = continuation
        }
    }

    func stop() {
        guard let process, process.isRunning else { return }
        process.terminationHandler = nil
        process.terminate()
        self.process = nil
    }

    private func handle(_ line: String) {
        log.append(line)
        if log.count > 2000 { log.removeFirst(log.count - 2000) }
        if line.hasPrefix("READY port="), let port = Int(line.dropFirst("READY port=".count)) {
            readyContinuation?.resume(returning: port)
            readyContinuation = nil
        }
    }

    private func failReady(_ error: Error) {
        readyContinuation?.resume(throwing: error)
        readyContinuation = nil
    }

    private func handleExit(_ status: Int32) {
        process = nil
        let tail = log.suffix(8).joined(separator: "\n")
        failReady(EngineError(message: "The engine stopped (exit \(status)).\n\(tail)"))
        onExit?(status)
    }
}

/// Splits a byte stream into lines across chunk boundaries.
private final class LineSplitter: @unchecked Sendable {
    private var buffer = Data()
    private let lock = NSLock()

    func append(_ data: Data) -> [String] {
        lock.lock()
        defer { lock.unlock() }
        buffer.append(data)
        var lines: [String] = []
        while let newline = buffer.firstIndex(of: 0x0A) {
            let lineData = buffer[buffer.startIndex..<newline]
            buffer.removeSubrange(buffer.startIndex...newline)
            lines.append(String(decoding: lineData, as: UTF8.self).trimmingCharacters(in: .newlines))
        }
        return lines
    }
}
