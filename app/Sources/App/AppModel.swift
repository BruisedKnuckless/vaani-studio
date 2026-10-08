import AppKit
import Foundation
import Observation

enum Screen: String, CaseIterable, Identifiable, Hashable {
    case enhance, transcribe, dub, jobs, models, settings
    var id: String { rawValue }

    var title: String {
        switch self {
        case .enhance: "Enhance"
        case .transcribe: "Transcribe"
        case .dub: "Dub"
        case .jobs: "Jobs"
        case .models: "Models"
        case .settings: "Settings"
        }
    }

    var symbol: String {
        switch self {
        case .enhance: "waveform.badge.magnifyingglass"
        case .transcribe: "text.bubble"
        case .dub: "person.wave.2"
        case .jobs: "list.bullet.rectangle"
        case .models: "shippingbox"
        case .settings: "gearshape"
        }
    }
}

enum EngineState: Equatable {
    case needsSetup
    case starting
    case ready
    case failed(String)
}

@MainActor
@Observable
final class AppModel {
    var screen: Screen = .enhance
    var engineState: EngineState = .starting
    var health: Health?
    var jobs: [Job] = []
    var lastError: String?

    private(set) var client: EngineClient?
    let installer = EngineInstaller()
    private let engine = EngineProcess()
    private var eventsTask: Task<Void, Never>?
    private var started = false

    init() {
        NotificationCenter.default.addObserver(forName: NSApplication.willTerminateNotification,
                                               object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.engine.stop()
                try? FileManager.default.removeItem(at: Self.engineInfoURL)
            }
        }
        engine.onExit = { [weak self] status in
            guard let self, self.engineState == .ready else { return }
            self.engineState = .failed("The engine stopped unexpectedly (exit \(status)). Restart it from Settings.")
        }
    }

    var engineLog: [String] { engine.log }
    var offline: Bool { health?.offline ?? false }

    func hasKey(_ service: APIService) -> Bool { health?.keys[service.rawValue] ?? false }

    // MARK: Engine lifecycle

    func start() async {
        guard !started else { return }
        started = true
        #if DEBUG
        if let screen = DebugSupport.screen { self.screen = screen }
        DebugSupport.startSnapshots()
        #endif
        if EngineInstaller.needsInstall {
            engineState = .needsSetup
            #if DEBUG
            if DebugSupport.autoSetup { await runSetup() }
            #endif
            return
        }
        await restartEngine()
    }

    /// First-launch (or post-update) engine install for shared builds.
    func runSetup() async {
        if await installer.install() {
            await restartEngine()
        }
    }

    func restartEngine() async {
        engineState = .starting
        eventsTask?.cancel()
        let token = UUID().uuidString + UUID().uuidString
        var env: [String: String] = [:]
        for service in APIService.allCases {
            if let key = KeychainStore.get(service.rawValue) { env[service.envName] = key }
        }
        if UserDefaults.standard.bool(forKey: "offline") { env["VAANI_OFFLINE"] = "1" }
        if let output = UserDefaults.standard.string(forKey: "outputDir"), !output.isEmpty {
            env["VAANI_OUTPUT"] = output
        }
        do {
            let port = try await engine.launch(token: token, environment: env)
            let client = EngineClient(baseURL: URL(string: "http://127.0.0.1:\(port)")!, token: token)
            self.client = client
            Self.publishEngineAddress(port: port, token: token)
            health = try await waitForHealth(client)
            engineState = .ready
            await refreshJobs()
            listenForEvents()
        } catch {
            engineState = .failed(error.localizedDescription)
        }
    }

    /// Where the running engine can be reached, for `vaani … --via-app` (readable only by this user).
    static let engineInfoURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appending(path: "VaaniStudio/engine.json")

    private static func publishEngineAddress(port: Int, token: String) {
        let data = try? JSONSerialization.data(withJSONObject: ["port": port, "token": token])
        try? FileManager.default.createDirectory(at: engineInfoURL.deletingLastPathComponent(),
                                                 withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: engineInfoURL.path, contents: data,
                                       attributes: [.posixPermissions: 0o600])
    }

    private func waitForHealth(_ client: EngineClient) async throws -> Health {
        var lastError: Error?
        for _ in 0..<40 {
            do { return try await client.health() } catch { lastError = error }
            try await Task.sleep(for: .milliseconds(250))
        }
        throw lastError ?? EngineError(message: "Engine did not respond")
    }

    private func listenForEvents() {
        eventsTask?.cancel()
        eventsTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let client = self?.client else { return }
                do {
                    for try await job in client.jobEvents() {
                        self?.upsert(job)
                    }
                } catch {}
                try? await Task.sleep(for: .seconds(1))
                await self?.refreshJobs()
            }
        }
    }

    // MARK: Jobs

    func refreshJobs() async {
        guard let client else { return }
        if let list = try? await client.jobs() {
            jobs = list
            updateBusyActivity()
        }
    }

    func upsert(_ job: Job) {
        if let index = jobs.firstIndex(where: { $0.id == job.id }) {
            jobs[index] = job
        } else {
            jobs.insert(job, at: 0)
        }
        updateBusyActivity()
    }

    /// While any job runs, stop macOS from App-Napping the app (which also throttles the engine's
    /// CPU and disk) when its window is in the background.
    private var busyActivity: NSObjectProtocol?

    private func updateBusyActivity() {
        let busy = jobs.contains(where: \.isActive)
        if busy, busyActivity == nil {
            busyActivity = ProcessInfo.processInfo.beginActivity(options: [.userInitiated, .idleSystemSleepDisabled],
                                                                 reason: "Processing audio")
        } else if !busy, let activity = busyActivity {
            ProcessInfo.processInfo.endActivity(activity)
            busyActivity = nil
        }
    }

    func job(_ id: String?) -> Job? {
        guard let id else { return nil }
        return jobs.first { $0.id == id }
    }

    func submit(_ plan: PlanRequest, file: URL) async -> Job? {
        guard let client else { return nil }
        do {
            let job = try await client.submit(JobRequest(plan: plan, inputPath: file.path))
            upsert(job)
            return job
        } catch {
            lastError = error.localizedDescription
            return nil
        }
    }

    func cancel(_ job: Job) async {
        guard let client else { return }
        if let updated = try? await client.cancel(job.id) { upsert(updated) }
    }

    func delete(_ job: Job) async {
        guard let client else { return }
        try? await client.deleteJob(job.id)
        jobs.removeAll { $0.id == job.id }
    }

    /// Re-runs a failed job with the failing step moved to this Mac.
    func retryLocally(_ job: Job) async -> Job? {
        guard let client, let error = job.error, let stage = error.stage,
              let local = error.localAlternative,
              let type = JobType(rawValue: job.type) else { return nil }
        var request = PlanRequest(jobType: type)
        request.srcLang = job.params["src_lang"]?.stringValue ?? "auto"
        request.tgtLang = job.params["tgt_lang"]?.stringValue
        request.mode = ProcessingMode(rawValue: job.params["mode"]?.stringValue ?? "auto") ?? .auto
        request.options = job.params["options"]?.objectValue ?? [:]
        var overrides = (job.params["overrides"]?.objectValue ?? [:]).compactMapValues(\.stringValue)
        overrides[stage] = local
        request.overrides = overrides
        do {
            let new = try await client.submit(JobRequest(plan: request, inputPath: job.inputPath))
            upsert(new)
            return new
        } catch {
            lastError = error.localizedDescription
            return nil
        }
    }

    // MARK: Settings

    func saveKey(_ service: APIService, value: String?) async {
        let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines)
        KeychainStore.set(trimmed, for: service.rawValue)
        guard let client else { return }
        let body = SettingsRequest(keys: [service.rawValue: (trimmed?.isEmpty ?? true) ? nil : trimmed])
        if let updated = try? await client.updateSettings(body) { health = updated }
    }

    func setOffline(_ offline: Bool) async {
        UserDefaults.standard.set(offline, forKey: "offline")
        guard let client else { return }
        if let updated = try? await client.updateSettings(SettingsRequest(offline: offline)) { health = updated }
    }

    func setOutputDir(_ url: URL) async {
        UserDefaults.standard.set(url.path, forKey: "outputDir")
        guard let client else { return }
        if let updated = try? await client.updateSettings(SettingsRequest(outputDir: url.path)) { health = updated }
    }

    func reveal(_ url: URL) {
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }
}
