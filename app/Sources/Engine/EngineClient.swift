import Foundation

struct EngineError: LocalizedError, Sendable {
    var message: String
    var status: Int = 0
    var errorDescription: String? { message }
}

/// Talks to the local Python engine over HTTP on 127.0.0.1.
struct EngineClient: Sendable {
    let baseURL: URL
    let token: String

    private static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }()

    private static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.keyEncodingStrategy = .convertToSnakeCase
        return e
    }()

    // MARK: Endpoints

    func health() async throws -> Health { try await get("health") }
    func updateSettings(_ body: SettingsRequest) async throws -> Health { try await post("settings", body) }
    func preview(_ request: PlanRequest) async throws -> Plan { try await post("plan/preview", request) }
    func submit(_ request: JobRequest) async throws -> Job { try await post("jobs", request) }
    func jobs() async throws -> [Job] { try await get("jobs") }
    func job(_ id: String) async throws -> Job { try await get("jobs/\(id)") }
    func cancel(_ id: String) async throws -> Job { try await post("jobs/\(id)/cancel", [String: String]()) }
    func deleteJob(_ id: String) async throws { let _: [String: String] = try await send("jobs/\(id)", method: "DELETE") }
    func models() async throws -> [ModelInfo] { try await get("models") }
    func download(model id: String) async throws -> ModelInfo { try await post("models/\(id)/download", [String: String]()) }
    func delete(model id: String) async throws -> ModelInfo { try await send("models/\(id)", method: "DELETE") }

    func peaks(for path: String, buckets: Int = 1200) async throws -> Peaks {
        try await post("audio/peaks", ["path": JSONValue.string(path), "buckets": .number(Double(buckets))])
    }

    func audioInfo(for path: String) async throws -> AudioInfo {
        try await post("audio/info", ["path": path])
    }

    /// Server-sent job updates. Ends when the connection drops.
    func jobEvents() -> AsyncThrowingStream<Job, Error> {
        let request = makeRequest("events", method: "GET", timeout: 3600)
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let (bytes, response) = try await URLSession.shared.bytes(for: request)
                    try Self.check(response, data: nil)
                    var event = ""
                    for try await line in bytes.lines {
                        if line.hasPrefix("event:") {
                            event = line.dropFirst(6).trimmingCharacters(in: .whitespaces)
                        } else if line.hasPrefix("data:"), event == "job" {
                            let payload = Data(line.dropFirst(5).utf8)
                            if let job = try? Self.decoder.decode(Job.self, from: payload) {
                                continuation.yield(job)
                            }
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: Plumbing

    private func makeRequest(_ path: String, method: String, timeout: TimeInterval = 120) -> URLRequest {
        var request = URLRequest(url: baseURL.appending(path: path))
        request.httpMethod = method
        request.timeoutInterval = timeout
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        return request
    }

    private func get<T: Decodable>(_ path: String) async throws -> T {
        try await send(path, method: "GET")
    }

    private func post<T: Decodable, B: Encodable>(_ path: String, _ body: B) async throws -> T {
        var request = makeRequest(path, method: "POST")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try Self.encoder.encode(body)
        return try await perform(request)
    }

    private func send<T: Decodable>(_ path: String, method: String) async throws -> T {
        try await perform(makeRequest(path, method: method))
    }

    private func perform<T: Decodable>(_ request: URLRequest) async throws -> T {
        let (data, response) = try await URLSession.shared.data(for: request)
        try Self.check(response, data: data)
        do {
            return try Self.decoder.decode(T.self, from: data)
        } catch {
            throw EngineError(message: "Unexpected reply from engine: \(error)")
        }
    }

    private static func check(_ response: URLResponse, data: Data?) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200..<300).contains(http.statusCode) else {
            var message = "Engine error \(http.statusCode)"
            if let data, let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let detail = body["detail"] {
                message = detail as? String ?? "\(detail)"
            }
            throw EngineError(message: message, status: http.statusCode)
        }
    }
}
