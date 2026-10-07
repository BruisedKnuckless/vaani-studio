import Foundation

// Mirrors of the engine's JSON. Keys arrive in snake_case and are decoded with
// `.convertFromSnakeCase`; dictionary keys (e.g. step names) stay as sent.

enum JobType: String, Codable, Sendable, CaseIterable {
    case enhance, transcribe, dub
}

enum ProcessingMode: String, Codable, Sendable, CaseIterable, Identifiable {
    case local, cloud, auto
    var id: String { rawValue }

    var title: String {
        switch self {
        case .local: "On this Mac"
        case .cloud: "Cloud API"
        case .auto: "Auto"
        }
    }

    var symbol: String {
        switch self {
        case .local: "desktopcomputer"
        case .cloud: "cloud"
        case .auto: "wand.and.stars"
        }
    }

    var explanation: String {
        switch self {
        case .local: "Private and free. Everything runs on your Mac."
        case .cloud: "Uses your API keys. Usually best for Marathi; audio leaves your Mac."
        case .auto: "Picks the best option per step and falls back if one fails."
        }
    }
}

enum Language: String, CaseIterable, Identifiable, Sendable {
    case auto, en, hi, mr, hinglish
    var id: String { rawValue }

    var title: String {
        switch self {
        case .auto: "Auto-detect"
        case .en: "English"
        case .hi: "Hindi"
        case .mr: "Marathi"
        case .hinglish: "Hinglish"
        }
    }
}

struct Health: Codable, Sendable {
    var status: String
    var version: String
    var offline: Bool
    var keys: [String: Bool]
    var outputDir: String
}

struct Choice: Codable, Hashable, Sendable, Identifiable {
    var id: String
    var name: String
    var kind: String
    var status: String
}

struct StagePlan: Codable, Hashable, Sendable, Identifiable {
    var stage: String
    var label: String
    var providerId: String
    var providerName: String
    var kind: String
    var status: String
    var note: String
    var fallbackId: String?
    var localAlternative: String?
    var alternatives: [Choice]
    var blocking: Bool

    var id: String { stage }
    var isCloud: Bool { kind == "cloud" }
}

struct Plan: Codable, Hashable, Sendable {
    var jobType: String
    var mode: String
    var srcLang: String
    var tgtLang: String?
    var offline: Bool
    var uploads: Bool
    var blocking: Bool
    var steps: [StagePlan]
    var summary: String
    var cloudAvailable: Bool?
}

struct JobError: Codable, Hashable, Sendable {
    var message: String
    var stage: String?
    var provider: String?
    var kind: String?
    var localAlternative: String?
}

struct Job: Codable, Identifiable, Hashable, Sendable {
    var id: String
    var type: String
    var inputPath: String
    var params: JSONValue
    var plan: Plan
    var outDir: String
    var status: String
    var progress: Double
    var message: String
    var providersUsed: [String: String]
    var outputs: [String: JSONValue]
    var error: JobError?
    var log: [String]
    var createdAt: Double
    var updatedAt: Double

    var isActive: Bool { status == "queued" || status == "running" }
    var inputName: String { URL(fileURLWithPath: inputPath).lastPathComponent }
    var created: Date { Date(timeIntervalSince1970: createdAt) }

    func outputPath(_ key: String) -> URL? {
        guard let path = outputs[key]?.stringValue else { return nil }
        return URL(fileURLWithPath: path)
    }

    var metrics: [String: Double] {
        guard case .object(let dict)? = outputs["metrics"] else { return [:] }
        return dict.compactMapValues(\.doubleValue)
    }
}

struct ModelInfo: Codable, Identifiable, Hashable, Sendable {
    var id: String
    var name: String
    var purpose: String
    var repo: String?
    var sizeMb: Int
    var license: String
    var gated: Bool
    var runtime: String
    var commercialOk: Bool
    var state: String
    var progress: Double?
    var diskMb: Int
    var error: String?
    var termsUrl: String?
}

struct Peaks: Codable, Sendable {
    var duration: Double
    var peaks: [Double]
}

struct AudioInfo: Codable, Sendable {
    var duration: Double
    var sampleRate: Int
    var channels: Int
    var codec: String
}

// MARK: - Requests

struct PlanRequest: Codable, Hashable, Sendable {
    var jobType: JobType
    var srcLang: String = "auto"
    var tgtLang: String?
    var mode: ProcessingMode = .auto
    var overrides: [String: String] = [:]
    var options: [String: JSONValue] = [:]
}

struct JobRequest: Codable, Sendable {
    var jobType: JobType
    var inputPath: String
    var srcLang: String
    var tgtLang: String?
    var mode: ProcessingMode
    var overrides: [String: String]
    var options: [String: JSONValue]

    init(plan: PlanRequest, inputPath: String) {
        jobType = plan.jobType
        self.inputPath = inputPath
        srcLang = plan.srcLang
        tgtLang = plan.tgtLang
        mode = plan.mode
        overrides = plan.overrides
        options = plan.options
    }
}

struct SettingsRequest: Codable, Sendable {
    var keys: [String: String?]?
    var offline: Bool?
    var outputDir: String?
}

// MARK: - Loose JSON

enum JSONValue: Codable, Hashable, Sendable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case object([String: JSONValue])
    case array([JSONValue])
    case null

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let b = try? c.decode(Bool.self) { self = .bool(b) }
        else if let n = try? c.decode(Double.self) { self = .number(n) }
        else if let s = try? c.decode(String.self) { self = .string(s) }
        else if let a = try? c.decode([JSONValue].self) { self = .array(a) }
        else { self = .object(try c.decode([String: JSONValue].self)) }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .string(let s): try c.encode(s)
        case .number(let n): try c.encode(n)
        case .bool(let b): try c.encode(b)
        case .object(let o): try c.encode(o)
        case .array(let a): try c.encode(a)
        case .null: try c.encodeNil()
        }
    }

    var stringValue: String? { if case .string(let s) = self { s } else { nil } }
    var doubleValue: Double? { if case .number(let n) = self { n } else { nil } }
    var boolValue: Bool? { if case .bool(let b) = self { b } else { nil } }
    var arrayValue: [JSONValue]? { if case .array(let a) = self { a } else { nil } }
    var objectValue: [String: JSONValue]? { if case .object(let o) = self { o } else { nil } }

    subscript(key: String) -> JSONValue? { objectValue?[key] }
}

extension JSONValue: ExpressibleByStringLiteral, ExpressibleByFloatLiteral, ExpressibleByBooleanLiteral,
    ExpressibleByIntegerLiteral {
    init(stringLiteral value: String) { self = .string(value) }
    init(floatLiteral value: Double) { self = .number(value) }
    init(booleanLiteral value: Bool) { self = .bool(value) }
    init(integerLiteral value: Int) { self = .number(Double(value)) }
}
