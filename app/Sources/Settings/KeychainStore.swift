import Foundation
import Security

/// API keys live in the login Keychain, never in files or UserDefaults.
enum KeychainStore {
    private static let service = "com.viraj.vaani.api-keys"

    static func get(_ account: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    @discardableResult
    static func set(_ value: String?, for account: String) -> Bool {
        let base: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        guard let value, !value.isEmpty else {
            let status = SecItemDelete(base as CFDictionary)
            return status == errSecSuccess || status == errSecItemNotFound
        }
        let data = Data(value.utf8)
        let update = SecItemUpdate(base as CFDictionary, [kSecValueData as String: data] as CFDictionary)
        if update == errSecSuccess { return true }
        var add = base
        add[kSecValueData as String] = data
        add[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        return SecItemAdd(add as CFDictionary, nil) == errSecSuccess
    }
}

/// The services Vaani can use, with the engine's key names.
enum APIService: String, CaseIterable, Identifiable {
    case sarvam, anthropic, openai, huggingface
    var id: String { rawValue }

    var title: String {
        switch self {
        case .sarvam: "Sarvam AI"
        case .anthropic: "Anthropic (Claude)"
        case .openai: "OpenAI"
        case .huggingface: "Hugging Face"
        }
    }

    var detail: String {
        switch self {
        case .sarvam: "Marathi/Hindi/Hinglish transcription, translation and voice cloning."
        case .anthropic: "Context-aware translation, Hinglish, dubbing-length scripts."
        case .openai: "Transcription and translation."
        case .huggingface: "Free token; needed to download gated models (IndicConformer, IndicTrans2, IndicF5)."
        }
    }

    var signupURL: URL {
        switch self {
        case .sarvam: URL(string: "https://dashboard.sarvam.ai")!
        case .anthropic: URL(string: "https://console.anthropic.com/settings/keys")!
        case .openai: URL(string: "https://platform.openai.com/api-keys")!
        case .huggingface: URL(string: "https://huggingface.co/settings/tokens")!
        }
    }

    var envName: String {
        switch self {
        case .sarvam: "VAANI_SARVAM_API_KEY"
        case .anthropic: "VAANI_ANTHROPIC_API_KEY"
        case .openai: "VAANI_OPENAI_API_KEY"
        case .huggingface: "VAANI_HF_TOKEN"
        }
    }
}
