import SwiftUI

struct SettingsView: View {
    @Environment(AppModel.self) private var app
    @AppStorage("offline") private var offline = false
    @AppStorage("mode.enhance") private var enhanceMode: ProcessingMode = .local
    @AppStorage("mode.transcribe") private var transcribeMode: ProcessingMode = .auto
    @AppStorage("mode.dub") private var dubMode: ProcessingMode = .auto
    @State private var showLog = false

    var body: some View {
        Form {
            Section {
                ForEach(APIService.allCases) { service in
                    KeyRow(service: service)
                }
            } header: {
                Text("API keys")
            } footer: {
                Text("Keys are stored in your Mac's Keychain and only sent to the service they belong to.")
            }

            Section("Privacy") {
                Toggle(isOn: $offline) {
                    VStack(alignment: .leading) {
                        Text("Offline mode")
                        Text("Never use cloud APIs. Every step runs on this Mac.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                .onChange(of: offline) { _, value in Task { await app.setOffline(value) } }
            }

            Section {
                Picker("Enhance", selection: $enhanceMode) { modeOptions }
                Picker("Transcribe", selection: $transcribeMode) { modeOptions }
                Picker("Dub", selection: $dubMode) { modeOptions }
            } header: {
                Text("Starting choice for each screen")
            } footer: {
                Text("You still choose for every file you add; each screen remembers your last choice.")
            }

            Section("Output") {
                LabeledContent("Save results in") {
                    HStack {
                        Text(app.health?.outputDir ?? "–").lineLimit(1).truncationMode(.middle)
                            .foregroundStyle(.secondary)
                        Button("Change…") { chooseOutput() }
                        if let dir = app.health?.outputDir {
                            Button("Show") { app.reveal(URL(fileURLWithPath: dir)) }
                        }
                    }
                }
            }

            Section("Engine") {
                LabeledContent("Status") { EngineStatusBadge().frame(width: 200) }
                LabeledContent("Version") { Text(app.health?.version ?? "–") }
                HStack {
                    Button("Restart Engine") { Task { await app.restartEngine() } }
                    Button(showLog ? "Hide Log" : "Show Log") { showLog.toggle() }
                }
                if showLog {
                    ScrollView {
                        Text(app.engineLog.suffix(300).joined(separator: "\n"))
                            .font(.system(.caption, design: .monospaced))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(height: 220)
                }
            }
        }
        .formStyle(.grouped)
        .navigationTitle("Settings")
    }

    @ViewBuilder private var modeOptions: some View {
        ForEach(ProcessingMode.allCases) { Text($0.title).tag($0) }
    }

    private func chooseOutput() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.prompt = "Choose"
        if panel.runModal() == .OK, let url = panel.url {
            Task { await app.setOutputDir(url) }
        }
    }
}

private struct KeyRow: View {
    @Environment(AppModel.self) private var app
    let service: APIService
    @State private var editing = false
    @State private var value = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(service.title)
                    Text(service.detail).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                if app.hasKey(service) {
                    Label("Saved", systemImage: "checkmark.seal.fill").foregroundStyle(.green).font(.callout)
                }
                Button(app.hasKey(service) ? "Change" : "Add") { editing.toggle() }
                if app.hasKey(service) {
                    Button("Remove", role: .destructive) { Task { await app.saveKey(service, value: nil) } }
                }
            }
            if editing {
                HStack {
                    SecureField("Paste your \(service.title) key", text: $value)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit(save)
                    Button("Save", action: save).disabled(value.trimmingCharacters(in: .whitespaces).isEmpty)
                    Link("Get a key", destination: service.signupURL).font(.callout)
                }
            }
        }
        .padding(.vertical, 2)
    }

    private func save() {
        let key = value
        value = ""
        editing = false
        Task { await app.saveKey(service, value: key) }
    }
}
