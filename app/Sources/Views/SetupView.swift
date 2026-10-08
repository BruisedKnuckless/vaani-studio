import SwiftUI

/// Shown on first launch of a shared build until the engine is installed.
struct SetupView: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        let installer = app.installer
        VStack(alignment: .leading, spacing: 18) {
            HStack(spacing: 14) {
                Image(nsImage: NSApp.applicationIconImage).resizable().frame(width: 64, height: 64)
                VStack(alignment: .leading, spacing: 4) {
                    Text("Welcome to Vaani Studio").font(.largeTitle.weight(.bold))
                    Text("One-time setup").foregroundStyle(.secondary)
                }
            }

            Text("Vaani runs its AI on your Mac. Before the first use it installs its engine: Python and the "
                 + "speech, translation and voice libraries. This is about a 2 GB download (around 4 GB on disk) "
                 + "and takes a few minutes. AI models are downloaded later, the first time you use each feature.")
                .fixedSize(horizontal: false, vertical: true)

            switch installer.phase {
            case .idle:
                Button {
                    Task { await app.runSetup() }
                } label: {
                    Label("Install the Engine", systemImage: "arrow.down.circle").frame(minWidth: 180)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            case .running(let step, let title):
                VStack(alignment: .leading, spacing: 8) {
                    ProgressView(value: Double(step), total: Double(EngineInstaller.steps.count))
                    HStack {
                        Text("Step \(step + 1) of \(EngineInstaller.steps.count): \(title)…").font(.headline)
                        Spacer()
                        Button("Cancel") { installer.cancel() }
                    }
                    Text(installer.log.last ?? "Starting…")
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            case .failed(let message):
                VStack(alignment: .leading, spacing: 10) {
                    Label("Setup didn't finish", systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange).font(.headline)
                    ScrollView {
                        Text(message).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(maxHeight: 180)
                    Button("Try Again") { Task { await app.runSetup() } }
                        .buttonStyle(.borderedProminent)
                }
            case .done:
                Label("Installed. Starting the engine…", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
            }

            Spacer()
            Text("Needs an Apple Silicon Mac and an internet connection for setup. Everything is stored in "
                 + "~/Library/Application Support/VaaniStudio.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(32)
        .frame(maxWidth: 640, maxHeight: .infinity, alignment: .topLeading)
        .frame(maxWidth: .infinity)
    }
}
