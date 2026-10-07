import SwiftUI

struct ContentView: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        @Bindable var app = app
        NavigationSplitView {
            List(selection: $app.screen) {
                Section("Create") {
                    ForEach([Screen.enhance, .transcribe, .dub]) { row($0) }
                }
                Section("Library") {
                    row(.jobs)
                }
                Section("Setup") {
                    ForEach([Screen.models, .settings]) { row($0) }
                }
            }
            .navigationSplitViewColumnWidth(min: 190, ideal: 210)
            .safeAreaInset(edge: .bottom) { EngineStatusBadge().padding(10) }
        } detail: {
            Group {
                switch app.engineState {
                case .ready:
                    detail
                case .starting:
                    ProgressView("Starting the Vaani engine…")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                case .failed(let message):
                    EngineFailedView(message: message)
                }
            }
        }
        .alert("Something went wrong", isPresented: Binding(
            get: { app.lastError != nil }, set: { if !$0 { app.lastError = nil } })
        ) {
            Button("OK") { app.lastError = nil }
        } message: {
            Text(app.lastError ?? "")
        }
    }

    @ViewBuilder private var detail: some View {
        switch app.screen {
        case .enhance: EnhanceView()
        case .transcribe: TranscribeView()
        case .dub: DubView()
        case .jobs: JobsView()
        case .models: ModelsView()
        case .settings: SettingsView()
        }
    }

    private func row(_ screen: Screen) -> some View {
        Label(screen.title, systemImage: screen.symbol)
            .badge(screen == .jobs ? app.jobs.filter(\.isActive).count : 0)
            .tag(screen)
    }
}

struct EngineStatusBadge: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        HStack(spacing: 6) {
            Circle().fill(color).frame(width: 8, height: 8)
            Text(text).font(.caption).foregroundStyle(.secondary)
            Spacer()
        }
    }

    private var color: Color {
        switch app.engineState {
        case .ready: app.offline ? .orange : .green
        case .starting: .yellow
        case .failed: .red
        }
    }

    private var text: String {
        switch app.engineState {
        case .ready: app.offline ? "Engine ready · Offline" : "Engine ready"
        case .starting: "Starting engine…"
        case .failed: "Engine stopped"
        }
    }
}

struct EngineFailedView: View {
    @Environment(AppModel.self) private var app
    let message: String

    var body: some View {
        ContentUnavailableView {
            Label("The engine isn't running", systemImage: "exclamationmark.triangle")
        } description: {
            Text(message).textSelection(.enabled)
        } actions: {
            Button("Try Again") { Task { await app.restartEngine() } }
                .buttonStyle(.borderedProminent)
        }
    }
}
