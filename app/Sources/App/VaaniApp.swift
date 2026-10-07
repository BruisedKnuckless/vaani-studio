import SwiftUI

@main
struct VaaniApp: App {
    @State private var app = AppModel()

    var body: some Scene {
        WindowGroup("Vaani Studio") {
            ContentView()
                .environment(app)
                .frame(minWidth: 980, minHeight: 660)
                .task { await app.start() }
        }
        .windowToolbarStyle(.unified)
        .commands {
            CommandGroup(replacing: .newItem) {}
            CommandMenu("Engine") {
                Button("Restart Engine") { Task { await app.restartEngine() } }
                    .keyboardShortcut("r", modifiers: [.command, .shift])
            }
        }
    }
}
