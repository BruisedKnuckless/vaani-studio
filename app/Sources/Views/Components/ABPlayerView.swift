import AVFoundation
import SwiftUI

/// Plays two versions of the same audio and switches between them at the same position.
@MainActor
@Observable
final class ABPlayer {
    private(set) var current: Int = 1
    private(set) var isPlaying = false
    private(set) var position: Double = 0
    private(set) var duration: Double = 0
    private(set) var loadError: String?
    private var players: [AVAudioPlayer] = []
    private var timer: Timer?

    func load(_ urls: [URL]) {
        stop()
        loadError = nil
        do {
            players = try urls.map { url in
                let player = try AVAudioPlayer(contentsOf: url)
                player.prepareToPlay()
                return player
            }
            duration = players.map(\.duration).max() ?? 0
            current = min(current, max(players.count - 1, 0))
        } catch {
            players = []
            loadError = "Can't play this file: \(error.localizedDescription)"
        }
    }

    func toggle() {
        guard players.indices.contains(current) else { return }
        if isPlaying { pause() } else { play() }
    }

    func play() {
        guard players.indices.contains(current) else { return }
        players[current].currentTime = min(position, players[current].duration)
        players[current].play()
        isPlaying = true
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.tick() }
        }
    }

    func pause() {
        players.forEach { $0.pause() }
        isPlaying = false
        timer?.invalidate()
    }

    func stop() {
        pause()
        position = 0
    }

    func select(_ index: Int) {
        guard players.indices.contains(index), index != current else { return }
        let wasPlaying = isPlaying
        if players.indices.contains(current) { position = players[current].currentTime }
        pause()
        current = index
        if wasPlaying { play() }
    }

    func seek(fraction: Double) {
        seek(to: fraction * duration)
    }

    func seek(to seconds: Double) {
        position = max(0, min(seconds, duration))
        if isPlaying { players[current].currentTime = position }
    }

    /// Play from `start`, used by transcript rows.
    func play(from start: Double) {
        seek(to: start)
        if !isPlaying { play() }
    }

    private func tick() {
        guard players.indices.contains(current) else { return }
        let player = players[current]
        position = player.currentTime
        if !player.isPlaying {
            isPlaying = false
            timer?.invalidate()
            if position >= player.duration - 0.05 || position == 0 { position = 0 }
        }
    }
}

struct ABPlayerView: View {
    let items: [(label: String, url: URL)]
    /// Owned by the screen so it can seek (e.g. jump to a transcript line).
    var player: ABPlayer

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 12) {
                Button { player.toggle() } label: {
                    Image(systemName: player.isPlaying ? "pause.fill" : "play.fill")
                        .frame(width: 22, height: 22)
                }
                .keyboardShortcut(.space, modifiers: [])
                .buttonStyle(.borderedProminent)
                .controlSize(.large)

                if items.count > 1 {
                    Picker("Listen to", selection: Binding(get: { player.current }, set: { player.select($0) })) {
                        ForEach(items.indices, id: \.self) { i in Text(items[i].label).tag(i) }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .frame(maxWidth: 320)
                    .help("Switch versions without losing your place")
                }

                Text("\(formatDuration(player.position)) / \(formatDuration(player.duration))")
                    .monospacedDigit().font(.callout).foregroundStyle(.secondary)
                Spacer()
            }
            if items.indices.contains(player.current) {
                WaveformView(path: items[player.current].url.path,
                             progress: player.duration > 0 ? player.position / player.duration : 0,
                             onSeek: { player.seek(fraction: $0) })
                    .frame(height: 70)
            }
            if let error = player.loadError {
                Text(error).font(.caption).foregroundStyle(.red)
            }
        }
        .onAppear { player.load(items.map(\.url)) }
        .onChange(of: items.map(\.url)) { _, urls in player.load(urls) }
        .onDisappear { player.stop() }
    }
}
