import SwiftUI
import UniformTypeIdentifiers

/// Drop or pick an audio (or video) file. Shows its waveform once chosen.
struct FileDropZone: View {
    @Environment(AppModel.self) private var app
    @Binding var file: URL?
    @State private var targeted = false
    @State private var importing = false
    @State private var info: AudioInfo?

    var body: some View {
        Group {
            if let file {
                chosen(file)
            } else {
                empty
            }
        }
        .dropDestination(for: URL.self) { urls, _ in
            guard let url = urls.first else { return false }
            file = url
            return true
        } isTargeted: { targeted = $0 }
        .fileImporter(isPresented: $importing, allowedContentTypes: [.audio, .movie]) { result in
            if case .success(let url) = result { file = url }
        }
        .task(id: file) {
            info = nil
            guard let file, let client = app.client else { return }
            info = try? await client.audioInfo(for: file.path)
        }
    }

    private var empty: some View {
        VStack(spacing: 10) {
            Image(systemName: "waveform.circle")
                .font(.system(size: 40, weight: .light))
                .foregroundStyle(.secondary)
            Text("Drop an audio file here").font(.headline)
            Text("WAV, MP3, M4A, FLAC — or a video file").font(.callout).foregroundStyle(.secondary)
            Button("Choose File…") { importing = true }
        }
        .frame(maxWidth: .infinity, minHeight: 160)
        .background(RoundedRectangle(cornerRadius: 12)
            .strokeBorder(style: StrokeStyle(lineWidth: 1.5, dash: [6, 5]))
            .foregroundStyle(targeted ? Color.accentColor : Color.secondary.opacity(0.5)))
        .background(RoundedRectangle(cornerRadius: 12).fill(targeted ? Color.accentColor.opacity(0.08) : .clear))
        .contentShape(Rectangle())
        .onTapGesture { importing = true }
    }

    private func chosen(_ url: URL) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "music.note").foregroundStyle(.secondary)
                Text(url.lastPathComponent).font(.headline).lineLimit(1).truncationMode(.middle)
                if let info {
                    Text(Self.describe(info)).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Button("Change…") { importing = true }
                Button { file = nil } label: { Image(systemName: "xmark.circle.fill") }
                    .buttonStyle(.borderless)
                    .help("Remove file")
            }
            WaveformView(path: url.path)
                .frame(height: 64)
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 12).fill(.quaternary.opacity(0.5)))
    }

    static func describe(_ info: AudioInfo) -> String {
        let channels = info.channels == 1 ? "mono" : info.channels == 2 ? "stereo" : "\(info.channels) ch"
        return "\(formatDuration(info.duration)) · \(info.sampleRate / 1000) kHz \(channels)"
    }
}

func formatDuration(_ seconds: Double) -> String {
    guard seconds.isFinite else { return "–" }
    let total = Int(seconds.rounded())
    return total >= 3600
        ? String(format: "%d:%02d:%02d", total / 3600, total / 60 % 60, total % 60)
        : String(format: "%d:%02d", total / 60, total % 60)
}
