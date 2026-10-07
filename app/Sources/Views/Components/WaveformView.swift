import SwiftUI

/// Draws the max-amplitude envelope the engine computes for a file.
struct WaveformView: View {
    @Environment(AppModel.self) private var app
    let path: String
    var progress: Double? = nil
    var tint: Color = .accentColor
    var onSeek: ((Double) -> Void)? = nil
    @State private var peaks: [Double] = []

    var body: some View {
        GeometryReader { geo in
            Canvas { context, size in
                guard !peaks.isEmpty else { return }
                let mid = size.height / 2
                let step = size.width / CGFloat(peaks.count)
                let played = (progress ?? 0) * size.width
                for (i, peak) in peaks.enumerated() {
                    let x = CGFloat(i) * step
                    let h = max(1, CGFloat(peak) * mid * 0.95)
                    let rect = CGRect(x: x, y: mid - h, width: max(1, step * 0.7), height: h * 2)
                    let color = progress != nil && x <= played ? tint : tint.opacity(progress == nil ? 0.75 : 0.35)
                    context.fill(Path(rect), with: .color(color))
                }
                if let progress {
                    let x = progress * size.width
                    context.fill(Path(CGRect(x: x - 0.75, y: 0, width: 1.5, height: size.height)),
                                 with: .color(.primary.opacity(0.8)))
                }
            }
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0).onEnded { value in
                onSeek?(min(1, max(0, value.location.x / geo.size.width)))
            })
            .overlay {
                if peaks.isEmpty { ProgressView().controlSize(.small) }
            }
        }
        .task(id: path) {
            peaks = []
            guard let client = app.client else { return }
            let buckets = 600
            if let result = try? await client.peaks(for: path, buckets: buckets) {
                peaks = result.peaks
            }
        }
    }
}
