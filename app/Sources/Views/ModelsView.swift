import SwiftUI

struct ModelsView: View {
    @Environment(AppModel.self) private var app
    @State private var models: [ModelInfo] = []

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                ScreenHeader(title: "Models",
                             subtitle: "On-device AI models. They download automatically on first use, or you can fetch them now.")
                if !app.hasKey(.huggingface), models.contains(where: { $0.gated && $0.state != "downloaded" }) {
                    Label {
                        Text("Some models (marked “Accept terms”) need a free Hugging Face token. Open the model page, accept its terms, then add your token in Settings.")
                    } icon: { Image(systemName: "key") }
                    .font(.callout)
                    .padding(10)
                    .background(RoundedRectangle(cornerRadius: 8).fill(.orange.opacity(0.1)))
                }
                VStack(spacing: 0) {
                    ForEach(models) { model in
                        ModelRow(model: model) { await refresh() }
                        Divider()
                    }
                }
                Text("Total on disk: \(formatSize(models.reduce(0) { $0 + $1.diskMb }))")
                    .font(.callout).foregroundStyle(.secondary)
            }
            .padding(24)
            .frame(maxWidth: 920, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .task {
            while !Task.isCancelled {
                await refresh()
                let downloading = models.contains { $0.state == "downloading" }
                try? await Task.sleep(for: .seconds(downloading ? 1 : 5))
            }
        }
    }

    private func refresh() async {
        if let list = try? await app.client?.models() { models = list }
    }
}

private struct ModelRow: View {
    @Environment(AppModel.self) private var app
    let model: ModelInfo
    let refresh: () async -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(model.name).font(.headline)
                    if model.gated { Tag(text: "Accept terms", color: .orange) }
                    if !model.commercialOk { Tag(text: "Non-commercial", color: .purple) }
                }
                Text(model.purpose).font(.callout)
                Text("\(formatSize(model.sizeMb)) · \(model.license)")
                    .font(.caption).foregroundStyle(.secondary)
                if let error = model.error {
                    Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled)
                }
            }
            Spacer()
            actions
        }
        .padding(.vertical, 10)
    }

    @ViewBuilder private var actions: some View {
        HStack {
            if let terms = model.termsUrl, let url = URL(string: terms), model.state != "downloaded" {
                Link("Model page", destination: url).font(.callout)
            }
            switch model.state {
            case "managed":
                Text("Automatic").foregroundStyle(.secondary).font(.callout)
            case "downloaded":
                Label("Downloaded", systemImage: "checkmark.circle.fill").foregroundStyle(.green).font(.callout)
                Button("Delete", role: .destructive) {
                    Task { _ = try? await app.client?.delete(model: model.id); await refresh() }
                }
            case "downloading":
                ProgressView(value: model.progress ?? 0).frame(width: 120)
                Text("\(Int((model.progress ?? 0) * 100))%").monospacedDigit().font(.caption)
            default:
                Button("Download") {
                    Task { _ = try? await app.client?.download(model: model.id); await refresh() }
                }
            }
        }
    }
}

func formatSize(_ mb: Int) -> String {
    mb >= 1000 ? String(format: "%.1f GB", Double(mb) / 1000) : "\(mb) MB"
}
