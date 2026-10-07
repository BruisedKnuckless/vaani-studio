import SwiftUI

/// Progress, failure and the "used" summary for one job.
struct JobProgressView: View {
    @Environment(AppModel.self) private var app
    let job: Job
    var onRetry: ((Job) -> Void)? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            switch job.status {
            case "queued", "running":
                HStack {
                    ProgressView(value: job.progress).frame(maxWidth: .infinity)
                    Text("\(Int(job.progress * 100))%").monospacedDigit().foregroundStyle(.secondary)
                    Button("Cancel") { Task { await app.cancel(job) } }
                }
                Text(job.message).font(.callout).foregroundStyle(.secondary)
            case "failed":
                Label(job.error?.message ?? job.message, systemImage: "xmark.octagon.fill")
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
                if job.error?.localAlternative != nil {
                    Button("Run that step on this Mac instead") {
                        Task {
                            if let retried = await app.retryLocally(job) { onRetry?(retried) }
                        }
                    }
                }
            case "cancelled":
                Label("Cancelled", systemImage: "stop.circle").foregroundStyle(.secondary)
            default:
                Label("Finished in \(formatDuration(job.updatedAt - job.createdAt))", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                ForEach(job.outputs["warnings"]?.arrayValue?.compactMap(\.stringValue) ?? [], id: \.self) { warning in
                    Label(warning, systemImage: "exclamationmark.triangle.fill")
                        .font(.callout)
                        .foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            if !job.providersUsed.isEmpty {
                ProvidersUsedView(job: job)
            }
        }
    }
}

struct ProvidersUsedView: View {
    let job: Job

    var body: some View {
        HStack(spacing: 6) {
            Text("Used:").font(.caption).foregroundStyle(.secondary)
            ForEach(job.providersUsed.sorted(by: { $0.key < $1.key }), id: \.key) { _, id in
                let info = names[id] ?? (name: id, kind: "local")
                Label(info.name, systemImage: info.kind == "cloud" ? "cloud.fill" : "desktopcomputer")
                    .font(.caption)
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(Capsule().fill(.quaternary))
            }
        }
    }

    private var names: [String: (name: String, kind: String)] {
        var map: [String: (name: String, kind: String)] = [:]
        for step in job.plan.steps {
            for choice in step.alternatives { map[choice.id] = (choice.name, choice.kind) }
            map[step.providerId] = (step.providerName, step.kind)
        }
        return map
    }
}
