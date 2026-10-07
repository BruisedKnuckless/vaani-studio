import SwiftUI

struct JobsView: View {
    @Environment(AppModel.self) private var app
    @State private var selection: Job.ID?

    var body: some View {
        HSplitView {
            List(app.jobs, selection: $selection) { job in
                JobRow(job: job).tag(job.id)
                    .contextMenu {
                        Button("Show in Finder") { app.reveal(URL(fileURLWithPath: job.outDir)) }
                        if job.isActive { Button("Cancel") { Task { await app.cancel(job) } } }
                        Button("Remove from List", role: .destructive) { Task { await app.delete(job) } }
                    }
            }
            .frame(minWidth: 320, idealWidth: 380)
            .overlay {
                if app.jobs.isEmpty {
                    ContentUnavailableView("No jobs yet", systemImage: "tray",
                                           description: Text("Enhance, transcribe or dub a file and it shows up here."))
                }
            }

            Group {
                if let job = app.job(selection) {
                    JobDetail(job: job) { selection = $0.id }
                } else {
                    Text("Select a job").foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .frame(minWidth: 420)
        }
        .navigationTitle("Jobs")
        .task { await app.refreshJobs() }
    }
}

private struct JobRow: View {
    let job: Job

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: symbol).foregroundStyle(color).frame(width: 18)
            VStack(alignment: .leading, spacing: 2) {
                Text(job.inputName).lineLimit(1).truncationMode(.middle)
                HStack(spacing: 6) {
                    Text(job.type.capitalized)
                    Text("·")
                    Text(job.created, style: .relative)
                    if job.plan.uploads { Image(systemName: "cloud").help("Used a cloud API") }
                }
                .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            if job.isActive {
                ProgressView(value: job.progress).frame(width: 60)
            }
        }
        .padding(.vertical, 3)
    }

    private var symbol: String {
        switch job.status {
        case "done": "checkmark.circle.fill"
        case "failed": "xmark.octagon.fill"
        case "cancelled": "stop.circle"
        case "running": "gearshape.2"
        default: "clock"
        }
    }

    private var color: Color {
        switch job.status {
        case "done": .green
        case "failed": .red
        case "running": .accentColor
        default: .secondary
        }
    }
}

private struct JobDetail: View {
    @Environment(AppModel.self) private var app
    let job: Job
    let onRetry: (Job) -> Void
    @State private var showLog = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Text(job.inputName).font(.title2.weight(.semibold)).textSelection(.enabled)
                Text("\(job.type.capitalized) · mode: \(ProcessingMode(rawValue: job.plan.mode)?.title ?? job.plan.mode)")
                    .foregroundStyle(.secondary)
                JobProgressView(job: job, onRetry: onRetry)

                GroupBox("Plan") {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(job.plan.steps) { step in
                            HStack {
                                Image(systemName: step.isCloud ? "cloud.fill" : "desktopcomputer")
                                    .foregroundStyle(step.isCloud ? .blue : .secondary).frame(width: 18)
                                Text(step.label).frame(width: 130, alignment: .leading)
                                Text(step.providerName)
                                Spacer()
                            }
                            .font(.callout)
                        }
                    }
                    .padding(4)
                }

                if job.status == "done" {
                    GroupBox("Files") {
                        VStack(alignment: .leading, spacing: 6) {
                            ForEach(job.outputs.keys.sorted(), id: \.self) { key in
                                if let url = job.outputPath(key), key != "original" {
                                    HStack {
                                        Text(key.replacingOccurrences(of: "_", with: " ").capitalized)
                                            .frame(width: 120, alignment: .leading)
                                        Text(url.lastPathComponent).foregroundStyle(.secondary)
                                        Spacer()
                                        Button("Show") { app.reveal(url) }.controlSize(.small)
                                    }
                                }
                            }
                        }
                        .padding(4)
                    }
                }

                DisclosureGroup("Log", isExpanded: $showLog) {
                    Text(job.log.joined(separator: "\n"))
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }

                HStack {
                    Button("Show Output Folder") { app.reveal(URL(fileURLWithPath: job.outDir)) }
                    if job.isActive { Button("Cancel") { Task { await app.cancel(job) } } }
                }
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}
