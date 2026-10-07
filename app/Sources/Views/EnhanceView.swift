import SwiftUI

struct EnhanceView: View {
    @Environment(AppModel.self) private var app
    @AppStorage("mode.enhance") private var savedMode: ProcessingMode = .local
    @AppStorage("enhance.quality") private var quality = "fast"
    @AppStorage("enhance.isolate") private var isolate = false
    @AppStorage("enhance.strength") private var strength = 1.0
    @AppStorage("enhance.polish") private var polish = true
    @AppStorage("enhance.loudness") private var loudness = "podcast"
    @AppStorage("enhance.format") private var format = "wav"

    @State private var file: URL?
    @State private var request = PlanRequest(jobType: .enhance)
    @State private var plan: Plan?
    @State private var jobID: String?
    @State private var starting = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                ScreenHeader(title: "Enhance", subtitle: "Remove background noise and make speech clear and even.")
                FileDropZone(file: $file)
                if file != nil {
                    options
                    ProcessingModePicker(request: $request, plan: $plan)
                    StartBar(title: "Enhance Audio", symbol: "sparkles", plan: plan,
                             busy: starting || app.job(jobID)?.isActive == true) { start() }
                }
                if let job = app.job(jobID) {
                    result(job)
                }
            }
            .padding(24)
            .frame(maxWidth: 920, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .onAppear {
            request.mode = savedMode
            syncOptions()
            #if DEBUG
            if file == nil, let debugFile = DebugSupport.file { file = debugFile }
            #endif
        }
        #if DEBUG
        .onChange(of: plan) { _, plan in
            if DebugSupport.autostart, jobID == nil, !starting, let plan, !plan.blocking { start() }
        }
        #endif
        .onChange(of: request.mode) { _, mode in savedMode = mode }
        .onChange(of: file) { _, _ in jobID = nil }
        .onChange(of: OptionsKey(quality: quality, isolate: isolate, strength: strength, polish: polish,
                                 loudness: loudness, format: format)) { _, _ in syncOptions() }
    }

    private var options: some View {
        GroupBox {
            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 12) {
                GridRow {
                    Text("Noise removal")
                    Picker("Noise removal", selection: $quality) {
                        Text("Fast · DeepFilterNet").tag("fast")
                        Text("High quality · MossFormer2").tag("high")
                        Text("Basic · spectral gate").tag("basic")
                    }
                    .labelsHidden()
                    .fixedSize()
                }
                GridRow {
                    Text("Music / background")
                    Toggle("Separate the voice from music and background sounds first", isOn: $isolate)
                }
                GridRow {
                    Text("Strength")
                    HStack {
                        Slider(value: $strength, in: 0.3...1.0)
                            .frame(maxWidth: 260)
                        Text("\(Int(strength * 100))%").monospacedDigit().foregroundStyle(.secondary)
                    }
                }
                GridRow {
                    Text("Voice polish")
                    Toggle("EQ, de-esser and gentle compression", isOn: $polish)
                }
                GridRow {
                    Text("Loudness")
                    Picker("Loudness", selection: $loudness) {
                        Text("Podcast · −16 LUFS").tag("podcast")
                        Text("Social / Reels · −14 LUFS").tag("social")
                        Text("Broadcast · −23 LUFS").tag("broadcast")
                        Text("Keep original level").tag("off")
                    }
                    .labelsHidden()
                    .fixedSize()
                }
                GridRow {
                    Text("Save as")
                    Picker("Format", selection: $format) {
                        Text("WAV (24-bit)").tag("wav")
                        Text("M4A (AAC)").tag("m4a")
                        Text("MP3").tag("mp3")
                        Text("FLAC").tag("flac")
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .fixedSize()
                }
            }
            .padding(6)
        } label: {
            Text("Options").font(.headline)
        }
    }

    @ViewBuilder private func result(_ job: Job) -> some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 14) {
                JobProgressView(job: job) { jobID = $0.id }
                if job.status == "done", let enhanced = job.outputPath("enhanced") {
                    let original = job.outputPath("original") ?? URL(fileURLWithPath: job.inputPath)
                    ABPlayerView(items: [("Original", original), ("Enhanced", enhanced)])
                    MetricsRow(metrics: job.metrics)
                    HStack {
                        Button("Show in Finder") { app.reveal(enhanced) }
                        if let background = job.outputPath("background") {
                            Button("Show background track") { app.reveal(background) }
                        }
                    }
                }
            }
            .padding(6)
        } label: {
            Text("Result").font(.headline)
        }
    }

    private func syncOptions() {
        request.options = [
            "quality": .string(quality),
            "isolate_vocals": .bool(isolate),
            "strength": .number(strength),
            "polish": .bool(polish),
            "loudness": .string(loudness),
            "format": .string(format),
        ]
    }

    private func start() {
        guard let file else { return }
        starting = true
        Task {
            if let job = await app.submit(request, file: file) { jobID = job.id }
            starting = false
        }
    }
}

private struct OptionsKey: Equatable {
    var quality: String
    var isolate: Bool
    var strength: Double
    var polish: Bool
    var loudness: String
    var format: String
}

private struct MetricsRow: View {
    let metrics: [String: Double]

    var body: some View {
        HStack(spacing: 24) {
            if let before = metrics["input_noise_floor_db"], let after = metrics["output_noise_floor_db"] {
                metric("Background noise", "\(Int(before)) → \(Int(after)) dB",
                       "Quietest-moment level; lower is cleaner")
            }
            if let lufs = metrics["output_lufs"] {
                metric("Loudness", String(format: "%.1f LUFS", lufs), "Integrated loudness of the result")
            }
            if let peak = metrics["output_peak_db"] {
                metric("Peak", String(format: "%.1f dB", peak), "Highest sample; stays below 0 dB")
            }
        }
    }

    private func metric(_ title: String, _ value: String, _ help: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.title3.weight(.semibold)).monospacedDigit()
        }
        .help(help)
    }
}

struct ScreenHeader: View {
    let title: String
    let subtitle: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.largeTitle.weight(.bold))
            Text(subtitle).foregroundStyle(.secondary)
        }
    }
}

/// The Start button with the reason it's disabled, if any.
struct StartBar: View {
    let title: String
    let symbol: String
    let plan: Plan?
    let busy: Bool
    let action: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Button(action: action) {
                Label(title, systemImage: symbol).frame(minWidth: 160)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .keyboardShortcut(.return, modifiers: .command)
            .disabled(plan == nil || plan?.blocking == true || busy)

            if plan?.blocking == true {
                Label("Fix the steps marked in red first", systemImage: "exclamationmark.triangle")
                    .font(.callout).foregroundStyle(.red)
            } else if busy {
                Text("Working…").foregroundStyle(.secondary)
            }
        }
    }
}
