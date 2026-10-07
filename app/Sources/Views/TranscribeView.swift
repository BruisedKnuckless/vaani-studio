import SwiftUI

struct TranscribeView: View {
    @Environment(AppModel.self) private var app
    @AppStorage("mode.transcribe") private var savedMode: ProcessingMode = .auto
    @AppStorage("transcribe.source") private var source = Language.auto.rawValue
    @AppStorage("transcribe.target") private var target = ""
    @AppStorage("transcribe.clean") private var cleanFirst = false

    @State private var file: URL?
    @State private var request = PlanRequest(jobType: .transcribe)
    @State private var plan: Plan?
    @State private var jobID: String?
    @State private var starting = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                ScreenHeader(title: "Transcribe & Translate",
                             subtitle: "Turn speech into editable text and subtitles in Marathi, Hindi, English or Hinglish.")
                FileDropZone(file: $file)
                if file != nil {
                    options
                    ProcessingModePicker(request: $request, plan: $plan)
                    StartBar(title: target.isEmpty ? "Transcribe" : "Transcribe & Translate",
                             symbol: "text.bubble", plan: plan,
                             busy: starting || app.job(jobID)?.isActive == true) { start() }
                }
                if let job = app.job(jobID) {
                    GroupBox {
                        VStack(alignment: .leading, spacing: 12) {
                            JobProgressView(job: job) { jobID = $0.id }
                            if job.status == "done" {
                                TranscriptEditor(job: job) { jobID = $0.id }
                            }
                        }
                        .padding(6)
                    } label: {
                        Text("Transcript").font(.headline)
                    }
                }
            }
            .padding(24)
            .frame(maxWidth: 1000, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .onAppear {
            request.mode = savedMode
            sync()
            #if DEBUG
            if file == nil, DebugSupport.screen == .transcribe, let debugFile = DebugSupport.file { file = debugFile }
            #endif
        }
        #if DEBUG
        .onChange(of: plan) { _, plan in
            if DebugSupport.autostart, DebugSupport.screen == .transcribe, jobID == nil, !starting,
               let plan, !plan.blocking { start() }
        }
        #endif
        .onChange(of: request.mode) { _, mode in savedMode = mode }
        .onChange(of: file) { _, _ in jobID = nil }
        .onChange(of: [source, target]) { _, _ in sync() }
        .onChange(of: cleanFirst) { _, _ in sync() }
    }

    private var options: some View {
        GroupBox {
            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 12) {
                GridRow {
                    Text("Spoken language")
                    Picker("Spoken language", selection: $source) {
                        ForEach(Language.allCases) { Text($0.title).tag($0.rawValue) }
                    }
                    .labelsHidden().fixedSize()
                }
                GridRow {
                    Text("Translate to")
                    Picker("Translate to", selection: $target) {
                        Text("Don't translate").tag("")
                        ForEach(Language.allCases.filter { $0 != .auto }) { Text($0.title).tag($0.rawValue) }
                    }
                    .labelsHidden().fixedSize()
                }
                GridRow {
                    Text("Noisy audio")
                    Toggle("Clean up background noise before transcribing", isOn: $cleanFirst)
                }
            }
            .padding(6)
        } label: {
            Text("Options").font(.headline)
        }
    }

    private func sync() {
        request.srcLang = source
        request.tgtLang = target.isEmpty || target == source ? nil : target
        request.options = ["enhance_first": .bool(cleanFirst)]
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

/// One editable line of a transcript.
struct EditableSegment: Identifiable, Hashable {
    let id: Int
    var start: Double
    var end: Double
    var text: String
    var translation: String?

    init(index: Int, json: JSONValue) {
        id = index
        start = json["start"]?.doubleValue ?? 0
        end = json["end"]?.doubleValue ?? 0
        text = json["text"]?.stringValue ?? ""
        translation = json["translation"]?.stringValue
    }

    var json: JSONValue {
        var object: [String: JSONValue] = ["start": .number(start), "end": .number(end), "text": .string(text)]
        if let translation { object["translation"] = .string(translation) }
        return .object(object)
    }
}

struct TranscriptEditor: View {
    @Environment(AppModel.self) private var app
    let job: Job
    let onNewJob: (Job) -> Void

    @State private var segments: [EditableSegment] = []
    @State private var original: [EditableSegment] = []
    @State private var saving = false
    @State private var retranslateTarget = Language.en.rawValue
    @State private var player = ABPlayer()

    private var dirty: Bool { segments != original }
    private var language: String { job.outputs["language"]?.stringValue ?? "auto" }
    private var targetLanguage: String? { job.outputs["target_language"]?.stringValue }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            ABPlayerView(items: [("Original", URL(fileURLWithPath: job.inputPath))], player: player)

            HStack(spacing: 10) {
                Label(Language(rawValue: language)?.title ?? language, systemImage: "waveform")
                if let tgt = targetLanguage {
                    Image(systemName: "arrow.right")
                    Label(Language(rawValue: tgt)?.title ?? tgt, systemImage: "character.bubble")
                }
                Text("· \(segments.count) lines").foregroundStyle(.secondary)
                Spacer()
                if dirty {
                    Button(saving ? "Saving…" : "Save Edits") { save() }
                        .buttonStyle(.borderedProminent)
                        .disabled(saving)
                }
            }
            .font(.callout)

            if segments.isEmpty {
                Text("No speech was found in this file.").foregroundStyle(.secondary)
            }

            LazyVStack(alignment: .leading, spacing: 0) {
                ForEach($segments) { $segment in
                    SegmentRow(segment: $segment, showTranslation: targetLanguage != nil) {
                        player.play(from: segment.start)
                    }
                    Divider()
                }
            }

            HStack(spacing: 10) {
                Menu("Export") {
                    exportItem("Subtitles (.srt)", key: "transcript_srt")
                    exportItem("Web subtitles (.vtt)", key: "transcript_vtt")
                    exportItem("Plain text (.txt)", key: "transcript_txt")
                    if targetLanguage != nil {
                        Divider()
                        exportItem("Translated subtitles (.srt)", key: "translation_srt")
                        exportItem("Translated web subtitles (.vtt)", key: "translation_vtt")
                        exportItem("Translated text (.txt)", key: "translation_txt")
                    }
                    Divider()
                    exportItem("Everything (JSON)", key: "transcript_json")
                }
                .fixedSize()
                Button("Copy Text") { copyText() }
                Spacer()
                Picker("Translate to", selection: $retranslateTarget) {
                    ForEach(Language.allCases.filter { $0 != .auto && $0.rawValue != language }) {
                        Text($0.title).tag($0.rawValue)
                    }
                }
                .fixedSize()
                Button("Translate This Transcript") { retranslate() }
                    .help("Uses your edited text; speech recognition is not run again")
            }
        }
        .onAppear(perform: load)
        .onChange(of: job.updatedAt) { _, _ in if !dirty { load() } }
    }

    private func exportItem(_ title: String, key: String) -> some View {
        Button(title) {
            if let url = job.outputPath(key) { app.reveal(url) }
        }
        .disabled(job.outputPath(key) == nil)
    }

    private func load() {
        let raw = job.outputs["segments"]?.arrayValue ?? []
        segments = raw.enumerated().map { EditableSegment(index: $0.offset, json: $0.element) }
        original = segments
        if let tgt = targetLanguage { retranslateTarget = tgt }
    }

    private func save() {
        saving = true
        Task {
            if let updated = try? await app.client?.saveSegments(job.id, segments: segments.map(\.json)) {
                original = segments
                app.upsert(updated)
            } else {
                app.lastError = "Couldn't save the edits."
            }
            saving = false
        }
    }

    private func copyText() {
        let text = segments.map { seg in
            [seg.text, seg.translation].compactMap { $0 }.joined(separator: "\n")
        }.joined(separator: "\n\n")
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }

    private func retranslate() {
        var request = PlanRequest(jobType: .transcribe)
        request.srcLang = language
        request.tgtLang = retranslateTarget
        request.mode = ProcessingMode(rawValue: job.plan.mode) ?? .auto
        request.options = ["segments": .array(segments.map(\.json))]
        Task {
            if let new = await app.submit(request, file: URL(fileURLWithPath: job.inputPath)) { onNewJob(new) }
        }
    }
}

private struct SegmentRow: View {
    @Binding var segment: EditableSegment
    let showTranslation: Bool
    let onPlay: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Button(action: onPlay) {
                VStack(alignment: .leading, spacing: 2) {
                    Image(systemName: "play.circle")
                    Text(timestamp(segment.start)).font(.caption.monospacedDigit())
                }
                .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .help("Play from here")
            .frame(width: 56, alignment: .leading)

            VStack(alignment: .leading, spacing: 6) {
                TextField("", text: $segment.text, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(.body)
                if showTranslation {
                    TextField("", text: Binding(get: { segment.translation ?? "" },
                                                set: { segment.translation = $0 }), axis: .vertical)
                        .textFieldStyle(.plain)
                        .foregroundStyle(Color.accentColor)
                }
            }
        }
        .padding(.vertical, 8)
    }

    private func timestamp(_ seconds: Double) -> String {
        let total = Int(seconds)
        return String(format: "%d:%02d", total / 60, total % 60)
    }
}
