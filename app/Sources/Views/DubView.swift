import AVFoundation
import SwiftUI

struct DubView: View {
    @Environment(AppModel.self) private var app
    @AppStorage("mode.dub") private var savedMode: ProcessingMode = .auto
    @AppStorage("dub.source") private var source = Language.auto.rawValue
    @AppStorage("dub.target") private var target = Language.en.rawValue
    @AppStorage("dub.withBackground") private var withBackground = false
    @AppStorage("dub.voiceMatch") private var voiceMatch = false
    @AppStorage("dub.format") private var format = "wav"
    @AppStorage("dub.consent") private var consent = false
    @AppStorage("dub.voiceKind") private var voiceKind = "clone"        // clone | studio
    @AppStorage("dub.cloneEngine") private var cloneEngine = "auto"     // auto | sarvam_clone | local
    @AppStorage("dub.voiceName") private var voiceName = "shubh"
    @AppStorage("dub.pace") private var pace = 1.0

    @State private var file: URL?
    @State private var request = PlanRequest(jobType: .dub)
    @State private var plan: Plan?
    @State private var jobID: String?
    @State private var starting = false

    private static let targets: [Language] = [.en, .hi, .mr]

    private var clonesVoice: Bool { voiceKind == "clone" }
    private var voiceOption: String { clonesVoice ? cloneEngine : "sarvam_studio" }
    /// Voice matching only applies to voices made on this Mac.
    private var voiceMadeLocally: Bool {
        clonesVoice && (cloneEngine == "local" || (cloneEngine == "auto" && (!app.hasKey(.sarvam) || request.mode == .local)))
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                ScreenHeader(title: "Dub",
                             subtitle: "Translate speech and voice it naturally, in the speaker's own voice or a studio voice.")
                FileDropZone(file: $file)
                if file != nil {
                    languages
                    VoiceChooser(kind: $voiceKind, cloneEngine: $cloneEngine, voiceName: $voiceName,
                                 language: target)
                    output
                    ProcessingModePicker(request: $request, plan: $plan)
                    if clonesVoice { consentRow }
                    StartBar(title: "Dub Audio", symbol: "person.wave.2", plan: (consent || !clonesVoice) ? plan : nil,
                             busy: starting || app.job(jobID)?.isActive == true) { start() }
                }
                if let job = app.job(jobID) {
                    GroupBox {
                        VStack(alignment: .leading, spacing: 14) {
                            JobProgressView(job: job) { jobID = $0.id }
                            if job.status == "done" {
                                DubResult(job: job) { jobID = $0.id }
                            }
                        }
                        .padding(6)
                    } label: {
                        Text("Result").font(.headline)
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
            if file == nil, DebugSupport.screen == .dub, let debugFile = DebugSupport.file { file = debugFile }
            #endif
        }
        #if DEBUG
        .onChange(of: plan) { _, plan in
            if DebugSupport.autostart, DebugSupport.screen == .dub, consent || !clonesVoice, jobID == nil, !starting,
               let plan, !plan.blocking { start() }
        }
        #endif
        .onChange(of: request.mode) { _, mode in
            savedMode = mode
            sync()
        }
        .onChange(of: file) { _, _ in jobID = nil }
        .onChange(of: [source, target, format, voiceKind, cloneEngine, voiceName]) { _, _ in sync() }
        .onChange(of: [withBackground, voiceMatch]) { _, _ in sync() }
        .onChange(of: pace) { _, _ in sync() }
    }

    private var languages: some View {
        GroupBox {
            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 12) {
                GridRow {
                    Text("Speaker's language")
                    Picker("Speaker's language", selection: $source) {
                        ForEach([Language.auto, .mr, .hi, .en]) { Text($0.title).tag($0.rawValue) }
                    }
                    .labelsHidden().fixedSize()
                }
                GridRow {
                    Text("Dub into")
                    Picker("Dub into", selection: $target) {
                        ForEach(Self.targets) { Text($0.title).tag($0.rawValue) }
                    }
                    .pickerStyle(.segmented).labelsHidden().fixedSize()
                }
            }
            .padding(6)
        } label: {
            Text("Languages").font(.headline)
        }
    }

    private var output: some View {
        GroupBox {
            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 12) {
                GridRow {
                    Text("Output")
                    VStack(alignment: .leading, spacing: 4) {
                        Text("The dubbed voice on its own, at natural speed, cleaned up and levelled.")
                            .foregroundStyle(.secondary)
                        Toggle("Also make a version with the original background music", isOn: $withBackground)
                    }
                }
                if voiceMadeLocally {
                    GridRow {
                        Text("Voice likeness")
                        Toggle("Extra voice-matching pass (slower, closer to the speaker)", isOn: $voiceMatch)
                    }
                }
                GridRow {
                    Text("Speaking pace")
                    Picker("Speaking pace", selection: $pace) {
                        Text("Relaxed").tag(0.9)
                        Text("Natural").tag(1.0)
                        Text("Brisk").tag(1.1)
                    }
                    .pickerStyle(.segmented).labelsHidden().fixedSize()
                }
                GridRow {
                    Text("Save as")
                    Picker("Format", selection: $format) {
                        Text("WAV").tag("wav")
                        Text("M4A").tag("m4a")
                        Text("MP3").tag("mp3")
                    }
                    .pickerStyle(.segmented).labelsHidden().fixedSize()
                }
            }
            .padding(6)
        } label: {
            Text("Output").font(.headline)
        }
    }

    private var consentRow: some View {
        Toggle(isOn: $consent) {
            VStack(alignment: .leading, spacing: 2) {
                Text("I have permission to clone this speaker's voice")
                Text("Vaani copies the voice in the file. Only dub people who've agreed to it, such as yourself.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private func sync() {
        request.srcLang = source
        request.tgtLang = target
        request.options = [
            "voice": .string(voiceOption),
            "voice_name": .string(voiceName),
            "timing": .string("natural"),
            "pace": .number(pace),
            "clarity": .bool(true),
            "keep_background": .bool(withBackground),
            "voice_match": .bool(voiceMatch && voiceMadeLocally),
            "format": .string(format),
        ]
    }

    private func start() {
        guard let file, consent || !clonesVoice else { return }
        starting = true
        Task {
            if let job = await app.submit(request, file: file) { jobID = job.id }
            starting = false
        }
    }
}

/// Two ways to voice the dub: the speaker's own (cloned) voice, or a Sarvam studio voice.
private struct VoiceChooser: View {
    @Environment(AppModel.self) private var app
    @Binding var kind: String
    @Binding var cloneEngine: String
    @Binding var voiceName: String
    let language: String
    @State private var voices: [StudioVoice] = []
    @State private var preview = PreviewPlayer()

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 10) {
                    card("clone", title: "Speaker's own voice", symbol: "person.wave.2",
                         detail: "The original speaker, speaking the translation.")
                    card("studio", title: "Studio voice", symbol: "music.mic",
                         detail: "A natural, professional Sarvam voice that you choose.")
                }
                if kind == "clone" {
                    HStack {
                        Text("Made by")
                        Picker("Made by", selection: $cloneEngine) {
                            Text("Automatic").tag("auto")
                            Text("Sarvam · most natural").tag("sarvam_clone")
                            Text("This Mac · free, private").tag("local")
                        }
                        .pickerStyle(.segmented).labelsHidden().fixedSize()
                    }
                    if cloneEngine != "local", !app.hasKey(.sarvam) { keyHint }
                } else {
                    HStack(spacing: 10) {
                        Text("Voice")
                        Picker("Voice", selection: $voiceName) {
                            ForEach(voices) { voice in
                                Text(voice.recommended ? "\(voice.name) · top-rated" : voice.name).tag(voice.id)
                            }
                        }
                        .labelsHidden().fixedSize()
                        Button {
                            Task { await preview.toggle(voice: voiceName, language: language, app: app) }
                        } label: {
                            if preview.loading {
                                ProgressView().controlSize(.small)
                            } else {
                                Label(preview.playing ? "Stop" : "Preview",
                                      systemImage: preview.playing ? "stop.fill" : "play.fill")
                            }
                        }
                        .disabled(!app.hasKey(.sarvam) || preview.loading)
                    }
                    if let error = preview.error {
                        Text(error).font(.caption).foregroundStyle(.red)
                    }
                    if !app.hasKey(.sarvam) { keyHint }
                }
            }
            .padding(6)
        } label: {
            Text("Voice").font(.headline)
        }
        .task {
            if voices.isEmpty, let list = try? await app.client?.studioVoices() { voices = list }
        }
        .onChange(of: voiceName) { _, _ in preview.stop() }
        .onChange(of: language) { _, _ in preview.stop() }
    }

    private var keyHint: some View {
        HStack(spacing: 6) {
            Text(kind == "clone" ? "No Sarvam key yet, so the voice is made on this Mac."
                                 : "Studio voices need a Sarvam API key.")
            Button("Add key…") { app.screen = .settings }.controlSize(.small)
        }
        .font(.caption).foregroundStyle(.secondary)
    }

    private func card(_ value: String, title: String, symbol: String, detail: String) -> some View {
        let selected = kind == value
        return VStack(alignment: .leading, spacing: 4) {
            Label(title, systemImage: symbol).font(.callout.weight(.semibold))
            Text(detail).font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 10).fill(selected ? Color.accentColor.opacity(0.14) : .clear))
        .overlay(RoundedRectangle(cornerRadius: 10)
            .strokeBorder(selected ? Color.accentColor : Color.secondary.opacity(0.3), lineWidth: selected ? 2 : 1))
        .contentShape(Rectangle())
        .onTapGesture { kind = value }
        .accessibilityAddTraits(selected ? [.isButton, .isSelected] : .isButton)
    }
}

/// Plays a cached sample of a studio voice.
@MainActor
@Observable
final class PreviewPlayer {
    private(set) var loading = false
    private(set) var playing = false
    private(set) var error: String?
    private var player: AVAudioPlayer?
    private var timer: Timer?

    func toggle(voice: String, language: String, app: AppModel) async {
        if playing { stop(); return }
        guard let client = app.client else { return }
        loading = true
        error = nil
        defer { loading = false }
        do {
            let result = try await client.previewVoice(voice, language: language)
            let player = try AVAudioPlayer(contentsOf: URL(fileURLWithPath: result.path))
            self.player = player
            player.play()
            playing = true
            timer?.invalidate()
            timer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] _ in
                MainActor.assumeIsolated {
                    if self?.player?.isPlaying != true { self?.stop() }
                }
            }
        } catch {
            self.error = "Couldn't play the preview: \(error.localizedDescription)"
        }
    }

    func stop() {
        player?.stop()
        playing = false
        timer?.invalidate()
    }
}

private struct DubResult: View {
    @Environment(AppModel.self) private var app
    let job: Job
    let onNewJob: (Job) -> Void
    @State private var player = ABPlayer()
    @State private var segments: [EditableSegment] = []
    @State private var original: [EditableSegment] = []
    @State private var showScript = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let dubbed = job.outputPath("dubbed") {
                ABPlayerView(items: playerItems(dubbed), player: player)
            }
            summary
            HStack {
                if let dubbed = job.outputPath("dubbed") { Button("Show in Finder") { app.reveal(dubbed) } }
                if let srt = job.outputPath("subtitles") { Button("Subtitles (.srt)") { app.reveal(srt) } }
                Spacer()
            }
            DisclosureGroup("Script: fix a translation and dub again", isExpanded: $showScript) {
                VStack(alignment: .leading, spacing: 0) {
                    ForEach($segments) { $segment in
                        HStack(alignment: .top, spacing: 10) {
                            Button { player.play(from: segment.start) } label: {
                                Image(systemName: "play.circle").foregroundStyle(.secondary)
                            }
                            .buttonStyle(.plain)
                            VStack(alignment: .leading, spacing: 4) {
                                Text(segment.text).font(.callout).foregroundStyle(.secondary)
                                TextField("", text: Binding(get: { segment.translation ?? "" },
                                                            set: { segment.translation = $0 }), axis: .vertical)
                                    .textFieldStyle(.plain)
                            }
                        }
                        .padding(.vertical, 6)
                        Divider()
                    }
                    HStack {
                        Spacer()
                        Button("Dub Again With My Edits") { redub() }
                            .buttonStyle(.borderedProminent)
                            .disabled(segments == original)
                    }
                    .padding(.top, 8)
                }
                .padding(.top, 6)
            }
        }
        .onAppear(perform: load)
        .onChange(of: job.id) { _, _ in load() }
    }

    private func playerItems(_ dubbed: URL) -> [(label: String, url: URL)] {
        var items: [(label: String, url: URL)] = [("Original", URL(fileURLWithPath: job.inputPath)), ("Dubbed", dubbed)]
        if let mix = job.outputPath("dubbed_with_background") { items.append(("With background", mix)) }
        return items
    }

    @ViewBuilder private var summary: some View {
        let m = job.metrics
        HStack(spacing: 20) {
            if let lines = m["lines"] { stat("Passages voiced", "\(Int(lines))") }
            if let length = m["duration_s"] {
                stat("Length", "\(formatDuration(length))" + (m["original_duration_s"].map { " (original \(formatDuration($0)))" } ?? ""))
            }
            if let fast = m["sped_up"], fast > 0 { stat("Sped up to fit", "\(Int(fast))") }
            if let cut = m["cut_short"], cut > 0 {
                stat("Too long (edit these)", "\(Int(cut))").foregroundStyle(.orange)
            }
        }
    }

    private func stat(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.title3.weight(.semibold)).monospacedDigit()
        }
    }

    private func load() {
        let raw = job.outputs["segments"]?.arrayValue ?? []
        segments = raw.enumerated().map { EditableSegment(index: $0.offset, json: $0.element) }
        original = segments
    }

    private func redub() {
        var request = PlanRequest(jobType: .dub)
        request.srcLang = job.outputs["language"]?.stringValue ?? "auto"
        request.tgtLang = job.outputs["target_language"]?.stringValue
        request.mode = ProcessingMode(rawValue: job.plan.mode) ?? .auto
        var options = job.params["options"]?.objectValue ?? [:]
        options["script"] = .array(segments.map(\.json))
        request.options = options
        Task {
            if let new = await app.submit(request, file: URL(fileURLWithPath: job.inputPath)) { onNewJob(new) }
        }
    }
}
