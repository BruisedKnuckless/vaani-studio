import SwiftUI

struct DubView: View {
    @Environment(AppModel.self) private var app
    @AppStorage("mode.dub") private var savedMode: ProcessingMode = .auto
    @AppStorage("dub.source") private var source = Language.auto.rawValue
    @AppStorage("dub.target") private var target = Language.en.rawValue
    @AppStorage("dub.background") private var keepBackground = true
    @AppStorage("dub.voiceMatch") private var voiceMatch = false
    @AppStorage("dub.format") private var format = "wav"
    @AppStorage("dub.consent") private var consent = false
    @AppStorage("dub.voice") private var voiceChoice = "auto"
    @AppStorage("dub.voiceName") private var voiceName = "shubh"

    @State private var file: URL?
    @State private var request = PlanRequest(jobType: .dub)
    @State private var plan: Plan?
    @State private var jobID: String?
    @State private var starting = false

    private static let targets: [Language] = [.en, .hi, .mr]
    /// Bulbul v3 preset voices (Sarvam).
    private static let studioVoices = ["shubh", "aditya", "ritu", "priya", "neha", "rahul", "pooja", "rohan",
        "simran", "kavya", "amit", "dev", "ishita", "shreya", "ratan", "varun", "manan", "sumit", "roopa", "kabir",
        "aayan", "ashutosh", "advait", "anand", "tanya", "tarun", "sunny", "mani", "gokul", "vijay", "shruti",
        "suhani", "mohit", "kavitha", "rehan", "soham", "rupali"]

    private var clonesVoice: Bool { voiceChoice != "sarvam_studio" }
    /// Voice matching only applies to voices made on this Mac.
    private var voiceMadeLocally: Bool {
        voiceChoice == "local" || (voiceChoice == "auto" && (!app.hasKey(.sarvam) || request.mode == .local))
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                ScreenHeader(title: "Dub",
                             subtitle: "Translate speech and say it in the original speaker's voice.")
                FileDropZone(file: $file)
                if file != nil {
                    options
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
        .onChange(of: request.mode) { _, mode in savedMode = mode }
        .onChange(of: file) { _, _ in jobID = nil }
        .onChange(of: [source, target, format, voiceChoice, voiceName]) { _, _ in sync() }
        .onChange(of: [keepBackground, voiceMatch]) { _, _ in sync() }
        .onChange(of: request.mode) { _, _ in sync() }
    }

    private var options: some View {
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
                GridRow {
                    Text("Voice")
                    VStack(alignment: .leading, spacing: 6) {
                        Picker("Voice", selection: $voiceChoice) {
                            Text("Automatic: Sarvam if you've added a key, otherwise this Mac").tag("auto")
                            Text("Speaker's own voice · Sarvam (most natural, cloud)").tag("sarvam_clone")
                            Text("Speaker's own voice · on this Mac (free, private)").tag("local")
                            Text("Sarvam studio voice (natural, but not the speaker's)").tag("sarvam_studio")
                        }
                        .labelsHidden().fixedSize()
                        if voiceChoice == "sarvam_studio" {
                            Picker("Studio voice", selection: $voiceName) {
                                ForEach(Self.studioVoices, id: \.self) { Text($0.capitalized).tag($0) }
                            }
                            .fixedSize()
                        }
                        if voiceChoice != "local", !app.hasKey(.sarvam) {
                            HStack(spacing: 6) {
                                Text(voiceChoice == "auto" ? "No Sarvam key yet, so the voice is made on this Mac."
                                                           : "Needs a Sarvam API key.")
                                Button("Add key…") { app.screen = .settings }.controlSize(.small)
                            }
                            .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
                GridRow {
                    Text("Background")
                    Toggle("Keep the original music and ambience", isOn: $keepBackground)
                }
                if voiceMadeLocally {
                    GridRow {
                        Text("Voice likeness")
                        Toggle("Extra voice-matching pass (slower, closer to the speaker)", isOn: $voiceMatch)
                    }
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
            Text("Options").font(.headline)
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
        request.options = ["keep_background": .bool(keepBackground), "voice_match": .bool(voiceMatch && voiceMadeLocally),
                           "format": .string(format), "voice": .string(voiceChoice),
                           "voice_name": .string(voiceName)]
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
        if let voice = job.outputPath("dubbed_voice") { items.append(("Voice only", voice)) }
        return items
    }

    @ViewBuilder private var summary: some View {
        let m = job.metrics
        HStack(spacing: 20) {
            if let lines = m["lines"] { stat("Lines voiced", "\(Int(lines))") }
            if let fast = m["sped_up"] { stat("Sped up to fit", "\(Int(fast))") }
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
