import SwiftUI

/// Shown after a file is added: choose On this Mac / Cloud API / Auto, see
/// exactly which service will run each step, and fix anything blocking.
struct ProcessingModePicker: View {
    @Environment(AppModel.self) private var app
    @Binding var request: PlanRequest
    /// Updated with the latest plan so the screen can enable/disable Start.
    @Binding var plan: Plan?
    @State private var loading = false
    @State private var showAdvanced = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                Text("Processing").font(.headline)
                Spacer()
                if loading { ProgressView().controlSize(.small) }
            }

            HStack(spacing: 8) {
                ForEach(ProcessingMode.allCases) { mode in
                    ModeCard(mode: mode, selected: request.mode == mode, disabled: isDisabled(mode))
                        .onTapGesture {
                            guard !isDisabled(mode) else { return }
                            request.mode = mode
                            request.overrides = [:]
                        }
                }
            }

            if let note = disabledNote {
                Label(note, systemImage: "info.circle").font(.caption).foregroundStyle(.secondary)
            }

            if let plan {
                willUse(plan)
            }
        }
        .padding(14)
        .background(RoundedRectangle(cornerRadius: 12).fill(.background.secondary))
        .task(id: PreviewKey(request: request, keys: app.health?.keys ?? [:], offline: app.offline)) {
            try? await Task.sleep(for: .milliseconds(120))  // debounce rapid option changes
            guard !Task.isCancelled, let client = app.client else { return }
            loading = true
            defer { loading = false }
            if let fresh = try? await client.preview(request) {
                plan = fresh
                // Cloud picked but this job type has no cloud step at all (e.g. Enhance).
                if fresh.cloudAvailable == false, request.mode == .cloud { request.mode = .local }
            }
        }
    }

    private func isDisabled(_ mode: ProcessingMode) -> Bool {
        guard mode == .cloud else { return false }
        return app.offline || plan?.cloudAvailable == false
    }

    private var disabledNote: String? {
        if app.offline { return "Offline mode is on, so cloud APIs are turned off. Change it in Settings." }
        if plan?.cloudAvailable == false {
            return "No cloud service removes noise yet (Sarvam and OpenAI don't offer it), so this runs on your Mac."
        }
        return nil
    }

    @ViewBuilder private func willUse(_ plan: Plan) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Will use").font(.subheadline.weight(.semibold)).foregroundStyle(.secondary)
            ForEach(plan.steps) { step in
                StepRow(step: step, override: overrideBinding(for: step.stage))
            }
            if plan.uploads {
                let services = Set(plan.steps.filter(\.isCloud).map(\.providerName)).sorted().joined(separator: ", ")
                Label("Your audio or text will be uploaded to: \(services)", systemImage: "exclamationmark.icloud")
                    .font(.callout)
                    .foregroundStyle(.orange)
                    .padding(.top, 2)
            }
            DisclosureGroup("Advanced: choose the service for each step", isExpanded: $showAdvanced) {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(plan.steps.filter { $0.alternatives.count > 1 }) { step in
                        Picker(step.label, selection: overrideBinding(for: step.stage)) {
                            Text("Automatic").tag(String?.none)
                            ForEach(step.alternatives) { choice in
                                Text("\(choice.name) · \(choice.kind == "cloud" ? "cloud" : "this Mac")\(statusSuffix(choice.status))")
                                    .tag(Optional(choice.id))
                            }
                        }
                    }
                    if !request.overrides.isEmpty {
                        Button("Reset to \(request.mode.title)") { request.overrides = [:] }
                    }
                }
                .padding(.top, 6)
            }
            .font(.callout)
            .padding(.top, 4)
        }
    }

    private func statusSuffix(_ status: String) -> String {
        switch status {
        case "missing_key": " (needs key)"
        case "needs_download": " (downloads first)"
        case "unavailable": " (unavailable)"
        default: ""
        }
    }

    private func overrideBinding(for stage: String) -> Binding<String?> {
        Binding(
            get: { request.overrides[stage] },
            set: { request.overrides[stage] = $0 }
        )
    }
}

private struct PreviewKey: Hashable {
    var request: PlanRequest
    var keys: [String: Bool]
    var offline: Bool
}

private struct ModeCard: View {
    let mode: ProcessingMode
    let selected: Bool
    let disabled: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Label(mode.title, systemImage: mode.symbol).font(.callout.weight(.semibold))
            Text(mode.explanation).font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 10)
            .fill(selected ? Color.accentColor.opacity(0.14) : Color.clear))
        .overlay(RoundedRectangle(cornerRadius: 10)
            .strokeBorder(selected ? Color.accentColor : Color.secondary.opacity(0.3), lineWidth: selected ? 2 : 1))
        .opacity(disabled ? 0.4 : 1)
        .contentShape(Rectangle())
        .accessibilityAddTraits(selected ? [.isButton, .isSelected] : .isButton)
    }
}

private struct StepRow: View {
    @Environment(AppModel.self) private var app
    let step: StagePlan
    @Binding var override: String?

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: step.isCloud ? "cloud.fill" : "desktopcomputer")
                .foregroundStyle(step.isCloud ? .blue : .secondary)
                .frame(width: 18)
            Text(step.label).frame(width: 130, alignment: .leading)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(step.providerName).fontWeight(.medium)
                    badge
                }
                if !step.note.isEmpty {
                    Text(step.note).font(.caption).foregroundStyle(.secondary)
                }
                if step.status == "missing_key" {
                    HStack {
                        if let local = step.localAlternative {
                            Button("Run this step locally instead") { override = local }
                        }
                        Button("Add key…") { app.screen = .settings }
                    }
                    .controlSize(.small)
                }
            }
            Spacer(minLength: 0)
        }
        .font(.callout)
    }

    @ViewBuilder private var badge: some View {
        switch step.status {
        case "missing_key": Tag(text: "Needs API key", color: .red)
        case "unavailable": Tag(text: "Unavailable", color: .red)
        case "needs_download": Tag(text: "Downloads first", color: .orange)
        case "local_only": Tag(text: "Runs on Mac", color: .secondary)
        default: EmptyView()
        }
    }
}

struct Tag: View {
    let text: String
    let color: Color

    var body: some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(Capsule().fill(color.opacity(0.15)))
            .foregroundStyle(color)
    }
}
