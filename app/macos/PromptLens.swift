import SwiftUI
import AppKit

struct Signal: Identifiable {
    let id: String
    let value: Double
}

@MainActor
final class InferenceBridge: ObservableObject {
    @Published var ready = false
    @Published var busy = false
    @Published var fatal = false
    @Published var errorCode: String? = nil
    @Published var details: [String: Any] = [:]
    @Published var signals: [Signal] = []
    @Published var elapsed = 0
    @Published var tokens = 0
    private var process: Process?
    private var input: Pipe?
    private var output: Pipe?
    private var buffer = Data()
    private var ticket = UUID().uuidString
    private var intentionalStop = false

    func start() {
        guard process == nil else { return }
        ready = false; fatal = false; errorCode = nil; intentionalStop = false
        let root = Bundle.main.bundleURL.deletingLastPathComponent()
        let worker = Process()
        worker.executableURL = root.appendingPathComponent(".venv/bin/python")
        worker.arguments = ["-u", root.appendingPathComponent("scripts/predict_worker.py").path]
        worker.currentDirectoryURL = root
        var env = ProcessInfo.processInfo.environment
        env["HF_HUB_OFFLINE"] = "1"; env["TRANSFORMERS_OFFLINE"] = "1"
        worker.environment = env
        let incoming = Pipe(); let outgoing = Pipe()
        worker.standardInput = incoming; worker.standardOutput = outgoing
        worker.standardError = FileHandle.nullDevice
        input = incoming; output = outgoing; process = worker
        outgoing.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let bytes = handle.availableData
            guard !bytes.isEmpty else { return }
            Task { @MainActor in self?.consume(bytes) }
        }
        worker.terminationHandler = { [weak self, weak worker] _ in
            Task { @MainActor in
                guard let self = self, self.process === worker, !self.intentionalStop else { return }
                self.ready = false; self.busy = false; self.fatal = true
                self.errorCode = "model_load_failed"
            }
        }
        do { try worker.run() }
        catch { ready = false; fatal = true; errorCode = "model_load_failed"; process = nil }
    }

    private func consume(_ bytes: Data) {
        buffer.append(bytes)
        while let newline = buffer.firstIndex(of: 10) {
            let line = buffer.prefix(upTo: newline)
            buffer.removeSubrange(...newline)
            guard let object = try? JSONSerialization.jsonObject(with: line) as? [String: Any],
                  let type = object["type"] as? String else { continue }
            if type == "ready" { ready = true; fatal = false; errorCode = nil }
            if type == "result" {
                guard object["id"] as? String == ticket else { continue }
                signals = (object["signals"] as? [[String: Any]] ?? []).compactMap {
                    guard let id = $0["id"] as? String, let value = $0["value"] as? Double else { return nil }
                    return Signal(id: id, value: value)
                }
                elapsed = object["elapsed_ms"] as? Int ?? 0
                tokens = object["tokens"] as? Int ?? 0
                busy = false; errorCode = nil
            }
            if type == "error" || type == "fatal" {
                busy = false; details = object; errorCode = object["code"] as? String ?? "inference_failed"
                if type == "fatal" { fatal = true; ready = false }
            }
        }
    }

    func analyze(request: String, context: String) {
        guard ready && !busy else { return }
        signals = []; errorCode = nil
        guard !request.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            errorCode = "empty_request"; return
        }
        ticket = UUID().uuidString
        let payload: [String: Any] = ["id": ticket, "request": request, "context": context]
        do {
            var data = try JSONSerialization.data(withJSONObject: payload)
            data.append(10)
            try input?.fileHandleForWriting.write(contentsOf: data)
            busy = true
        } catch { errorCode = "inference_failed"; busy = false }
    }

    func clearResults() { signals = []; errorCode = nil }
    func stop() {
        intentionalStop = true
        output?.fileHandleForReading.readabilityHandler = nil
        try? input?.fileHandleForWriting.close()
        if process?.isRunning == true { process?.terminate() }
        process = nil; ready = false
    }
    func retry() { stop(); buffer = Data(); start() }
}

struct ContentView: View {
    @StateObject private var bridge = InferenceBridge()
    @State private var request = ""
    @State private var context = ""
    @State private var chinese = false
    @State private var showContext = false
    @State private var showGuide = false
    private let teal = Color(red: 0.10, green: 0.39, blue: 0.35)
    private let muted = Color(red: 0.40, green: 0.44, blue: 0.45)
    private let canvas = Color(red: 0.965, green: 0.969, blue: 0.957)

    private func t(_ en: String, _ zh: String) -> String { chinese ? zh : en }
    private var disabled: Bool { !bridge.ready || bridge.busy }
    private var labelMap: [String: (String, String, String, String)] {
        ["unclear_goal": ("Unclear goal", "目标不明确", "What outcome do you want?", "你希望得到什么结果？"),
         "missing_information": ("Missing information", "缺少必要信息", "Which details are essential to complete this task?", "完成任务还需要哪些关键信息？"),
         "unresolved_reference": ("Ambiguous reference", "指代不清", "Is it clear which person, file or item you mean?", "涉及的人、文件或对象是否明确？"),
         "conflicting_constraints": ("Conflicting requirements", "要求相互冲突", "Can all your requirements be satisfied together?", "所有要求能否同时满足？"),
         "unclear_deliverable": ("Unclear deliverable", "交付形式不明确", "What should the finished result look like?", "最终应该交付什么形式的结果？")]
    }
    private var errorText: String {
        switch bridge.errorCode {
        case "empty_request": return t("Write a request first.", "请先输入你的需求。")
        case "too_many_tokens":
            let count = bridge.details["tokens"] as? Int ?? 0
            return t("Your request and context use \(count) tokens. Please shorten them to fit the 256-token limit. Nothing was cut off.", "需求与背景共 \(count) 个词元，请缩短至 256 个词元以内。内容未被截断。")
        case "too_many_characters": return t("This input is too long. Please shorten your request and context.", "内容过长，请缩短需求和背景。")
        case "model_load_failed": return t("The local model could not be opened. Keep this app in its project folder and try again.", "无法打开本地模型。请将应用保留在项目文件夹内，然后重试。")
        default: return t("The analysis could not finish. Your text is still here; please try again.", "分析未能完成。你的输入仍在，可以重试。")
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                Image(systemName: "viewfinder").font(.system(size: 22, weight: .medium)).foregroundStyle(teal)
                    .frame(width: 42, height: 42).background(teal.opacity(0.09), in: RoundedRectangle(cornerRadius: 13))
                VStack(alignment: .leading, spacing: 3) {
                    Text("Prompt Lens").font(.system(size: 20, weight: .semibold, design: .rounded))
                    Text(t("A little clarity before you ask.", "让每一次提问更清晰。")).font(.system(size: 12)).foregroundStyle(muted)
                }
                Spacer()
                Picker("Language", selection: $chinese) {
                    Text("EN").tag(false); Text("中文").tag(true)
                }.pickerStyle(.segmented).labelsHidden().frame(width: 100).accessibilityLabel("Interface language / 界面语言")
            }.padding(.horizontal, 24).padding(.vertical, 18)
            Divider().opacity(0.5)
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    HStack {
                        Text(t("YOUR REQUEST", "你的需求")).font(.system(size: 10, weight: .semibold)).tracking(1.7).foregroundStyle(muted)
                        Spacer()
                        HStack(spacing: 5) {
                            Circle().fill(bridge.ready ? teal : Color.orange).frame(width: 5, height: 5)
                            Text(bridge.ready ? t("On-device", "本地运行") : bridge.fatal ? t("Unavailable", "暂不可用") : t("Loading model…", "正在加载模型…"))
                                .font(.system(size: 11)).foregroundStyle(muted)
                        }
                    }
                    VStack(alignment: .leading, spacing: 8) {
                        ZStack(alignment: .topLeading) {
                            if request.isEmpty {
                                Text(t("What would you like the AI to do?", "你希望 AI 帮你完成什么？"))
                                    .foregroundStyle(muted.opacity(0.65)).padding(.top, 9).padding(.leading, 5).allowsHitTesting(false)
                            }
                            TextEditor(text: $request).font(.system(size: 14)).scrollContentBackground(.hidden)
                                .frame(minHeight: 108, maxHeight: 140).accessibilityLabel(t("Your request", "你的需求"))
                                .disabled(bridge.busy)
                        }
                        HStack {
                            Text(t("English or Chinese", "支持中文与英文")).font(.system(size: 11)).foregroundStyle(muted)
                            Spacer()
                            Text("\(request.count)").monospacedDigit().font(.system(size: 11)).foregroundStyle(muted)
                        }
                    }.padding(13).background(Color.white, in: RoundedRectangle(cornerRadius: 13))
                     .overlay(RoundedRectangle(cornerRadius: 13).stroke(Color.black.opacity(0.08)))
                    DisclosureGroup(isExpanded: $showContext) {
                        TextEditor(text: $context).font(.system(size: 13)).scrollContentBackground(.hidden)
                            .frame(height: 75).padding(8).background(Color.white, in: RoundedRectangle(cornerRadius: 9))
                            .accessibilityLabel(t("Context", "背景信息")).disabled(bridge.busy).padding(.top, 8)
                    } label: {
                        Text(t("Add context", "补充背景") + "  " + t("· optional", "· 选填")).font(.system(size: 12)).foregroundStyle(muted)
                    }.tint(teal)
                    HStack {
                        Button(action: { bridge.analyze(request: request, context: context) }) {
                            HStack(spacing: 8) {
                                if bridge.busy { ProgressView().controlSize(.small).tint(.white) }
                                else { Image(systemName: "sparkle.magnifyingglass") }
                                Text(bridge.busy ? t("Reading your request…", "正在分析需求…") : t("Analyze request", "分析需求")).fontWeight(.semibold)
                            }.frame(maxWidth: .infinity).padding(.vertical, 11)
                        }.buttonStyle(.plain).foregroundStyle(.white).background(disabled ? teal.opacity(0.45) : teal, in: RoundedRectangle(cornerRadius: 10))
                         .disabled(disabled).keyboardShortcut(.return, modifiers: [.command])
                        Button(t("Clear", "清空")) { request = ""; context = ""; bridge.clearResults() }
                            .buttonStyle(.plain).foregroundStyle(muted).padding(.horizontal, 12).disabled(bridge.busy)
                    }
                    if bridge.errorCode != nil {
                        VStack(alignment: .leading, spacing: 8) {
                            Label(errorText, systemImage: "exclamationmark.circle").font(.system(size: 12)).fixedSize(horizontal: false, vertical: true)
                            if bridge.fatal { Button(t("Try again", "重试")) { bridge.retry() } }
                        }.foregroundStyle(Color(red: 0.55, green: 0.28, blue: 0.11)).padding(12)
                         .frame(maxWidth: .infinity, alignment: .leading).background(Color.orange.opacity(0.09), in: RoundedRectangle(cornerRadius: 10))
                    }
                    if !bridge.signals.isEmpty {
                        VStack(alignment: .leading, spacing: 14) {
                            HStack {
                                Text(t("Model signals", "模型信号")).font(.system(size: 16, weight: .semibold))
                                Spacer()
                                Text(t("Experimental", "实验版")).font(.system(size: 10, weight: .medium)).foregroundStyle(teal)
                                    .padding(.horizontal, 8).padding(.vertical, 4).background(teal.opacity(0.08), in: Capsule())
                            }
                            Text(t("Higher = stronger issue signal, not a measured error probability.", "越高表示该问题的模型信号越强，不代表实际出错概率。"))
                                .font(.system(size: 11)).foregroundStyle(muted).fixedSize(horizontal: false, vertical: true)
                            ForEach(bridge.signals) { signal in
                                let copy = labelMap[signal.id]!
                                VStack(spacing: 6) {
                                    HStack {
                                        Text(t(copy.0, copy.1)).font(.system(size: 12))
                                        Spacer()
                                        Text(String(format: "%.3f", signal.value)).font(.system(size: 12, design: .monospaced)).foregroundStyle(teal)
                                    }
                                    GeometryReader { geo in
                                        ZStack(alignment: .leading) {
                                            Capsule().fill(teal.opacity(0.07))
                                            Capsule().fill(teal.opacity(0.7)).frame(width: max(3, geo.size.width * signal.value))
                                        }
                                    }.frame(height: 5).accessibilityLabel(t(copy.0, copy.1) + String(format: " %.3f", signal.value))
                                }
                            }
                            Divider().opacity(0.4)
                            DisclosureGroup(isExpanded: $showGuide) {
                                VStack(alignment: .leading, spacing: 9) {
                                    Text(t("General prompts for self-review, not model-generated findings.", "以下是通用自查问题，并非模型生成的诊断。")).font(.system(size: 11)).foregroundStyle(muted)
                                    ForEach(bridge.signals) { signal in
                                        let copy = labelMap[signal.id]!
                                        Text("• " + t(copy.2, copy.3)).font(.system(size: 12)).fixedSize(horizontal: false, vertical: true)
                                    }
                                }.padding(.top, 10)
                            } label: {
                                Text(t("Questions to sharpen your request", "帮助你完善需求的自查问题")).font(.system(size: 12, weight: .medium))
                            }.tint(teal)
                            Text(t("\(bridge.tokens) tokens · \(bridge.elapsed) ms", "\(bridge.tokens) 个词元 · \(bridge.elapsed) 毫秒")).font(.system(size: 10)).foregroundStyle(muted)
                        }.padding(18).background(Color.white, in: RoundedRectangle(cornerRadius: 13))
                    } else if bridge.errorCode == nil && !bridge.busy {
                        HStack(alignment: .top, spacing: 11) {
                            Image(systemName: "text.bubble").font(.system(size: 20)).foregroundStyle(teal.opacity(0.7))
                            VStack(alignment: .leading, spacing: 5) {
                                Text(t("Your results will appear here", "分析结果将显示在这里")).font(.system(size: 13, weight: .medium))
                                Text(t("Review clarity, essential details, references, constraints and the expected result.", "查看目标、必要信息、指代、约束与交付形式五个维度。"))
                                    .font(.system(size: 12)).foregroundStyle(muted).fixedSize(horizontal: false, vertical: true)
                            }
                        }.padding(18).frame(maxWidth: .infinity, alignment: .leading)
                         .background(teal.opacity(0.04), in: RoundedRectangle(cornerRadius: 13))
                    }
                    Text(t("Experimental model trained on machine-labeled examples. Signals are not a validated quality rating; use your judgment.", "当前实验模型使用机器标注示例训练。信号尚未验证为可靠的质量评分，请结合自己的判断。"))
                        .font(.system(size: 11)).foregroundStyle(muted).fixedSize(horizontal: false, vertical: true)
                }.padding(24)
            }
            Divider().opacity(0.4)
            HStack {
                Label(t("Private · stays on your Mac", "隐私保护 · 内容留在本机"), systemImage: "lock.shield").font(.system(size: 10))
                Spacer()
                Text("EN / 中文 · 0.278B").font(.system(size: 10, design: .monospaced))
            }.foregroundStyle(muted).padding(.horizontal, 24).padding(.vertical, 12)
        }.background(canvas).frame(minWidth: 480, idealWidth: 570, minHeight: 580, idealHeight: 800)
         .preferredColorScheme(.light)
         .onAppear { bridge.start() }
         .onChange(of: request) { bridge.clearResults() }
         .onChange(of: context) { bridge.clearResults() }
         .onReceive(NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)) { _ in bridge.stop() }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
    func applicationDidFinishLaunching(_ notification: Notification) { NSApp.activate(ignoringOtherApps: true) }
}

@main
struct PromptLensApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    var body: some Scene {
        Window("Prompt Lens", id: "main") { ContentView() }
            .defaultSize(width: 570, height: 800)
            .windowResizability(.contentMinSize)
            .commands { CommandGroup(replacing: .newItem) {} }
    }
}
