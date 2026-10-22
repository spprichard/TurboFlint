import Foundation
import TurboFieldfare
import TurboFieldfareFormat

/// OpenAI-compatible backend for Recursive Flint on the streamed-expert runtime.
///
/// - Prompts are rendered with Flint's chat template (recursive-oss/rec `chat_template.jinja`).
/// - The model's `<think>…</think>` reasoning is hidden from the response; `<tool_call>` blocks
///   become OpenAI tool calls.
/// - KV state is reused across requests when the new prompt shares a prefix with the tokens
///   already in the cache and the sliding-window rings still hold the positions it needs.
///
/// Generation blocks the actor's thread for the length of a request; the server runs one request
/// at a time, and expert reads happen on their own I/O threads.
public actor FlintServerBackend: ServerInferenceBackend {
    private let runner: FlintRunner
    private let tokenizer: FlintTokenizer
    private let eos: Set<Int>
    /// Tokens whose K/V currently sit in the cache at positions 0..<cachedIDs.count.
    private var cachedIDs: [Int] = []

    public static func load(manifestPath: String?, maxContext: Int, slotsPerLayer: Int,
                            readConcurrency: Int = 8) async throws -> FlintServerBackend {
        let manifest = manifestPath.map { URL(fileURLWithPath: $0) } ?? FlintOllamaCheckpoint.defaultManifest
        let checkpoint = try FlintOllamaCheckpoint(manifestURL: manifest)
        let tokenizer = try await FlintTokenizer.load(checkpoint: checkpoint)
        var options = FlintRunnerOptions()
        options.maxContext = maxContext
        options.slotsPerLayer = slotsPerLayer
        options.readConcurrency = readConcurrency
        options.rewindSlack = 4096
        let runner = try FlintRunner(checkpoint: checkpoint, context: try MetalContext(), options: options)
        return FlintServerBackend(runner: runner, tokenizer: tokenizer, eos: Set(checkpoint.config.eosTokenID))
    }

    private init(runner: FlintRunner, tokenizer: FlintTokenizer, eos: Set<Int>) {
        self.runner = runner
        self.tokenizer = tokenizer
        self.eos = eos
    }

    public func generate(_ request: ValidatedChatRequest,
                         onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void) async throws -> ServerCompletion {
        guard request.imageFiles.isEmpty, request.multimodalMessages == nil else {
            throw ServerRequestError.invalid(message: "image support is unavailable",
                                             param: "messages", code: "vision_unavailable")
        }
        let prompt = FlintChatFormat.render(messages: request.messages, tools: request.tools)
        let ids = tokenizer.encode(prompt)
        let budget = runner.options.maxContext - ids.count
        guard budget > 0 else {
            throw ServerRequestError.invalid(message: "prompt is \(ids.count) tokens; context is \(runner.options.maxContext)",
                                             param: "messages", code: "context_length_exceeded")
        }
        let maxNew = min(request.maximumCompletionTokens, budget)

        // Reuse the cached prefix when every position it needs is still intact.
        var reuse = zip(cachedIDs, ids).prefix { $0 == $1 }.count
        if reuse == ids.count { reuse -= 1 }  // always run at least one token to get logits
        if cachedIDs.count - reuse > runner.maxRewind { reuse = 0 }
        cachedIDs = Array(ids[..<reuse])

        let prefillStart = Date()
        var next = try runner.prefill(Array(ids[reuse...]), startPosition: reuse)
        cachedIDs = ids
        let prefillSeconds = Date().timeIntervalSince(prefillStart)

        var sampler = FlintSampler(config: request.generationConfig)
        var parser = FlintOutputParser(tools: request.tools)
        var stop = StreamingStopMatcher(stops: request.generationConfig.stopStrings)
        var generated: [Int] = []
        var emittedText = ""
        var content = ""
        var finish = "stop"
        let decodeStart = Date()

        func emit(_ events: [FlintOutputParser.Event]) {
            for event in events {
                switch event {
                case let .content(text):
                    let visible = stop.push(text)
                    if !visible.isEmpty { content += visible; onEvent(.content(visible)) }
                case let .toolCall(call):
                    onEvent(.toolCall(call))
                }
            }
        }

        while true {
            if !sampler.isGreedy { next = sampler.sample(runner.lastLogits()) }
            if eos.contains(next) { break }
            generated.append(next)
            // Decode the whole completion and emit the new suffix; hold back a trailing
            // replacement character, which marks an incomplete multi-byte sequence.
            let text = tokenizer.decode(generated)
            if !text.hasSuffix("\u{FFFD}"), text.count > emittedText.count, text.hasPrefix(emittedText) {
                emit(parser.push(String(text.dropFirst(emittedText.count))))
                emittedText = text
            }
            if stop.isStopped { break }
            if generated.count >= maxNew { finish = "length"; break }
            if Task.isCancelled { throw CancellationError() }
            next = try runner.step(token: next, position: ids.count + generated.count - 1)!
            cachedIDs.append(generated.last!)
        }
        emit(parser.finish())
        let tail = stop.finish()
        if !tail.isEmpty { content += tail; onEvent(.content(tail)) }
        if !parser.toolCalls.isEmpty { finish = "tool_calls" }

        return ServerCompletion(
            content: content, toolCalls: parser.toolCalls, finishReason: finish,
            usage: OpenAIUsage(promptTokens: ids.count, completionTokens: generated.count,
                               totalTokens: ids.count + generated.count, cachedTokens: reuse),
            prefillSeconds: prefillSeconds, decodeSeconds: Date().timeIntervalSince(decodeStart))
    }
}

/// Flint's chat format, matching recursive-oss/rec `engine/src/rec_mlx/chat_template.jinja`.
enum FlintChatFormat {
    static func render(messages: [GFTokenizer.Message], tools: [GFTokenizer.FunctionDefinition]) -> String {
        var preamble = ""
        if !tools.isEmpty {
            preamble = "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                + "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
            for tool in tools {
                let fn: JSONValue = .object(["name": .string(tool.name), "description": .string(tool.description),
                                             "parameters": tool.parameters])
                preamble += "\n" + pythonJSON(fn, keyOrder: ["name", "description", "parameters"])
            }
            preamble += "\n</tools>\n\nFor each function call, output the function name and arguments within the "
                + "following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key>"
                + "<arg_value>{arg-value-1}</arg_value>...</tool_call>\n"
        }
        var out = ""
        var toolsPending = true
        for m in messages {
            if toolsPending && m.role != .system {
                out += preamble
                toolsPending = false
            }
            switch m.role {
            case .system: out += "<|system|>" + (m.content ?? "")
            case .developer: out += "<|developer|>" + (m.content ?? "")
            case .user: out += "<|user|>" + (m.content ?? "")
            case .assistant:
                out += "<|assistant|>" + (m.content ?? "")
                for call in m.toolCalls {
                    out += "<tool_call>" + call.name
                    if case let .object(args) = call.arguments {
                        for key in args.keys.sorted() {
                            let value = args[key]!
                            let rendered: String
                            if case let .string(s) = value { rendered = s } else { rendered = pythonJSON(value) }
                            out += "<arg_key>\(key)</arg_key><arg_value>\(rendered)</arg_value>"
                        }
                    }
                    out += "</tool_call>"
                }
            case .tool: out += "<|observation|><tool_response>" + (m.content ?? "") + "</tool_response>"
            }
        }
        if toolsPending { out += preamble }
        return out + "<|assistant|>"
    }

    /// `json.dumps(..., ensure_ascii=False)` formatting (", " and ": " separators), as the template's
    /// `tojson` produces. Object keys follow `keyOrder` first, then sorted order.
    static func pythonJSON(_ v: JSONValue, keyOrder: [String] = []) -> String {
        switch v {
        case let .object(o):
            let keys = keyOrder.filter { o[$0] != nil } + o.keys.filter { !keyOrder.contains($0) }.sorted()
            return "{" + keys.map { quote($0) + ": " + pythonJSON(o[$0]!) }.joined(separator: ", ") + "}"
        case let .array(a): return "[" + a.map { pythonJSON($0) }.joined(separator: ", ") + "]"
        case let .string(s): return quote(s)
        case let .integer(i): return String(i)
        case let .unsignedInteger(u): return String(u)
        case let .decimal(d): return "\(d)"
        case let .number(n): return n == n.rounded() && abs(n) < 1e15 ? String(format: "%.1f", n) : String(n)
        case let .bool(b): return b ? "true" : "false"
        case .null: return "null"
        }
    }

    private static func quote(_ s: String) -> String {
        var out = "\""
        for ch in s.unicodeScalars {
            switch ch {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if ch.value < 0x20 { out += String(format: "\\u%04x", ch.value) } else { out.unicodeScalars.append(ch) }
            }
        }
        return out + "\""
    }
}

/// Streams Flint's raw completion into visible content and tool calls. Reasoning inside
/// `<think>…</think>` is dropped; `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>…</tool_call>`
/// becomes a `ParsedToolCall`, with values typed by the tool's JSON schema where possible.
struct FlintOutputParser {
    enum Event { case content(String), toolCall(ParsedToolCall) }

    private enum State { case start, thinking, content, tool }
    private var state = State.start
    private var buffer = ""
    private var leadingTrim = true
    private let schemas: [String: JSONValue]
    private(set) var toolCalls: [ParsedToolCall] = []

    init(tools: [GFTokenizer.FunctionDefinition]) {
        schemas = Dictionary(tools.map { ($0.name, $0.parameters) }, uniquingKeysWith: { a, _ in a })
    }

    mutating func push(_ text: String) -> [Event] {
        buffer += text
        var events: [Event] = []
        while true {
            switch state {
            case .start:
                if buffer.hasPrefix("<think>") { buffer.removeFirst(7); state = .thinking; continue }
                if "<think>".hasPrefix(buffer) { return events }  // could still become <think>
                state = .content
            case .thinking:
                guard let r = buffer.range(of: "</think>") else { return events }
                buffer = String(buffer[r.upperBound...])
                state = .content
                leadingTrim = true
            case .content:
                if leadingTrim {
                    buffer = String(buffer.drop { $0.isWhitespace })
                    if buffer.isEmpty { return events }
                    leadingTrim = false
                }
                if let r = buffer.range(of: "<tool_call>") {
                    let before = String(buffer[..<r.lowerBound])
                    if !before.isEmpty { events.append(.content(before)) }
                    buffer = String(buffer[r.upperBound...])
                    state = .tool
                    continue
                }
                // Hold back a suffix that might be the start of "<tool_call>".
                let hold = (1..<min(buffer.count + 1, 11)).reversed().first { "<tool_call>".hasPrefix(buffer.suffix($0)) } ?? 0
                let emitCount = buffer.count - hold
                if emitCount > 0 {
                    events.append(.content(String(buffer.prefix(emitCount))))
                    buffer = String(buffer.suffix(hold))
                }
                return events
            case .tool:
                guard let r = buffer.range(of: "</tool_call>") else { return events }
                let body = String(buffer[..<r.lowerBound])
                buffer = String(buffer[r.upperBound...])
                state = .content
                leadingTrim = true
                if let call = parseCall(body) {
                    toolCalls.append(call)
                    events.append(.toolCall(call))
                }
            }
        }
    }

    mutating func finish() -> [Event] {
        defer { buffer = "" }
        switch state {
        case .content where !buffer.isEmpty: return [.content(buffer)]
        case .start where !buffer.isEmpty: return [.content(buffer)]
        default: return []
        }
    }

    private func parseCall(_ body: String) -> ParsedToolCall? {
        let parts = body.components(separatedBy: "<arg_key>")
        let name = parts[0].trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return nil }
        var props: [String: JSONValue] = [:]
        if case let .object(schema)? = schemas[name], case let .object(p)? = schema["properties"] { props = p }
        var args: [String: JSONValue] = [:]
        for part in parts.dropFirst() {
            guard let k = part.range(of: "</arg_key>"), let v0 = part.range(of: "<arg_value>"),
                  let v1 = part.range(of: "</arg_value>", options: .backwards) else { return nil }
            let key = String(part[..<k.lowerBound])
            let raw = String(part[v0.upperBound..<v1.lowerBound])
            var declaredString = false
            if case let .object(prop)? = props[key], case let .string(t)? = prop["type"] { declaredString = t == "string" }
            if !declaredString, let data = raw.data(using: .utf8),
               let parsed = try? JSONDecoder().decode(JSONValue.self, from: data) {
                args[key] = parsed
            } else {
                args[key] = .string(raw)
            }
        }
        let value = JSONValue.object(args)
        let json = (try? value.encoded(sortedKeys: true)) ?? "{}"
        return ParsedToolCall(id: "call_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24).lowercased(),
                              name: name, arguments: value, argumentsJSON: json)
    }
}

/// Temperature / top-k / top-p sampling over the head's logits on the CPU.
struct FlintSampler {
    private let temperature: Float
    private let topK: Int
    private let topP: Float
    private var rng: UInt64

    init(config: GenerationConfig) {
        temperature = config.temperature
        topK = min(max(config.topK ?? 64, 1), 1024)
        topP = config.topP ?? 1
        rng = config.seed ?? UInt64.random(in: .min ... .max)
    }

    var isGreedy: Bool { temperature <= 0 || topK == 1 }

    private mutating func nextUniform() -> Float {
        rng &+= 0x9E37_79B9_7F4A_7C15
        var z = rng
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z ^= z >> 31
        return Float(z >> 40) / Float(1 << 24)
    }

    mutating func sample(_ logits: [Float]) -> Int {
        // Keep the top-k candidates (k is small; a sorted insertion buffer is enough).
        var cand: [(Int, Float)] = []
        cand.reserveCapacity(topK + 1)
        for (i, l) in logits.enumerated() where cand.count < topK || l > cand[cand.count - 1].1 {
            let pos = cand.firstIndex { $0.1 < l } ?? cand.count
            cand.insert((i, l), at: pos)
            if cand.count > topK { cand.removeLast() }
        }
        let m = cand[0].1
        var probs = cand.map { exp(($0.1 - m) / temperature) }
        let total = probs.reduce(0, +)
        probs = probs.map { $0 / total }
        var cut = probs.count
        var acc: Float = 0
        for (i, p) in probs.enumerated() {
            acc += p
            if acc >= topP { cut = i + 1; break }
        }
        let kept = probs.prefix(cut).reduce(0, +)
        var r = nextUniform() * kept
        for i in 0..<cut {
            r -= probs[i]
            if r <= 0 { return cand[i].0 }
        }
        return cand[cut - 1].0
    }
}
