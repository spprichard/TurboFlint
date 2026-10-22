import ArgumentParser
import Foundation
import TurboFieldfare
import TurboFieldfareFormat

@main
struct TurboFlintCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "turboflint",
        abstract: "Run Recursive Flint with routed experts streamed from SSD.")

    @Option(help: "Ollama manifest for the Flint checkpoint.")
    var manifest: String = FlintOllamaCheckpoint.defaultManifest.path

    @Option(help: "Raw prompt text (no chat template).")
    var prompt: String?

    @Option(help: "User message, wrapped in Flint's chat template.")
    var chat: String?

    @Option(name: .customLong("prompt-ids"), help: "Comma-separated prompt token ids.")
    var promptIDs: String?

    @Option(help: "Tokens to generate (greedy).")
    var maxNew = 32

    @Option(help: "Routed-expert cache slots per MoE layer (1.77 MB each).")
    var slots = 32

    @Option(help: "Maximum context length (KV cache rows for full-attention layers).")
    var maxContext = 8192

    @Option(help: "Prompt tokens per prefill chunk (1 = token-by-token).")
    var prefillChunk = 256

    @Option(help: "Parallel pread lanes for expert loads.")
    var readConcurrency = 8

    @Flag(help: "Disable next-layer expert prefetch.")
    var noPrefetch = false

    @Flag(help: "Read experts with F_NOCACHE (no OS page-cache copy).")
    var noPageCache = false

    @Flag(help: "Print generated ids as JSON on the last line (for oracle comparison).")
    var json = false

    @Option(help: "Write per-layer residual stream of the last prompt token to this JSON file (debug).")
    var dumpHidden: String?

    @Option(help: "Debug: JSON {routing: {layer: {ids, weights}}} to force expert choices during prefill.")
    var forceRouting: String?

    @Flag(help: "Measure next-layer expert prediction accuracy (debug).")
    var measurePrediction = false

    @Flag(help: "Time individual kernels and exit.")
    var profileKernels = false

    func run() async throws {
        let checkpoint = try FlintOllamaCheckpoint(manifestURL: URL(fileURLWithPath: manifest))
        let tokenizer = try await FlintTokenizer.load(checkpoint: checkpoint)
        let ids: [Int]
        if let promptIDs {
            ids = promptIDs.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        } else if let chat {
            ids = tokenizer.encode(FlintTokenizer.chatPrompt(user: chat))
        } else if let prompt {
            ids = tokenizer.encode(prompt)
        } else {
            throw ValidationError("pass --prompt, --chat, or --prompt-ids")
        }
        guard !ids.isEmpty else { throw ValidationError("empty prompt") }

        var options = FlintRunnerOptions()
        options.maxContext = maxContext
        options.slotsPerLayer = slots
        options.readConcurrency = readConcurrency
        options.prefillChunk = prefillChunk
        options.bypassPageCache = noPageCache
        options.prefetchNextLayer = !noPrefetch
        let loadStart = Date()
        let context = try MetalContext()
        let runner = try FlintRunner(checkpoint: checkpoint, context: context, options: options)
        log(String(format: "loaded in %.1fs, resident %.2f GB, expert cache %.2f GB reserved",
                   Date().timeIntervalSince(loadStart), Double(runner.residentBytes) / 1e9,
                   Double(runner.expertCacheBytes) / 1e9))
        if profileKernels {
            for (name, ms) in try runner.profileKernels() { log(String(format: "%-28@ %8.3f ms", name as NSString, ms)) }
            return
        }
        runner.measurePrediction = measurePrediction
        if let forceRouting {
            struct Dump: Decodable {
                struct Layer: Decodable { let ids: [[Int]]; let weights: [[Float]] }
                let routing: [String: Layer]
            }
            let dump = try JSONDecoder().decode(Dump.self, from: Data(contentsOf: URL(fileURLWithPath: forceRouting)))
            var table: [Int: [(ids: [Int], weights: [Float])]] = [:]
            for (layer, v) in dump.routing { table[Int(layer)!] = zip(v.ids, v.weights).map { ($0, $1) } }
            runner.forcedRouting = table
        }
        log("prompt: \(ids.count) tokens")

        let eos = Set(checkpoint.config.eosTokenID)
        let prefillStart = Date()
        var next = 0
        if dumpHidden != nil {
            // Token-by-token so the per-layer residual of the last prompt row can be captured.
            for (pos, t) in ids.enumerated() {
                let last = pos == ids.count - 1
                if last { runner.debugHidden = runner.makeDebugHiddenBuffer() }
                if let n = try runner.step(token: t, position: pos, wantLogits: last) { next = n }
            }
        } else {
            next = try runner.prefill(ids)
        }
        if let dumpHidden, let buf = runner.debugHidden {
            let c = checkpoint.config
            let p = buf.contents().assumingMemoryBound(to: Float.self)
            let rows = (0..<c.numHiddenLayers).map { l in (0..<c.hiddenSize).map { Double(p[l * c.hiddenSize + $0]) } }
            let logits = runner.lastLogits().map { $0.isFinite ? Double($0) : -1e30 }
            let obj: [String: Any] = ["hidden": rows.map { $0.map { $0.isFinite ? $0 : 1e30 } },
                                      "logits_top": Array(logits.enumerated().sorted { $0.element > $1.element }.prefix(5).map { [$0.offset, $0.element] })]
            try JSONSerialization.data(withJSONObject: obj).write(to: URL(fileURLWithPath: dumpHidden))
            runner.debugHidden = nil
        }
        let prefill = Date().timeIntervalSince(prefillStart)
        let firstTop = runner.lastLogits().enumerated().sorted { $0.element > $1.element }.prefix(5)
            .map { [Double($0.offset), Double($0.element)] }
        let afterPrefill = runner.expertStats
        log(String(format: "prefill %.2fs (%.1f tok/s)", prefill, Double(ids.count) / prefill))

        var generated: [Int] = []
        var gpu = 0.0, io = 0.0
        var gA = 0.0, gS = 0.0, gR = 0.0, gH = 0.0
        let decodeStart = Date()
        while true {
            generated.append(next)
            FileHandle.standardOutput.write(tokenizer.decode([next]).data(using: .utf8)!)
            if eos.contains(next) || generated.count >= maxNew { break }
            next = try runner.step(token: next, position: ids.count + generated.count - 1)!
            gpu += runner.lastTiming.gpuWaitSeconds
            io += runner.lastTiming.ioSeconds
            gA += runner.lastTiming.gpuAttention; gS += runner.lastTiming.gpuShared
            gR += runner.lastTiming.gpuRouted; gH += runner.lastTiming.gpuHead
        }
        let decode = Date().timeIntervalSince(decodeStart)
        let steps = max(1, generated.count - 1)
        let s = runner.expertStats
        let hits = s.hits - afterPrefill.hits, misses = s.misses - afterPrefill.misses
        print("")
        log(String(format: "decode %d tokens in %.2fs: %.2f tok/s (gpu wait %.1f ms/tok, expert io %.1f ms/tok)",
                   generated.count, decode, Double(steps) / decode, gpu / Double(steps) * 1000,
                   io / Double(steps) * 1000))
        log(String(format: "gpu busy ms/tok: attention+router %.1f, shared %.1f, routed %.1f, head %.1f",
                   gA / Double(steps) * 1000, gS / Double(steps) * 1000, gR / Double(steps) * 1000,
                   gH / Double(steps) * 1000))
        let pre = s.prefetchHits - afterPrefill.prefetchHits
        let total = max(1, hits + misses + pre)
        log(String(format: "decode experts: %.1f%% cached, %.1f%% prefetched, %.1f%% read on demand; %.0f MB read/tok, %.1f ms/tok blocked; %d prefetches wasted",
                   100 * Double(hits) / Double(total), 100 * Double(pre) / Double(total),
                   100 * Double(misses) / Double(total),
                   Double(s.bytesRead - afterPrefill.bytesRead) / Double(steps) / 1e6,
                   (s.waitSeconds - afterPrefill.waitSeconds) / Double(steps) * 1000,
                   s.prefetchWasted - afterPrefill.prefetchWasted))
        if measurePrediction {
            log(String(format: "next-layer prediction: %.1f%% of experts guessed",
                       100 * Double(runner.predictionOverlap) / Double(max(1, runner.predictionTotal))))
        }
        log(String(format: "peak footprint %.2f GB", Double(peakFootprint()) / 1e9))
        if json {
            let out: [String: Any] = ["prompt_ids": ids, "generated_ids": generated, "first_top": firstTop]
            print(String(data: try JSONSerialization.data(withJSONObject: out), encoding: .utf8)!)
        }
    }

    private func log(_ s: String) {
        FileHandle.standardError.write((s + "\n").data(using: .utf8)!)
    }

    private func peakFootprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? UInt64(max(0, info.ledger_phys_footprint_peak)) : 0
    }
}
