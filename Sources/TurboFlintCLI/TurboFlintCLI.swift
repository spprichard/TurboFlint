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

    @Option(help: "Comma-separated prompt token ids.")
    var promptIDs: String?

    @Option(help: "Tokens to generate (greedy).")
    var maxNew = 32

    @Option(help: "Routed-expert cache slots per MoE layer (1.77 MB each).")
    var slots = 32

    @Option(help: "Maximum context length (KV cache rows for full-attention layers).")
    var maxContext = 8192

    @Option(help: "Parallel pread lanes for expert loads.")
    var readConcurrency = 8

    @Flag(help: "Print generated ids as JSON on the last line (for oracle comparison).")
    var json = false

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
        let loadStart = Date()
        let context = try MetalContext()
        let runner = try FlintRunner(checkpoint: checkpoint, context: context, options: options)
        log(String(format: "loaded in %.1fs, resident %.2f GB, expert cache %.2f GB reserved",
                   Date().timeIntervalSince(loadStart), Double(runner.residentBytes) / 1e9,
                   Double(runner.expertCacheBytes) / 1e9))
        log("prompt: \(ids.count) tokens")

        let eos = Set(checkpoint.config.eosTokenID)
        let prefillStart = Date()
        var next = 0
        for (pos, t) in ids.enumerated() {
            if let n = try runner.step(token: t, position: pos, wantLogits: pos == ids.count - 1) { next = n }
        }
        let prefill = Date().timeIntervalSince(prefillStart)
        let afterPrefill = runner.expertStats
        log(String(format: "prefill %.2fs (%.1f tok/s)", prefill, Double(ids.count) / prefill))

        var generated: [Int] = []
        var gpu = 0.0, io = 0.0
        let decodeStart = Date()
        while true {
            generated.append(next)
            FileHandle.standardOutput.write(tokenizer.decode([next]).data(using: .utf8)!)
            if eos.contains(next) || generated.count >= maxNew { break }
            next = try runner.step(token: next, position: ids.count + generated.count - 1)!
            gpu += runner.lastTiming.gpuWaitSeconds
            io += runner.lastTiming.ioSeconds
        }
        let decode = Date().timeIntervalSince(decodeStart)
        let steps = max(1, generated.count - 1)
        let s = runner.expertStats
        let hits = s.hits - afterPrefill.hits, misses = s.misses - afterPrefill.misses
        print("")
        log(String(format: "decode %d tokens in %.2fs: %.2f tok/s (gpu wait %.1f ms/tok, expert io %.1f ms/tok)",
                   generated.count, decode, Double(steps) / decode, gpu / Double(steps) * 1000,
                   io / Double(steps) * 1000))
        log(String(format: "decode expert cache: hit rate %.1f%%, %.0f MB read/tok",
                   100 * Double(hits) / Double(max(1, hits + misses)),
                   Double(s.bytesRead - afterPrefill.bytesRead) / Double(steps) / 1e6))
        log(String(format: "peak footprint %.2f GB", Double(peakFootprint()) / 1e9))
        if json {
            let out: [String: Any] = ["prompt_ids": ids, "generated_ids": generated]
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
