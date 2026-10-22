import ArgumentParser
import Darwin
import Foundation
import TurboFieldfareServerCore

@main
struct TurboFlintServer: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "turboflint-server",
        abstract: "OpenAI-compatible Chat Completions server for Recursive Flint on 127.0.0.1.")

    @Option(help: "Loopback port.")
    var port = 8080

    @Option(help: "Model id reported by /v1/models.")
    var modelID = "flint"

    @Option(help: "Ollama manifest for the Flint checkpoint (default: the local flint:40b-a2b-qat-int4).")
    var manifest: String?

    @Option(help: "Maximum context (prompt + completion) in tokens.")
    var maxContext = 16384

    @Option(help: "Routed-expert cache slots per MoE layer (1.77 MB each, x47 layers).")
    var slots = 32

    @Option(help: "Requests allowed to wait while one runs.")
    var queueLimit = 4

    func run() async throws {
        setvbuf(stdout, nil, _IOLBF, 0)
        let signals = ServerTerminationSignals()
        let backend = try await FlintServerBackend.load(manifestPath: manifest, maxContext: maxContext,
                                                        slotsPerLayer: slots)
        let server = TurboFieldfareHTTPServer(modelID: modelID, queueLimit: queueLimit, backend: backend)
        _ = try await server.start(port: port)
        print("turboflint-server ready at http://127.0.0.1:\(port) model=\(modelID) context=\(maxContext) slots=\(slots)")
        _ = await signals.wait()
        try await server.shutdown()
        await signals.cancel()
    }
}
