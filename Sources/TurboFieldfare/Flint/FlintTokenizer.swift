import Foundation
import Hub
import Tokenizers
import TurboFieldfareFormat

/// Flint's tokenizer, loaded from the `tokenizer.json`/`tokenizer_config.json` blobs in the Ollama
/// store. swift-transformers wants a folder with those file names, so we keep one of symlinks in
/// the user's caches directory.
package final class FlintTokenizer: @unchecked Sendable {
    private let tokenizer: any Tokenizer

    package static func load(checkpoint: FlintOllamaCheckpoint) async throws -> FlintTokenizer {
        let fm = FileManager.default
        let folder = fm.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("TurboFlint/tokenizer", isDirectory: true)
        try fm.createDirectory(at: folder, withIntermediateDirectories: true)
        for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"] {
            guard let target = checkpoint.files[name] else { continue }
            let link = folder.appendingPathComponent(name)
            if (try? fm.destinationOfSymbolicLink(atPath: link.path)) != target.path {
                try? fm.removeItem(at: link)
                try fm.createSymbolicLink(at: link, withDestinationURL: target)
            }
        }
        let hub = LanguageModelConfigurationFromHub(modelFolder: folder)
        guard let config = try await hub.tokenizerConfig else {
            throw FlintRuntimeError.invalid("tokenizer_config.json missing")
        }
        let data = try await hub.tokenizerData
        return FlintTokenizer(try AutoTokenizer.from(tokenizerConfig: config, tokenizerData: data))
    }

    private init(_ tokenizer: any Tokenizer) { self.tokenizer = tokenizer }

    /// Flint adds no BOS token; special tokens such as `<|user|>` are matched in the text.
    package func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }

    package func decode(_ ids: [Int]) -> String {
        tokenizer.decode(tokens: ids, skipSpecialTokens: false)
    }

    /// Single user turn in Flint's chat format (see recursive-oss/rec chat_template.jinja).
    package static func chatPrompt(user: String, system: String? = nil) -> String {
        var s = ""
        if let system { s += "<|system|>\(system)" }
        return s + "<|user|>\(user)<|assistant|>"
    }
}
