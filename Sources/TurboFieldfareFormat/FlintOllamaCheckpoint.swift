import Foundation

/// Location of one tensor inside an Ollama per-tensor safetensors blob.
package struct FlintTensorLocation: Sendable, Equatable {
    package enum DType: String, Sendable, Equatable {
        case u32 = "U32"
        case bf16 = "BF16"
        case f32 = "F32"
        case u8 = "U8"

        package var byteWidth: UInt64 {
            switch self {
            case .u32, .f32: 4
            case .bf16: 2
            case .u8: 1
            }
        }
    }

    package let file: URL
    /// Absolute byte offset of the tensor payload within `file`.
    package let offset: UInt64
    package let byteCount: UInt64
    package let dtype: DType
    package let shape: [Int]
}

/// An int4/int8 affine-quantized matrix as MLX stores it: packed `weight` (u32), plus bf16 `scale`
/// and `bias` per 64-wide input group. For stacked expert tensors the leading dimension is the
/// expert index; `slice(expert:)` returns the contiguous byte ranges for one expert.
package struct FlintQuantizedTensor: Sendable, Equatable {
    package static let groupSize = 64

    package let name: String
    package let weight: FlintTensorLocation
    package let scale: FlintTensorLocation
    package let bias: FlintTensorLocation

    package var outFeatures: Int { weight.shape[weight.shape.count - 2] }
    package var inFeatures: Int { scale.shape[scale.shape.count - 1] * Self.groupSize }
    package var bits: Int { weight.shape[weight.shape.count - 1] * 32 / inFeatures }
    package var expertCount: Int? { weight.shape.count == 3 ? weight.shape[0] : nil }

    package struct ByteRange: Sendable, Equatable {
        package let file: URL
        package let offset: UInt64
        package let count: UInt64
    }

    /// Weight, scale, and bias byte ranges for one expert of a stacked [experts, out, in] tensor.
    package func slice(expert: Int) throws -> (weight: ByteRange, scale: ByteRange, bias: ByteRange) {
        guard let experts = expertCount, (0..<experts).contains(expert) else {
            throw FlintCheckpointError.invalid("\(name): expert \(expert) out of range")
        }
        func range(_ loc: FlintTensorLocation) -> ByteRange {
            let per = loc.byteCount / UInt64(experts)
            return ByteRange(file: loc.file, offset: loc.offset + per * UInt64(expert), count: per)
        }
        return (range(weight), range(scale), range(bias))
    }
}

package enum FlintCheckpointError: Error, Equatable, CustomStringConvertible, Sendable {
    case invalid(String)
    case missingTensor(String)

    package var description: String {
        switch self {
        case let .invalid(reason): reason
        case let .missingTensor(name): "missing tensor \(name)"
        }
    }
}

/// Subset of Flint's `config.json` the runtime depends on.
package struct FlintConfig: Codable, Sendable, Equatable {
    package struct Quantization: Codable, Sendable, Equatable {
        package let bits: Int
        package let groupSize: Int
        enum CodingKeys: String, CodingKey {
            case bits
            case groupSize = "group_size"
        }
    }

    package let modelType: String
    package let hiddenSize: Int
    package let intermediateSize: Int
    package let moeIntermediateSize: Int
    package let sharedExpertIntermediateSize: Int
    package let numHiddenLayers: Int
    package let numAttentionHeads: Int
    package let numKeyValueHeads: Int
    package let headDim: Int
    package let numExperts: Int
    package let numExpertsPerTok: Int
    package let vocabSize: Int
    package let rmsNormEps: Double
    package let ropeTheta: Double
    package let slidingWindow: Int
    package let layerTypes: [String]
    package let noRopeLayerIndices: [Int]
    package let mlpOnlyLayers: [Int]
    package let eosTokenID: [Int]
    package let maxPositionEmbeddings: Int
    package let tieWordEmbeddings: Bool
    package let normTopkProb: Bool
    package let scoringFunc: String
    package let quantization: Quantization

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case noRopeLayerIndices = "no_rope_layer_indices"
        case mlpOnlyLayers = "mlp_only_layers"
        case eosTokenID = "eos_token_id"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case normTopkProb = "norm_topk_prob"
        case scoringFunc = "scoring_func"
        case quantization
    }

    package func isSliding(layer: Int) -> Bool { layerTypes[layer] == "sliding_attention" }
    package func usesRoPE(layer: Int) -> Bool { !noRopeLayerIndices.contains(layer) }
    package func isDense(layer: Int) -> Bool { mlpOnlyLayers.contains(layer) }

    /// Rejects checkpoints whose shape differs from what the Flint kernels are written for.
    package func validateForRuntime() throws {
        func require(_ ok: Bool, _ what: String) throws {
            if !ok { throw FlintCheckpointError.invalid("unsupported Flint config: \(what)") }
        }
        try require(modelType == "flint", "model_type \(modelType)")
        try require(hiddenSize == 2048, "hidden_size \(hiddenSize)")
        try require(headDim == 128, "head_dim \(headDim)")
        try require(numAttentionHeads == 32 && numKeyValueHeads == 4, "heads \(numAttentionHeads)/\(numKeyValueHeads)")
        try require(numExperts == 256 && numExpertsPerTok == 6, "experts \(numExperts) top-\(numExpertsPerTok)")
        try require(moeIntermediateSize == 512 && sharedExpertIntermediateSize == 1024, "expert widths")
        try require(slidingWindow == 513, "sliding_window \(slidingWindow)")
        try require(layerTypes.count == numHiddenLayers, "layer_types count")
        try require(scoringFunc == "sigmoid" && normTopkProb, "router \(scoringFunc)")
        try require(!tieWordEmbeddings, "tied embeddings")
        try require(quantization == Quantization(bits: 4, groupSize: 64), "quantization")
    }
}

/// Index over the Flint checkpoint as Ollama stores it: one manifest listing ~780 blobs, each blob
/// a standalone safetensors file holding one tensor group (weight + `.scale` + `.bias`).
/// No bytes are copied; callers read tensor payloads with `pread`/`mmap` at the recorded offsets.
package struct FlintOllamaCheckpoint: Sendable {
    package static let defaultManifest = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent(".ollama/models/manifests/registry.ollama.ai/library/flint/40b-a2b-qat-int4")

    package let manifestURL: URL
    package let tensors: [String: FlintTensorLocation]
    /// Non-tensor files (config.json, tokenizer.json, ...) by their original file name.
    package let files: [String: URL]
    package let config: FlintConfig

    private struct Manifest: Decodable {
        struct Layer: Decodable {
            let mediaType: String
            let digest: String
            let size: UInt64
            let name: String?
        }
        let layers: [Layer]
    }

    private struct SafetensorsEntry: Decodable {
        let dtype: String
        let shape: [Int]
        let dataOffsets: [UInt64]
        enum CodingKeys: String, CodingKey {
            case dtype, shape
            case dataOffsets = "data_offsets"
        }
    }

    private struct DynamicKey: CodingKey {
        var stringValue: String
        var intValue: Int? { nil }
        init(stringValue: String) { self.stringValue = stringValue }
        init?(intValue _: Int) { nil }
    }

    /// Decodes a safetensors header, skipping `__metadata__`.
    private struct SafetensorsHeader: Decodable {
        let entries: [String: SafetensorsEntry]
        init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: DynamicKey.self)
            var out: [String: SafetensorsEntry] = [:]
            for key in c.allKeys where key.stringValue != "__metadata__" {
                out[key.stringValue] = try c.decode(SafetensorsEntry.self, forKey: key)
            }
            entries = out
        }
    }

    package init(manifestURL: URL = FlintOllamaCheckpoint.defaultManifest) throws {
        self.manifestURL = manifestURL
        // .../models/manifests/<registry>/<namespace>/<model>/<tag>  ->  .../models/blobs
        let blobs = manifestURL.deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("blobs")
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
        var tensors: [String: FlintTensorLocation] = [:]
        var files: [String: URL] = [:]
        for layer in manifest.layers {
            let url = blobs.appendingPathComponent(layer.digest.replacingOccurrences(of: ":", with: "-"))
            if layer.mediaType.hasSuffix(".tensor") {
                for (name, loc) in try Self.readHeader(url, expectedSize: layer.size) {
                    tensors[name] = loc
                }
            } else if let name = layer.name {
                files[name] = url
            }
        }
        guard let configURL = files["config.json"] else {
            throw FlintCheckpointError.invalid("manifest has no config.json")
        }
        self.tensors = tensors
        self.files = files
        self.config = try JSONDecoder().decode(FlintConfig.self, from: Data(contentsOf: configURL))
    }

    private static func readHeader(_ url: URL, expectedSize: UInt64) throws -> [String: FlintTensorLocation] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        guard let lengthData = try handle.read(upToCount: 8), lengthData.count == 8 else {
            throw FlintCheckpointError.invalid("\(url.lastPathComponent): truncated header length")
        }
        let headerLength = lengthData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian
        guard headerLength < 100 * 1024 * 1024, 8 + headerLength <= expectedSize,
              let json = try handle.read(upToCount: Int(headerLength)), json.count == Int(headerLength)
        else {
            throw FlintCheckpointError.invalid("\(url.lastPathComponent): bad header length \(headerLength)")
        }
        let header = try JSONDecoder().decode(SafetensorsHeader.self, from: json)
        let base = 8 + headerLength
        var out: [String: FlintTensorLocation] = [:]
        for (name, e) in header.entries {
            guard let dtype = FlintTensorLocation.DType(rawValue: e.dtype), e.dataOffsets.count == 2,
                  e.dataOffsets[0] <= e.dataOffsets[1], base + e.dataOffsets[1] <= expectedSize
            else {
                throw FlintCheckpointError.invalid("\(name): bad dtype or offsets in \(url.lastPathComponent)")
            }
            let count = e.dataOffsets[1] - e.dataOffsets[0]
            let elements = e.shape.reduce(UInt64(1)) { $0 * UInt64($1) }
            guard elements * dtype.byteWidth == count else {
                throw FlintCheckpointError.invalid("\(name): shape \(e.shape) does not match \(count) bytes")
            }
            out[name] = FlintTensorLocation(file: url, offset: base + e.dataOffsets[0], byteCount: count,
                                            dtype: dtype, shape: e.shape)
        }
        return out
    }

    package func tensor(_ name: String) throws -> FlintTensorLocation {
        guard let t = tensors[name] else { throw FlintCheckpointError.missingTensor(name) }
        return t
    }

    package func quantized(_ name: String) throws -> FlintQuantizedTensor {
        FlintQuantizedTensor(name: name, weight: try tensor(name), scale: try tensor(name + ".scale"),
                             bias: try tensor(name + ".bias"))
    }
}
