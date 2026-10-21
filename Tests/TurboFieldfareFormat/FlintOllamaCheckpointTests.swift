import Foundation
import Testing
@testable import TurboFieldfareFormat

/// These tests read the real checkpoint from the local Ollama store and are skipped when it is absent.
@Suite(.enabled(if: FileManager.default.fileExists(atPath: FlintOllamaCheckpoint.defaultManifest.path)))
struct FlintOllamaCheckpointTests {
    let checkpoint: FlintOllamaCheckpoint

    init() throws {
        checkpoint = try FlintOllamaCheckpoint()
    }

    @Test func configMatchesRuntimeExpectations() throws {
        try checkpoint.config.validateForRuntime()
        #expect(checkpoint.config.numHiddenLayers == 49)
        #expect(checkpoint.config.noRopeLayerIndices.count == 14)
        #expect((0..<49).filter { checkpoint.config.isSliding(layer: $0) }.count == 35)
        #expect((0..<49).allSatisfy { checkpoint.config.isSliding(layer: $0) == checkpoint.config.usesRoPE(layer: $0) })
    }

    @Test func everyLayerHasItsTensors() throws {
        for layer in 0..<checkpoint.config.numHiddenLayers {
            let p = "model.layers.\(layer)."
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                let t = try checkpoint.quantized(p + "self_attn.\(proj).weight")
                #expect(t.bits == 4)
            }
            _ = try checkpoint.tensor(p + "self_attn.softmax_offset")
            if checkpoint.config.isDense(layer: layer) {
                #expect(try checkpoint.quantized(p + "mlp.gate_proj.weight").outFeatures == 6144)
            } else {
                let gate = try checkpoint.quantized(p + "mlp.experts.gate_proj.weight")
                let down = try checkpoint.quantized(p + "mlp.experts.down_proj.weight")
                #expect(gate.expertCount == 256 && gate.outFeatures == 512 && gate.inFeatures == 2048)
                #expect(down.expertCount == 256 && down.outFeatures == 2048 && down.inFeatures == 512)
                #expect(try checkpoint.tensor(p + "mlp.gate.weight").shape == [256, 2048])
                #expect(try checkpoint.quantized(p + "mlp.shared_expert.up_proj.weight").outFeatures == 1024)
            }
        }
    }

    @Test func embeddingIsInt8AndHeadIsInt4() throws {
        #expect(try checkpoint.quantized("model.embed_tokens.weight").bits == 8)
        let head = try checkpoint.quantized("lm_head.weight")
        #expect(head.bits == 4 && head.outFeatures == 150_144 && head.inFeatures == 2048)
    }

    @Test func expertSlicesAreContiguousAndSized() throws {
        let gate = try checkpoint.quantized("model.layers.2.mlp.experts.gate_proj.weight")
        let s0 = try gate.slice(expert: 0)
        let s1 = try gate.slice(expert: 1)
        #expect(s0.weight.count == 512 * 2048 / 2)
        #expect(s0.scale.count == 512 * 32 * 2 && s0.bias.count == 512 * 32 * 2)
        #expect(s1.weight.offset == s0.weight.offset + s0.weight.count)
        #expect(throws: FlintCheckpointError.self) { try gate.slice(expert: 256) }
    }
}
