import Foundation
import Metal
import TurboFieldfareFormat

package struct FlintRunnerOptions: Sendable {
    package var maxContext = 8192
    /// Routed-expert cache slots per MoE layer (each slot holds one 1.77 MB expert).
    package var slotsPerLayer = 32
    package var readConcurrency = 8
    /// Prompt tokens processed per prefill chunk. Sliding-window KV rings hold
    /// `sliding_window + prefillChunk` rows so a whole chunk can be written before attention.
    package var prefillChunk = 256
    /// Extra sliding-window ring rows so a later prompt can rewind this many positions and reuse
    /// the cached prefix (servers reusing KV across turns). Costs ~10 KB per row across all layers.
    package var rewindSlack = 0
    /// Read routed experts with F_NOCACHE so the OS page cache does not hold a second copy.
    package var bypassPageCache = false
    /// Run each layer's successor router on its input and start loading the predicted experts.
    package var prefetchNextLayer = true
    package init() {}
}

package struct FlintStepTiming: Sendable {
    /// GPU execution time by command-buffer kind: attention+router, shared expert, routed experts, head.
    package var gpuAttention = 0.0, gpuShared = 0.0, gpuRouted = 0.0, gpuHead = 0.0
    package var gpuWaitSeconds = 0.0
    package var ioSeconds = 0.0
    package var totalSeconds = 0.0
}

/// Single-sequence Flint decoder: resident attention/router/shared weights mapped from the Ollama
/// blobs, routed experts streamed into per-layer slot caches on demand.
///
/// Per MoE layer: cb1 = norm, QKV, QK-norm/RoPE, attention (+sink), o_proj, residual, norm, router
/// (commit + wait) -> CPU reads the top-6 ids and loads missing experts while the shared-expert
/// command buffer runs -> cb2 = routed gate/up, weighted down-reduce, residual (committed, not
/// awaited; queue order serialises it before the next layer's cb1).
package final class FlintRunner {
    typealias Layer = LayerView

    struct LayerView {
        let inputNorm: FlintBufferRef
        let postNorm: FlintBufferRef
        let q, k, v, o: FlintQMatrix
        let qNorm, kNorm, sinks: FlintBufferRef
        let sliding: Bool
        let rope: Bool
        // Dense layers: gate/up/down of the 6144-wide MLP. MoE layers: shared expert + router.
        let gate, up, down: FlintQMatrix
        let router: FlintBufferRef?
        let experts: FlintExpertCache?
        let kCache, vCache: MTLBuffer
    }

    package let config: FlintConfig
    package let options: FlintRunnerOptions
    private let context: MetalContext
    private let store: FlintWeightStore
    private var layers: [Layer] = []
    private var expertFDs: [URL: Int32] = [:]
    private let embed: FlintQMatrix
    private let finalNorm: FlintBufferRef
    private let lmHead: FlintQMatrix

    // Pipelines
    private let pGemv, pGateUp, pEmbed, pNorm, pQK, pCombine, pRouterScores, pRouterSelect, pMoEGateUp, pMoEDown,
                pAdd, pAdd2, pAttnFull, pAttnRing,
                pQKV, pGemvRes, pMoEDownRes, pRouterScores2: MTLComputePipelineState

    // Scratch
    private let h, xn, qp, kp, vp, qr, attn, tmp, mid, sharedOut, moeMid, routed, logits: MTLBuffer
    private let routerScores, routerIDs, routerWeights, slotBase, predIDs, predWeights, attnM, attnD, attnO: MTLBuffer
    private static let maxChunks = 32

    package private(set) var lastTiming = FlintStepTiming()
    private var prefillEngine: FlintPrefill?
    /// When true, each MoE layer also runs the next layer's router on its own post-attention input
    /// and records how many of that layer's real top-k experts the guess contained.
    package var measurePrediction = false
    /// Debug: teacher-forced routing, `[layer][position] -> (ids, weights)`. When set, `step`
    /// replaces the router's choice with these, isolating numerics from routing near-ties.
    package var forcedRouting: [Int: [(ids: [Int], weights: [Float])]]?
    package private(set) var predictionOverlap = 0
    package private(set) var predictionTotal = 0
    private var lastPrediction: [Int] = []
    /// When set, `step` copies the residual stream after every layer here (FP32, layers x hidden).
    package var debugHidden: MTLBuffer?

    package init(checkpoint: FlintOllamaCheckpoint, context: MetalContext, options: FlintRunnerOptions) throws {
        try checkpoint.config.validateForRuntime()
        self.config = checkpoint.config
        self.options = options
        self.context = context
        let device = context.device
        let store = FlintWeightStore(checkpoint: checkpoint, device: device)
        self.store = store

        pGemv = try context.pipeline("flint_int4_gemv")
        pGateUp = try context.pipeline("flint_int4_gate_up_silu")
        pEmbed = try context.pipeline("flint_embed_int8")
        pNorm = try context.pipeline("flint_rmsnorm")
        pQK = try context.pipeline("flint_qk_norm_rope")
        pCombine = try context.pipeline("flint_attention_combine_sink")
        pRouterScores = try context.pipeline("flint_router_scores")
        pRouterSelect = try context.pipeline("flint_router_select")
        pMoEGateUp = try context.pipeline("flint_moe_gate_up")
        pMoEDown = try context.pipeline("flint_moe_down")
        pAdd = try context.pipeline("flint_residual_add")
        pAdd2 = try context.pipeline("flint_residual_add2")
        pAttnFull = try context.pipeline("attention_decode_partial")
        pQKV = try context.pipeline("flint_int4_gemv_qkv")
        pGemvRes = try context.pipeline("flint_int4_gemv_residual")
        pMoEDownRes = try context.pipeline("flint_moe_down_residual")
        pRouterScores2 = try context.pipeline("flint_router_scores2")
        pAttnRing = try context.pipeline(
            "attention_decode_partial",
            constants: [MetalFunctionConstant(
                index: 69, value: .uint32(UInt32(
                    checkpoint.config.slidingWindow + options.prefillChunk + options.rewindSlack)))])

        embed = try store.matrix("model.embed_tokens.weight")
        finalNorm = try store.ref("model.norm.weight")
        lmHead = try store.matrix("lm_head.weight")

        let c = checkpoint.config
        let kvRow = c.numKeyValueHeads * c.headDim * MemoryLayout<UInt16>.size
        var fds: [URL: Int32] = [:]
        let io = FlintIOQueue(concurrency: options.readConcurrency)
        for i in 0..<c.numHiddenLayers {
            let p = "model.layers.\(i)."
            let a = p + "self_attn."
            let dense = c.isDense(layer: i)
            let mlp = dense ? p + "mlp." : p + "mlp.shared_expert."
            let rows = c.isSliding(layer: i)
                ? c.slidingWindow + options.prefillChunk + options.rewindSlack : options.maxContext
            guard let kc = device.makeBuffer(length: rows * kvRow, options: .storageModePrivate),
                  let vc = device.makeBuffer(length: rows * kvRow, options: .storageModePrivate)
            else { throw FlintRuntimeError.metal("KV cache allocation") }
            layers.append(Layer(
                inputNorm: try store.ref(p + "input_layernorm.weight"),
                postNorm: try store.ref(p + "post_attention_layernorm.weight"),
                q: try store.matrix(a + "q_proj.weight"), k: try store.matrix(a + "k_proj.weight"),
                v: try store.matrix(a + "v_proj.weight"), o: try store.matrix(a + "o_proj.weight"),
                qNorm: try store.ref(a + "q_norm.weight"), kNorm: try store.ref(a + "k_norm.weight"),
                sinks: try store.ref(a + "softmax_offset"),
                sliding: c.isSliding(layer: i), rope: c.usesRoPE(layer: i),
                gate: try store.matrix(mlp + "gate_proj.weight"), up: try store.matrix(mlp + "up_proj.weight"),
                down: try store.matrix(mlp + "down_proj.weight"),
                router: dense ? nil : try store.ref(p + "mlp.gate.weight"),
                experts: dense ? nil : try FlintExpertCache(layer: i, checkpoint: checkpoint, device: device,
                                                             slotCount: options.slotsPerLayer, io: io, fds: &fds),
                kCache: kc, vCache: vc))
        }
        expertFDs = fds
        if options.bypassPageCache {
            for fd in fds.values { _ = fcntl(fd, F_NOCACHE, 1) }
        }

        func buf(_ bytes: Int, shared: Bool = false) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(bytes, 16),
                                            options: shared ? .storageModeShared : .storageModePrivate)
            else { throw FlintRuntimeError.metal("scratch allocation") }
            return b
        }
        let H = c.hiddenSize, f32 = 4
        let qDim = c.numAttentionHeads * c.headDim, kvDim = c.numKeyValueHeads * c.headDim
        h = try buf(H * f32)
        xn = try buf(H * f32)
        qp = try buf(qDim * f32)
        kp = try buf(kvDim * f32)
        vp = try buf(kvDim * f32)
        qr = try buf(qDim * 2)  // FP16 query, the format attention_decode_partial reads
        attn = try buf(qDim * f32)
        tmp = try buf(H * f32)
        mid = try buf(max(c.intermediateSize, c.sharedExpertIntermediateSize) * f32)
        sharedOut = try buf(H * f32)
        moeMid = try buf(c.numExpertsPerTok * c.moeIntermediateSize * f32)
        routed = try buf(H * f32)
        logits = try buf(c.vocabSize * f32, shared: true)
        routerScores = try buf(2 * c.numExperts * 4)
        predIDs = try buf(16 * 4, shared: true)
        predWeights = try buf(16 * 4)
        routerIDs = try buf(16 * 4, shared: true)
        routerWeights = try buf(16 * 4, shared: true)
        slotBase = try buf(16 * 4, shared: true)
        attnM = try buf(c.numAttentionHeads * Self.maxChunks * 4)
        attnD = try buf(c.numAttentionHeads * Self.maxChunks * 4)
        attnO = try buf(c.numAttentionHeads * Self.maxChunks * c.headDim * 4)
        store.prefault()
        if options.prefillChunk > 1 {
            prefillEngine = try FlintPrefill(context: context, config: configView, maxTokens: options.prefillChunk)
        }
    }

    private var configView: FlintConfigView {
        let c = config
        return FlintConfigView(
            hidden: c.hiddenSize, qHeads: c.numAttentionHeads, kvHeads: c.numKeyValueHeads, headDim: c.headDim,
            qDim: c.numAttentionHeads * c.headDim, kvDim: c.numKeyValueHeads * c.headDim,
            denseFFN: c.intermediateSize, sharedFFN: c.sharedExpertIntermediateSize,
            expertFFN: c.moeIntermediateSize, experts: c.numExperts, topK: c.numExpertsPerTok,
            window: c.slidingWindow, ringCap: ringCap, theta: Float(c.ropeTheta), eps: Float(c.rmsNormEps))
    }

    /// Processes a prompt starting at `startPosition` and returns the greedy next token.
    /// Uses chunked prefill when enabled, otherwise one decode step per token.
    package func prefill(_ ids: [Int], startPosition: Int = 0) throws -> Int {
        precondition(!ids.isEmpty && startPosition + ids.count <= options.maxContext, "prompt exceeds context")
        guard let engine = prefillEngine else {
            var next = 0
            for (i, t) in ids.enumerated() {
                if let n = try step(token: t, position: startPosition + i, wantLogits: i == ids.count - 1) { next = n }
            }
            return next
        }
        var pos = startPosition
        for chunkStart in stride(from: 0, to: ids.count, by: options.prefillChunk) {
            let chunk = Array(ids[chunkStart..<min(chunkStart + options.prefillChunk, ids.count)])
            try engine.run(ids: chunk, start: pos, layers: layers, embed: embed, c: configView)
            pos += chunk.count
            if chunkStart + chunk.count == ids.count {
                // Output norm + head on the final row only.
                let (cb, enc) = try commandBuffer()
                enc.setComputePipelineState(pNorm)
                enc.setBuffer(engine.hidden, offset: (chunk.count - 1) * config.hiddenSize * 4, index: 0)
                set(enc, finalNorm, 1)
                enc.setBuffer(xn, offset: 0, index: 2)
                setU32(enc, config.hiddenSize, 3)
                var eps = Float(config.rmsNormEps)
                enc.setBytes(&eps, length: 4, index: 4)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
                gemv(enc, lmHead, x: xn, y: logits)
                enc.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
                if let error = cb.error { throw FlintRuntimeError.metal("\(error)") }
            }
        }
        return argmax()
    }

    deinit {
        for fd in expertFDs.values { close(fd) }
    }

    package var residentBytes: Int { store.mappedBytes }

    private var ringCap: Int { config.slidingWindow + options.prefillChunk + options.rewindSlack }

    /// How far back a later `prefill` may restart while every sliding-window key it needs is still
    /// in the rings. Callers must re-prefill from 0 when (last written position - restart) exceeds it.
    package var maxRewind: Int { options.prefillChunk + options.rewindSlack - 1 }

    package var expertCacheBytes: Int { layers.compactMap(\.experts).reduce(0) { $0 + $1.pool.length } }

    package var expertStats: FlintExpertCacheStats {
        layers.compactMap(\.experts).reduce(FlintExpertCacheStats()) { $0 + $1.stats }
    }

    // MARK: - Encoding helpers

    private func set(_ enc: MTLComputeCommandEncoder, _ r: FlintBufferRef, _ i: Int) {
        enc.setBuffer(r.buffer, offset: r.offset, index: i)
    }

    private func setU32(_ enc: MTLComputeCommandEncoder, _ v: Int, _ i: Int) {
        var x = UInt32(v)
        enc.setBytes(&x, length: 4, index: i)
    }

    private func gemv(_ enc: MTLComputeCommandEncoder, _ m: FlintQMatrix, x: MTLBuffer, y: MTLBuffer) {
        enc.setComputePipelineState(pGemv)
        set(enc, m.w, 0); set(enc, m.s, 1); set(enc, m.b, 2)
        enc.setBuffer(x, offset: 0, index: 3)
        enc.setBuffer(y, offset: 0, index: 4)
        setU32(enc, m.outFeatures, 5); setU32(enc, m.inFeatures, 6)
        enc.dispatchThreadgroups(MTLSize(width: (m.outFeatures + 7) / 8, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// h += m . x
    private func gemvResidual(_ enc: MTLComputeCommandEncoder, _ m: FlintQMatrix, x: MTLBuffer) {
        enc.setComputePipelineState(pGemvRes)
        set(enc, m.w, 0); set(enc, m.s, 1); set(enc, m.b, 2)
        enc.setBuffer(x, offset: 0, index: 3)
        enc.setBuffer(h, offset: 0, index: 4)
        setU32(enc, m.outFeatures, 5); setU32(enc, m.inFeatures, 6)
        enc.dispatchThreadgroups(MTLSize(width: (m.outFeatures + 7) / 8, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    private func gateUp(_ enc: MTLComputeCommandEncoder, _ g: FlintQMatrix, _ u: FlintQMatrix,
                        x: MTLBuffer, out: MTLBuffer) {
        enc.setComputePipelineState(pGateUp)
        set(enc, g.w, 0); set(enc, g.s, 1); set(enc, g.b, 2)
        set(enc, u.w, 3); set(enc, u.s, 4); set(enc, u.b, 5)
        enc.setBuffer(x, offset: 0, index: 6)
        enc.setBuffer(out, offset: 0, index: 7)
        setU32(enc, g.outFeatures, 8); setU32(enc, g.inFeatures, 9)
        enc.dispatchThreadgroups(MTLSize(width: (g.outFeatures + 7) / 8, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    private func rmsnorm(_ enc: MTLComputeCommandEncoder, weight: FlintBufferRef, out: MTLBuffer) {
        enc.setComputePipelineState(pNorm)
        enc.setBuffer(h, offset: 0, index: 0)
        set(enc, weight, 1)
        enc.setBuffer(out, offset: 0, index: 2)
        setU32(enc, config.hiddenSize, 3)
        var eps = Float(config.rmsNormEps)
        enc.setBytes(&eps, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    }

    private func addResidual(_ enc: MTLComputeCommandEncoder, _ a: MTLBuffer, _ b: MTLBuffer? = nil) {
        enc.setComputePipelineState(b == nil ? pAdd : pAdd2)
        enc.setBuffer(h, offset: 0, index: 0)
        enc.setBuffer(a, offset: 0, index: 1)
        if let b { enc.setBuffer(b, offset: 0, index: 2) }
        setU32(enc, config.hiddenSize, b == nil ? 2 : 3)
        enc.dispatchThreads(MTLSize(width: config.hiddenSize, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    private func encodeRouter(_ enc: MTLComputeCommandEncoder, _ router: FlintBufferRef,
                              ids: MTLBuffer? = nil, weights: MTLBuffer? = nil) {
        let c = config
        enc.setComputePipelineState(pRouterScores)
        enc.setBuffer(xn, offset: 0, index: 0)
        set(enc, router, 1)
        enc.setBuffer(routerScores, offset: 0, index: 2)
        setU32(enc, c.numExperts, 3); setU32(enc, c.hiddenSize, 4)
        enc.dispatchThreadgroups(MTLSize(width: (c.numExperts + 7) / 8, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.setComputePipelineState(pRouterSelect)
        enc.setBuffer(routerScores, offset: 0, index: 0)
        enc.setBuffer(ids ?? routerIDs, offset: 0, index: 1)
        enc.setBuffer(weights ?? routerWeights, offset: 0, index: 2)
        setU32(enc, c.numExperts, 3); setU32(enc, c.numExpertsPerTok, 4)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    /// Scores this layer's router and, when `next` is given, the next layer's router on the same
    /// input; selects top-k for both (ids/weights and predIDs/predWeights).
    private func encodeRouterPair(_ enc: MTLComputeCommandEncoder, _ router: FlintBufferRef, _ next: FlintBufferRef?) {
        let c = config
        let sets = next == nil ? 1 : 2
        enc.setComputePipelineState(pRouterScores2)
        enc.setBuffer(xn, offset: 0, index: 0)
        set(enc, router, 1)
        set(enc, next ?? router, 2)
        enc.setBuffer(routerScores, offset: 0, index: 3)
        setU32(enc, c.numExperts, 4); setU32(enc, c.hiddenSize, 5)
        enc.dispatchThreadgroups(MTLSize(width: (sets * c.numExperts + 7) / 8, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.setComputePipelineState(pRouterSelect)
        for s in 0..<sets {
            enc.setBuffer(routerScores, offset: s * c.numExperts * 4, index: 0)
            enc.setBuffer(s == 0 ? routerIDs : predIDs, offset: 0, index: 1)
            enc.setBuffer(s == 0 ? routerWeights : predWeights, offset: 0, index: 2)
            setU32(enc, c.numExperts, 3); setU32(enc, c.numExpertsPerTok, 4)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
    }

    private struct QKParams {
        var numQHeads: UInt32, numKVHeads: UInt32, headDim: UInt32, cacheSlot: UInt32, position: UInt32
        var useRope: UInt32
        var theta: Float, eps: Float
    }

    private func encodeAttention(_ enc: MTLComputeCommandEncoder, _ L: Layer, position: Int) {
        let c = config
        rmsnorm(enc, weight: L.inputNorm, out: xn)
        enc.setComputePipelineState(pQKV)
        set(enc, L.q.w, 0); set(enc, L.q.s, 1); set(enc, L.q.b, 2)
        set(enc, L.k.w, 3); set(enc, L.k.s, 4); set(enc, L.k.b, 5)
        set(enc, L.v.w, 6); set(enc, L.v.s, 7); set(enc, L.v.b, 8)
        enc.setBuffer(xn, offset: 0, index: 9)
        enc.setBuffer(qp, offset: 0, index: 10); enc.setBuffer(kp, offset: 0, index: 11); enc.setBuffer(vp, offset: 0, index: 12)
        setU32(enc, L.q.outFeatures, 13); setU32(enc, L.k.outFeatures, 14); setU32(enc, L.q.inFeatures, 15)
        enc.dispatchThreadgroups(MTLSize(width: (L.q.outFeatures + 2 * L.k.outFeatures + 7) / 8, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        let slot = L.sliding ? position % ringCap : position
        var params = QKParams(numQHeads: UInt32(c.numAttentionHeads), numKVHeads: UInt32(c.numKeyValueHeads),
                              headDim: UInt32(c.headDim), cacheSlot: UInt32(slot), position: UInt32(position),
                              useRope: L.rope ? 1 : 0, theta: Float(c.ropeTheta), eps: Float(c.rmsNormEps))
        enc.setComputePipelineState(pQK)
        enc.setBuffer(qp, offset: 0, index: 0)
        enc.setBuffer(kp, offset: 0, index: 1)
        enc.setBuffer(vp, offset: 0, index: 2)
        set(enc, L.qNorm, 3); set(enc, L.kNorm, 4)
        enc.setBuffer(qr, offset: 0, index: 5)
        enc.setBuffer(L.kCache, offset: 0, index: 6)
        enc.setBuffer(L.vCache, offset: 0, index: 7)
        enc.setBytes(&params, length: MemoryLayout<QKParams>.stride, index: 8)
        enc.dispatchThreadgroups(MTLSize(width: c.numAttentionHeads + c.numKeyValueHeads, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: c.headDim, height: 1, depth: 1))

        // Split-KV attention over [kvStart, position].
        let seqLen = position + 1
        let kvStart = L.sliding ? max(0, seqLen - c.slidingWindow) : 0
        let span = seqLen - kvStart
        let chunks = min(Self.maxChunks, max(1, (span + 63) / 64))
        let chunkLen = (span + chunks - 1) / chunks
        enc.setComputePipelineState(L.sliding ? pAttnRing : pAttnFull)
        enc.setBuffer(qr, offset: 0, index: 0)
        enc.setBuffer(L.kCache, offset: 0, index: 1)
        enc.setBuffer(L.vCache, offset: 0, index: 2)
        enc.setBuffer(attnM, offset: 0, index: 3)
        enc.setBuffer(attnD, offset: 0, index: 4)
        enc.setBuffer(attnO, offset: 0, index: 5)
        setU32(enc, c.headDim, 6); setU32(enc, c.numAttentionHeads, 7); setU32(enc, c.numKeyValueHeads, 8)
        setU32(enc, seqLen, 9); setU32(enc, kvStart, 10); setU32(enc, chunkLen, 11); setU32(enc, chunks, 12)
        var scale = Float(1.0 / Double(c.headDim).squareRoot())
        enc.setBytes(&scale, length: 4, index: 13)
        enc.dispatchThreadgroups(MTLSize(width: c.numAttentionHeads * chunks, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: c.headDim, height: 1, depth: 1))

        enc.setComputePipelineState(pCombine)
        enc.setBuffer(attnM, offset: 0, index: 0)
        enc.setBuffer(attnD, offset: 0, index: 1)
        enc.setBuffer(attnO, offset: 0, index: 2)
        set(enc, L.sinks, 3)
        enc.setBuffer(attn, offset: 0, index: 4)
        setU32(enc, c.headDim, 5); setU32(enc, chunks, 6)
        enc.dispatchThreadgroups(MTLSize(width: c.numAttentionHeads, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: c.headDim, height: 1, depth: 1))

        gemvResidual(enc, L.o, x: attn)
        rmsnorm(enc, weight: L.postNorm, out: xn)
    }

    private func copyHidden(_ cb: MTLCommandBuffer, layer: Int) {
        guard let dst = debugHidden, let blit = cb.makeBlitCommandEncoder() else { return }
        let bytes = config.hiddenSize * 4
        blit.copy(from: h, sourceOffset: 0, to: dst, destinationOffset: layer * bytes, size: bytes)
        blit.endEncoding()
    }

    package func makeDebugHiddenBuffer() -> MTLBuffer? {
        context.device.makeBuffer(length: config.numHiddenLayers * config.hiddenSize * 4, options: .storageModeShared)
    }

    private func commandBuffer() throws -> (MTLCommandBuffer, MTLComputeCommandEncoder) {
        guard let cb = context.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw FlintRuntimeError.metal("command buffer")
        }
        return (cb, enc)
    }

    private var pendingTimed: [(MTLCommandBuffer, Int)] = []

    private func finish(_ cb: MTLCommandBuffer, wait: Bool, timing: inout FlintStepTiming, kind: Int = 0) throws {
        pendingTimed.append((cb, kind))
        cb.commit()
        if wait {
            let t = DispatchTime.now().uptimeNanoseconds
            cb.waitUntilCompleted()
            timing.gpuWaitSeconds += Double(DispatchTime.now().uptimeNanoseconds - t) / 1e9
            if let error = cb.error { throw FlintRuntimeError.metal("\(error)") }
        }
    }

    // MARK: - Decode

    /// Runs one token at `position`. Returns the greedy next token when `wantLogits` is true.
    @discardableResult
    package func step(token: Int, position: Int, wantLogits: Bool = true) throws -> Int? {
        precondition(position < options.maxContext, "context full")
        let c = config
        var timing = FlintStepTiming()
        let start = DispatchTime.now().uptimeNanoseconds

        var (cb, enc) = try commandBuffer()
        enc.setComputePipelineState(pEmbed)
        set(enc, embed.w, 0); set(enc, embed.s, 1); set(enc, embed.b, 2)
        enc.setBuffer(h, offset: 0, index: 3)
        setU32(enc, token, 4); setU32(enc, c.hiddenSize, 5)
        enc.dispatchThreads(MTLSize(width: c.hiddenSize, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        let k = c.numExpertsPerTok
        for (layerIndex, L) in layers.enumerated() {
            encodeAttention(enc, L, position: position)
            guard let experts = L.experts, let router = L.router else {
                gateUp(enc, L.gate, L.up, x: xn, out: mid)
                gemvResidual(enc, L.down, x: mid)
                if debugHidden != nil {
                    enc.endEncoding()
                    copyHidden(cb, layer: layerIndex)
                    try finish(cb, wait: false, timing: &timing, kind: 2)
                    (cb, enc) = try commandBuffer()
                }
                continue
            }
            let nextRouter = layerIndex + 1 < layers.count ? layers[layerIndex + 1].router : nil
            let predict = (measurePrediction || options.prefetchNextLayer) ? nextRouter : nil
            encodeRouterPair(enc, router, predict)
            enc.endEncoding()
            try finish(cb, wait: true, timing: &timing, kind: 0)

            // Shared expert runs on the GPU while the CPU loads missing routed experts.
            (cb, enc) = try commandBuffer()
            gateUp(enc, L.gate, L.up, x: xn, out: mid)
            gemv(enc, L.down, x: mid, y: sharedOut)
            enc.endEncoding()
            try finish(cb, wait: false, timing: &timing, kind: 1)

            let idPtr = routerIDs.contents().assumingMemoryBound(to: UInt32.self)
            var ids = (0..<k).map { Int(idPtr[$0]) }
            if let forced = forcedRouting?[layerIndex], position < forced.count {
                ids = forced[position].ids
                let wp = routerWeights.contents().assumingMemoryBound(to: Float.self)
                for j in 0..<k { idPtr[j] = UInt32(ids[j]); wp[j] = forced[position].weights[j] }
            }
            if measurePrediction {
                if !lastPrediction.isEmpty {
                    predictionOverlap += Set(lastPrediction).intersection(ids).count
                    predictionTotal += k
                }
                let pp = predIDs.contents().assumingMemoryBound(to: UInt32.self)
                lastPrediction = nextRouter == nil ? [] : (0..<k).map { Int(pp[$0]) }
            }
            let ioStart = DispatchTime.now().uptimeNanoseconds
            let ticket = experts.request(ids)
            if options.prefetchNextLayer, nextRouter != nil, let next = layers[layerIndex + 1].experts {
                let pp = predIDs.contents().assumingMemoryBound(to: UInt32.self)
                next.prefetch((0..<k).map { Int(pp[$0]) })
            }
            let offsets = try experts.wait(ticket)
            timing.ioSeconds += Double(DispatchTime.now().uptimeNanoseconds - ioStart) / 1e9
            let sb = slotBase.contents().assumingMemoryBound(to: UInt32.self)
            for (i, o) in offsets.enumerated() { sb[i] = o }

            (cb, enc) = try commandBuffer()
            var lay = experts.layout.offsets
            enc.setComputePipelineState(pMoEGateUp)
            enc.setBuffer(experts.pool, offset: 0, index: 0)
            enc.setBuffer(slotBase, offset: 0, index: 1)
            enc.setBuffer(xn, offset: 0, index: 2)
            enc.setBuffer(moeMid, offset: 0, index: 3)
            enc.setBytes(&lay, length: lay.count * 4, index: 4)
            setU32(enc, c.moeIntermediateSize, 5); setU32(enc, c.hiddenSize, 6); setU32(enc, k, 7)
            enc.dispatchThreadgroups(MTLSize(width: (k * c.moeIntermediateSize + 7) / 8, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.setComputePipelineState(pMoEDownRes)
            enc.setBuffer(experts.pool, offset: 0, index: 0)
            enc.setBuffer(slotBase, offset: 0, index: 1)
            enc.setBuffer(routerWeights, offset: 0, index: 2)
            enc.setBuffer(moeMid, offset: 0, index: 3)
            enc.setBuffer(sharedOut, offset: 0, index: 4)
            enc.setBuffer(h, offset: 0, index: 5)
            enc.setBytes(&lay, length: lay.count * 4, index: 6)
            setU32(enc, c.moeIntermediateSize, 7); setU32(enc, c.hiddenSize, 8); setU32(enc, k, 9)
            enc.dispatchThreadgroups(MTLSize(width: (c.hiddenSize + 7) / 8, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            if debugHidden != nil { copyHidden(cb, layer: layerIndex) }
            try finish(cb, wait: false, timing: &timing, kind: 2)
            (cb, enc) = try commandBuffer()
        }

        var result: Int?
        if wantLogits {
            rmsnorm(enc, weight: finalNorm, out: xn)
            gemv(enc, lmHead, x: xn, y: logits)
            enc.endEncoding()
            try finish(cb, wait: true, timing: &timing, kind: 3)
            result = argmax()
        } else {
            enc.endEncoding()
            try finish(cb, wait: true, timing: &timing)
        }
        for (b, kind) in pendingTimed {
            let t = b.gpuEndTime - b.gpuStartTime
            switch kind {
            case 0: timing.gpuAttention += t
            case 1: timing.gpuShared += t
            case 2: timing.gpuRouted += t
            default: timing.gpuHead += t
            }
        }
        pendingTimed.removeAll(keepingCapacity: true)
        timing.totalSeconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        lastTiming = timing
        return result
    }

    /// FP32 copy of the last logits (valid after a `step` with `wantLogits`).
    package func lastLogits() -> [Float] {
        let p = logits.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: p, count: config.vocabSize))
    }

    private func argmax() -> Int {
        let p = logits.contents().assumingMemoryBound(to: Float.self)
        var best = 0
        var bv = p[0]
        for i in 1..<config.vocabSize where p[i] > bv { bv = p[i]; best = i }
        return best
    }
}

// MARK: - Kernel profiling

extension FlintRunner {
    /// GPU time (ms) of each attention-block kernel on MoE layer `layerIndex`, averaged over
    /// `reps` dispatches in one command buffer. For finding hot spots, not for benchmarks.
    package func profileKernels(layerIndex: Int = 2, position: Int = 100, reps: Int = 200) throws -> [(String, Double)] {
        let L = layers[layerIndex]
        func time(_ name: String, _ body: (MTLComputeCommandEncoder) -> Void) throws -> (String, Double) {
            let (cb, enc) = try commandBuffer()
            for _ in 0..<reps { body(enc) }
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            return (name, (cb.gpuEndTime - cb.gpuStartTime) / Double(reps) * 1000)
        }
        var out: [(String, Double)] = []
        out.append(try time("rmsnorm") { self.rmsnorm($0, weight: L.inputNorm, out: self.xn) })
        out.append(try time("q_proj 4096x2048") { self.gemv($0, L.q, x: self.xn, y: self.qp) })
        out.append(try time("k_proj 512x2048") { self.gemv($0, L.k, x: self.xn, y: self.kp) })
        out.append(try time("o_proj 2048x4096") { self.gemv($0, L.o, x: self.attn, y: self.tmp) })
        out.append(try time("residual_add") { self.addResidual($0, self.tmp) })
        out.append(try time("full attention block") { self.encodeAttention($0, L, position: position) })
        out.append(try time("router+top6") { self.encodeRouter($0, L.router!) })
        out.append(try time("shared gate_up 1024x2048") { self.gateUp($0, L.gate, L.up, x: self.xn, out: self.mid) })
        out.append(try time("lm_head 150144x2048") { self.gemv($0, self.lmHead, x: self.xn, y: self.logits) })
        // Same kernel, one dispatch per command buffer, after an idle gap (GPU clock ramp test).
        for gapMs in [0, 5, 20, 50] {
            var total = 0.0
            for _ in 0..<10 {
                if gapMs > 0 { usleep(useconds_t(gapMs * 1000)) }
                let (cb, enc) = try commandBuffer()
                gemv(enc, lmHead, x: xn, y: logits)
                enc.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
                total += cb.gpuEndTime - cb.gpuStartTime
            }
            out.append(("lm_head after \(gapMs) ms idle", total / 10 * 1000))
        }
        return out
    }
}
