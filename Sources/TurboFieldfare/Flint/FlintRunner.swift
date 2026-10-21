import Foundation
import Metal
import TurboFieldfareFormat

package struct FlintRunnerOptions: Sendable {
    package var maxContext = 8192
    /// Routed-expert cache slots per MoE layer (each slot holds one 1.77 MB expert).
    package var slotsPerLayer = 32
    package var readConcurrency = 8
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
    private struct Layer {
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
                pAdd, pAdd2, pAttnFull, pAttnRing: MTLComputePipelineState

    // Scratch
    private let h, xn, qp, kp, vp, qr, attn, tmp, mid, sharedOut, moeMid, routed, logits: MTLBuffer
    private let routerScores, routerIDs, routerWeights, slotBase, attnM, attnD, attnO: MTLBuffer
    private static let maxChunks = 32

    package private(set) var lastTiming = FlintStepTiming()
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
        pAttnRing = try context.pipeline(
            "attention_decode_partial",
            constants: [MetalFunctionConstant(index: 69, value: .uint32(UInt32(checkpoint.config.slidingWindow)))])

        embed = try store.matrix("model.embed_tokens.weight")
        finalNorm = try store.ref("model.norm.weight")
        lmHead = try store.matrix("lm_head.weight")

        let c = checkpoint.config
        let kvRow = c.numKeyValueHeads * c.headDim * MemoryLayout<UInt16>.size
        var fds: [URL: Int32] = [:]
        for i in 0..<c.numHiddenLayers {
            let p = "model.layers.\(i)."
            let a = p + "self_attn."
            let dense = c.isDense(layer: i)
            let mlp = dense ? p + "mlp." : p + "mlp.shared_expert."
            let rows = c.isSliding(layer: i) ? c.slidingWindow : options.maxContext
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
                                                             slotCount: options.slotsPerLayer, fds: &fds),
                kCache: kc, vCache: vc))
        }
        expertFDs = fds

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
        routerScores = try buf(c.numExperts * 4)
        routerIDs = try buf(16 * 4, shared: true)
        routerWeights = try buf(16 * 4)
        slotBase = try buf(16 * 4, shared: true)
        attnM = try buf(c.numAttentionHeads * Self.maxChunks * 4)
        attnD = try buf(c.numAttentionHeads * Self.maxChunks * 4)
        attnO = try buf(c.numAttentionHeads * Self.maxChunks * c.headDim * 4)
        store.prefault()
    }

    deinit {
        for fd in expertFDs.values { close(fd) }
    }

    package var residentBytes: Int { store.mappedBytes }

    package var expertCacheBytes: Int { layers.compactMap(\.experts).reduce(0) { $0 + $1.pool.length } }

    package var expertStats: FlintExpertCacheStats {
        layers.compactMap(\.experts).reduce(into: FlintExpertCacheStats()) { acc, e in
            acc.hits += e.stats.hits
            acc.misses += e.stats.misses
            acc.bytesRead += e.stats.bytesRead
            acc.readSeconds += e.stats.readSeconds
        }
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

    private func encodeRouter(_ enc: MTLComputeCommandEncoder, _ router: FlintBufferRef) {
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
        enc.setBuffer(routerIDs, offset: 0, index: 1)
        enc.setBuffer(routerWeights, offset: 0, index: 2)
        setU32(enc, c.numExperts, 3); setU32(enc, c.numExpertsPerTok, 4)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    private struct QKParams {
        var numQHeads: UInt32, numKVHeads: UInt32, headDim: UInt32, cacheSlot: UInt32, position: UInt32
        var useRope: UInt32
        var theta: Float, eps: Float
    }

    private func encodeAttention(_ enc: MTLComputeCommandEncoder, _ L: Layer, position: Int) {
        let c = config
        rmsnorm(enc, weight: L.inputNorm, out: xn)
        gemv(enc, L.q, x: xn, y: qp)
        gemv(enc, L.k, x: xn, y: kp)
        gemv(enc, L.v, x: xn, y: vp)

        let slot = L.sliding ? position % c.slidingWindow : position
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

        gemv(enc, L.o, x: attn, y: tmp)
        addResidual(enc, tmp)
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
                gemv(enc, L.down, x: mid, y: tmp)
                addResidual(enc, tmp)
                if debugHidden != nil {
                    enc.endEncoding()
                    copyHidden(cb, layer: layerIndex)
                    try finish(cb, wait: false, timing: &timing, kind: 2)
                    (cb, enc) = try commandBuffer()
                }
                continue
            }
            encodeRouter(enc, router)
            enc.endEncoding()
            try finish(cb, wait: true, timing: &timing, kind: 0)

            // Shared expert runs on the GPU while the CPU loads missing routed experts.
            (cb, enc) = try commandBuffer()
            gateUp(enc, L.gate, L.up, x: xn, out: mid)
            gemv(enc, L.down, x: mid, y: sharedOut)
            enc.endEncoding()
            try finish(cb, wait: false, timing: &timing, kind: 1)

            let idPtr = routerIDs.contents().assumingMemoryBound(to: UInt32.self)
            let ids = (0..<k).map { Int(idPtr[$0]) }
            let ioStart = DispatchTime.now().uptimeNanoseconds
            let offsets = try experts.acquire(ids, readConcurrency: options.readConcurrency)
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
            enc.setComputePipelineState(pMoEDown)
            enc.setBuffer(experts.pool, offset: 0, index: 0)
            enc.setBuffer(slotBase, offset: 0, index: 1)
            enc.setBuffer(routerWeights, offset: 0, index: 2)
            enc.setBuffer(moeMid, offset: 0, index: 3)
            enc.setBuffer(routed, offset: 0, index: 4)
            enc.setBytes(&lay, length: lay.count * 4, index: 5)
            setU32(enc, c.moeIntermediateSize, 6); setU32(enc, c.hiddenSize, 7); setU32(enc, k, 8)
            enc.dispatchThreadgroups(MTLSize(width: (c.hiddenSize + 7) / 8, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            addResidual(enc, routed, sharedOut)
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
        return out
    }
}
