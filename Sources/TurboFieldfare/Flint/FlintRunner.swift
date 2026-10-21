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
    private let pGemv, pGateUp, pEmbed, pNorm, pQK, pCombine, pRouter, pMoEGateUp, pMoEDown,
                pAdd, pAdd2, pAttnFull, pAttnRing: MTLComputePipelineState

    // Scratch
    private let h, xn, qp, kp, vp, qr, attn, tmp, mid, sharedOut, moeMid, routed, logits: MTLBuffer
    private let routerIDs, routerWeights, slotBase, attnM, attnD, attnO: MTLBuffer
    private static let maxChunks = 32

    package private(set) var lastTiming = FlintStepTiming()

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
        pRouter = try context.pipeline("flint_router_topk")
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
        let H = c.hiddenSize, half = 2
        let qDim = c.numAttentionHeads * c.headDim, kvDim = c.numKeyValueHeads * c.headDim
        h = try buf(H * 4)
        xn = try buf(H * half)
        qp = try buf(qDim * half)
        kp = try buf(kvDim * half)
        vp = try buf(kvDim * half)
        qr = try buf(qDim * half)
        attn = try buf(qDim * half)
        tmp = try buf(H * half)
        mid = try buf(max(c.intermediateSize, c.sharedExpertIntermediateSize) * half)
        sharedOut = try buf(H * half)
        moeMid = try buf(c.numExpertsPerTok * c.moeIntermediateSize * half)
        routed = try buf(H * half)
        logits = try buf(c.vocabSize * half, shared: true)
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

    private func commandBuffer() throws -> (MTLCommandBuffer, MTLComputeCommandEncoder) {
        guard let cb = context.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw FlintRuntimeError.metal("command buffer")
        }
        return (cb, enc)
    }

    private func finish(_ cb: MTLCommandBuffer, wait: Bool, timing: inout FlintStepTiming) throws {
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
        for L in layers {
            encodeAttention(enc, L, position: position)
            guard let experts = L.experts, let router = L.router else {
                gateUp(enc, L.gate, L.up, x: xn, out: mid)
                gemv(enc, L.down, x: mid, y: tmp)
                addResidual(enc, tmp)
                continue
            }
            enc.setComputePipelineState(pRouter)
            enc.setBuffer(xn, offset: 0, index: 0)
            set(enc, router, 1)
            enc.setBuffer(routerIDs, offset: 0, index: 2)
            enc.setBuffer(routerWeights, offset: 0, index: 3)
            setU32(enc, c.numExperts, 4); setU32(enc, c.hiddenSize, 5); setU32(enc, k, 6)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
            enc.endEncoding()
            try finish(cb, wait: true, timing: &timing)

            // Shared expert runs on the GPU while the CPU loads missing routed experts.
            (cb, enc) = try commandBuffer()
            gateUp(enc, L.gate, L.up, x: xn, out: mid)
            gemv(enc, L.down, x: mid, y: sharedOut)
            enc.endEncoding()
            try finish(cb, wait: false, timing: &timing)

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
        }

        var result: Int?
        if wantLogits {
            rmsnorm(enc, weight: finalNorm, out: xn)
            gemv(enc, lmHead, x: xn, y: logits)
            enc.endEncoding()
            try finish(cb, wait: true, timing: &timing)
            result = argmax()
        } else {
            enc.endEncoding()
            try finish(cb, wait: true, timing: &timing)
        }
        timing.totalSeconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        lastTiming = timing
        return result
    }

    /// FP32 copy of the last logits (valid after a `step` with `wantLogits`).
    package func lastLogits() -> [Float] {
        let p = logits.contents().assumingMemoryBound(to: Float16.self)
        return (0..<config.vocabSize).map { Float(p[$0]) }
    }

    private func argmax() -> Int {
        let p = logits.contents().assumingMemoryBound(to: Float16.self)
        var best = 0
        var bv = p[0]
        for i in 1..<config.vocabSize where p[i] > bv { bv = p[i]; best = i }
        return best
    }
}
