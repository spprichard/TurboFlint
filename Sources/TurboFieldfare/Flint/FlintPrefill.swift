import Foundation
import Metal

/// Chunked prompt processing. Each chunk runs layer by layer over all of its tokens, so a routed
/// expert is read from disk once per chunk instead of once per token.
package final class FlintPrefill {
    private struct ExpertGroup {
        var slotOffset: UInt32
        var count: UInt32
        var pairs: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)
    }

    private struct QKParams {
        var numQHeads: UInt32, numKVHeads: UInt32, headDim: UInt32, basePosition: UInt32
        var ringCap: UInt32, useRope: UInt32
        var theta: Float, eps: Float
    }

    private struct AttnParams {
        var numQHeads: UInt32, numKVHeads: UInt32, headDim: UInt32, basePosition: UInt32
        var window: UInt32, ringCap: UInt32
        var scale: Float
    }

    private let maxT: Int
    private let context: MetalContext
    private let pGemm, pGateUp, pEmbed, pNorm, pQK, pAttn, pScores, pSelect, pMoEGateUp, pMoEDown, pReduce:
        MTLComputePipelineState
    private let tokens, H, XN, Q, Kp, Vp, QR, ATT, MID, SHMID, SHARED, SCORES, IDS, WTS, HMID, ROUTED: MTLBuffer

    package private(set) var expertGroups = 0

    init(context: MetalContext, config c: FlintConfigView, maxTokens: Int) throws {
        self.context = context
        self.maxT = maxTokens
        pGemm = try context.pipeline("flint_int4_gemm_rows")
        pGateUp = try context.pipeline("flint_int4_gate_up_rows")
        pEmbed = try context.pipeline("flint_embed_int8_rows")
        pNorm = try context.pipeline("flint_rmsnorm_rows")
        pQK = try context.pipeline("flint_qk_norm_rope_rows")
        pAttn = try context.pipeline("flint_attention_prefill")
        pScores = try context.pipeline("flint_router_scores_rows")
        pSelect = try context.pipeline("flint_router_select_rows")
        pMoEGateUp = try context.pipeline("flint_moe_gate_up_grouped")
        pMoEDown = try context.pipeline("flint_moe_down_grouped")
        pReduce = try context.pipeline("flint_moe_reduce_rows")
        let d = context.device
        func buf(_ n: Int, shared: Bool = false) throws -> MTLBuffer {
            guard let b = d.makeBuffer(length: max(16, n), options: shared ? .storageModeShared : .storageModePrivate)
            else { throw FlintRuntimeError.metal("prefill scratch") }
            return b
        }
        let T = maxTokens, f = 4
        tokens = try buf(T * 4, shared: true)
        H = try buf(T * c.hidden * f)
        XN = try buf(T * c.hidden * f)
        Q = try buf(T * c.qDim * f)
        Kp = try buf(T * c.kvDim * f)
        Vp = try buf(T * c.kvDim * f)
        QR = try buf(T * c.qDim * 2)
        ATT = try buf(T * c.qDim * f)
        MID = try buf(T * c.denseFFN * f)
        SHMID = try buf(T * c.sharedFFN * f)
        SHARED = try buf(T * c.hidden * f)
        SCORES = try buf(T * c.experts * f)
        IDS = try buf(T * c.topK * 4, shared: true)
        WTS = try buf(T * c.topK * 4)
        HMID = try buf(T * c.topK * c.expertFFN * f)
        ROUTED = try buf(T * c.topK * c.hidden * f)
    }

    /// Hidden-state buffer after `run`; row `t` holds token `t`'s final residual stream.
    var hidden: MTLBuffer { H }

    // MARK: encoding helpers

    private func u32(_ e: MTLComputeCommandEncoder, _ v: Int, _ i: Int) {
        var x = UInt32(v)
        e.setBytes(&x, length: 4, index: i)
    }

    private func set(_ e: MTLComputeCommandEncoder, _ r: FlintBufferRef, _ i: Int) {
        e.setBuffer(r.buffer, offset: r.offset, index: i)
    }

    private func grid(_ x: Int, _ y: Int) -> MTLSize { MTLSize(width: x, height: y, depth: 1) }
    private var tg256: MTLSize { MTLSize(width: 256, height: 1, depth: 1) }

    private func gemm(_ e: MTLComputeCommandEncoder, _ m: FlintQMatrix, x: MTLBuffer, y: MTLBuffer, T: Int,
                      accumulate: Bool) {
        e.setComputePipelineState(pGemm)
        set(e, m.w, 0); set(e, m.s, 1); set(e, m.b, 2)
        e.setBuffer(x, offset: 0, index: 3)
        e.setBuffer(y, offset: 0, index: 4)
        u32(e, m.outFeatures, 5); u32(e, m.inFeatures, 6); u32(e, T, 7); u32(e, accumulate ? 1 : 0, 8)
        e.dispatchThreadgroups(grid((m.outFeatures + 7) / 8, (T + 7) / 8), threadsPerThreadgroup: tg256)
    }

    private func gateUp(_ e: MTLComputeCommandEncoder, _ g: FlintQMatrix, _ u: FlintQMatrix, x: MTLBuffer,
                        out: MTLBuffer, T: Int) {
        e.setComputePipelineState(pGateUp)
        set(e, g.w, 0); set(e, g.s, 1); set(e, g.b, 2)
        set(e, u.w, 3); set(e, u.s, 4); set(e, u.b, 5)
        e.setBuffer(x, offset: 0, index: 6)
        e.setBuffer(out, offset: 0, index: 7)
        u32(e, g.outFeatures, 8); u32(e, g.inFeatures, 9); u32(e, T, 10)
        e.dispatchThreadgroups(grid((g.outFeatures + 7) / 8, (T + 7) / 8), threadsPerThreadgroup: tg256)
    }

    private func norm(_ e: MTLComputeCommandEncoder, _ w: FlintBufferRef, c: FlintConfigView, T: Int) {
        e.setComputePipelineState(pNorm)
        e.setBuffer(H, offset: 0, index: 0)
        set(e, w, 1)
        e.setBuffer(XN, offset: 0, index: 2)
        u32(e, c.hidden, 3)
        var eps = c.eps
        e.setBytes(&eps, length: 4, index: 4)
        e.dispatchThreadgroups(grid(T, 1), threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    }

    private func commandBuffer() throws -> (MTLCommandBuffer, MTLComputeCommandEncoder) {
        guard let cb = context.queue.makeCommandBuffer(), let e = cb.makeComputeCommandEncoder() else {
            throw FlintRuntimeError.metal("command buffer")
        }
        return (cb, e)
    }

    private func commit(_ cb: MTLCommandBuffer, wait: Bool) throws {
        cb.commit()
        if wait {
            cb.waitUntilCompleted()
            if let error = cb.error { throw FlintRuntimeError.metal("\(error)") }
        }
    }

    // MARK: run

    /// Runs `ids` at positions `start ..< start + ids.count` through every layer, filling the KV caches.
    /// On return `hidden` holds each token's final residual (before the output norm).
    func run(ids: [Int], start: Int, layers: [FlintRunner.LayerView], embed: FlintQMatrix,
             c: FlintConfigView) throws {
        let T = ids.count
        precondition(T > 0 && T <= maxT)
        let tp = tokens.contents().assumingMemoryBound(to: UInt32.self)
        for (i, t) in ids.enumerated() { tp[i] = UInt32(t) }

        var (cb, e) = try commandBuffer()
        e.setComputePipelineState(pEmbed)
        set(e, embed.w, 0); set(e, embed.s, 1); set(e, embed.b, 2)
        e.setBuffer(H, offset: 0, index: 3)
        e.setBuffer(tokens, offset: 0, index: 4)
        u32(e, c.hidden, 5)
        e.dispatchThreads(grid(c.hidden, T), threadsPerThreadgroup: tg256)

        for L in layers {
            // Attention block.
            norm(e, L.inputNorm, c: c, T: T)
            gemm(e, L.q, x: XN, y: Q, T: T, accumulate: false)
            gemm(e, L.k, x: XN, y: Kp, T: T, accumulate: false)
            gemm(e, L.v, x: XN, y: Vp, T: T, accumulate: false)
            var qk = QKParams(numQHeads: UInt32(c.qHeads), numKVHeads: UInt32(c.kvHeads), headDim: UInt32(c.headDim),
                              basePosition: UInt32(start), ringCap: L.sliding ? UInt32(c.ringCap) : 0,
                              useRope: L.rope ? 1 : 0, theta: c.theta, eps: c.eps)
            e.setComputePipelineState(pQK)
            e.setBuffer(Q, offset: 0, index: 0); e.setBuffer(Kp, offset: 0, index: 1); e.setBuffer(Vp, offset: 0, index: 2)
            set(e, L.qNorm, 3); set(e, L.kNorm, 4)
            e.setBuffer(QR, offset: 0, index: 5)
            e.setBuffer(L.kCache, offset: 0, index: 6); e.setBuffer(L.vCache, offset: 0, index: 7)
            e.setBytes(&qk, length: MemoryLayout<QKParams>.stride, index: 8)
            e.dispatchThreadgroups(grid(c.qHeads + c.kvHeads, T),
                                   threadsPerThreadgroup: MTLSize(width: c.headDim, height: 1, depth: 1))
            var ap = AttnParams(numQHeads: UInt32(c.qHeads), numKVHeads: UInt32(c.kvHeads), headDim: UInt32(c.headDim),
                                basePosition: UInt32(start), window: L.sliding ? UInt32(c.window) : 0,
                                ringCap: L.sliding ? UInt32(c.ringCap) : 0,
                                scale: Float(1.0 / Double(c.headDim).squareRoot()))
            e.setComputePipelineState(pAttn)
            e.setBuffer(QR, offset: 0, index: 0)
            e.setBuffer(L.kCache, offset: 0, index: 1); e.setBuffer(L.vCache, offset: 0, index: 2)
            set(e, L.sinks, 3)
            e.setBuffer(ATT, offset: 0, index: 4)
            e.setBytes(&ap, length: MemoryLayout<AttnParams>.stride, index: 5)
            e.dispatchThreadgroups(grid(c.qHeads, T), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            gemm(e, L.o, x: ATT, y: H, T: T, accumulate: true)
            norm(e, L.postNorm, c: c, T: T)

            guard let experts = L.experts, let router = L.router else {
                gateUp(e, L.gate, L.up, x: XN, out: MID, T: T)
                gemm(e, L.down, x: MID, y: H, T: T, accumulate: true)
                continue
            }

            // Router for all rows, then the CPU groups (token, k) pairs by expert.
            e.setComputePipelineState(pScores)
            e.setBuffer(XN, offset: 0, index: 0); set(e, router, 1); e.setBuffer(SCORES, offset: 0, index: 2)
            u32(e, c.experts, 3); u32(e, c.hidden, 4)
            e.dispatchThreadgroups(grid((c.experts + 7) / 8, T), threadsPerThreadgroup: tg256)
            e.setComputePipelineState(pSelect)
            e.setBuffer(SCORES, offset: 0, index: 0); e.setBuffer(IDS, offset: 0, index: 1)
            e.setBuffer(WTS, offset: 0, index: 2)
            u32(e, c.experts, 3); u32(e, c.topK, 4)
            e.dispatchThreadgroups(grid(T, 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            // Shared expert rides in the same command buffer, ahead of the routed tiles.
            gateUp(e, L.gate, L.up, x: XN, out: SHMID, T: T)
            gemm(e, L.down, x: SHMID, y: SHARED, T: T, accumulate: false)
            e.endEncoding()
            try commit(cb, wait: true)

            let idp = IDS.contents().assumingMemoryBound(to: UInt32.self)
            var buckets: [Int: [UInt32]] = [:]
            for pair in 0..<(T * c.topK) { buckets[Int(idp[pair]), default: []].append(UInt32(pair)) }
            let order = buckets.keys.sorted()
            let tile = max(1, min(32, (experts.slotCount - 1) / 3))
            let tiles = stride(from: 0, to: order.count, by: tile).map { Array(order[$0..<min($0 + tile, order.count)]) }

            var inflight: [(cb: MTLCommandBuffer, slots: [Int])] = []
            var ticket = experts.request(tiles[0])
            for (ti, tileExperts) in tiles.enumerated() {
                let offsets = try experts.wait(ticket)
                let slots = ticket.slots
                // Free the slots of the tile from two steps back before choosing victims for the next one.
                if inflight.count >= 2 {
                    let old = inflight.removeFirst()
                    old.cb.waitUntilCompleted()
                    if let error = old.cb.error { throw FlintRuntimeError.metal("\(error)") }
                }
                if ti + 1 < tiles.count {
                    let avoid = Set(slots).union(inflight.flatMap(\.slots))
                    ticket = experts.request(tiles[ti + 1], avoiding: avoid)
                }

                var groups: [ExpertGroup] = []
                for (j, expert) in tileExperts.enumerated() {
                    let pairs = buckets[expert]!
                    for g in stride(from: 0, to: pairs.count, by: 8) {
                        var p = [UInt32](repeating: 0, count: 8)
                        let n = min(8, pairs.count - g)
                        for q in 0..<n { p[q] = pairs[g + q] }
                        groups.append(ExpertGroup(slotOffset: offsets[j], count: UInt32(n),
                                                  pairs: (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])))
                    }
                }
                expertGroups += groups.count
                guard let gbuf = context.device.makeBuffer(bytes: groups, length: groups.count * MemoryLayout<ExpertGroup>.stride,
                                                           options: .storageModeShared)
                else { throw FlintRuntimeError.metal("group buffer") }
                let (tcb, te) = try commandBuffer()
                var lay = experts.layout.offsets
                te.setComputePipelineState(pMoEGateUp)
                te.setBuffer(experts.pool, offset: 0, index: 0); te.setBuffer(gbuf, offset: 0, index: 1)
                te.setBuffer(XN, offset: 0, index: 2); te.setBuffer(HMID, offset: 0, index: 3)
                te.setBytes(&lay, length: lay.count * 4, index: 4)
                u32(te, c.expertFFN, 5); u32(te, c.hidden, 6); u32(te, c.topK, 7)
                te.dispatchThreadgroups(grid(c.expertFFN / 8, groups.count), threadsPerThreadgroup: tg256)
                te.setComputePipelineState(pMoEDown)
                te.setBuffer(experts.pool, offset: 0, index: 0); te.setBuffer(gbuf, offset: 0, index: 1)
                te.setBuffer(WTS, offset: 0, index: 2); te.setBuffer(HMID, offset: 0, index: 3)
                te.setBuffer(ROUTED, offset: 0, index: 4)
                te.setBytes(&lay, length: lay.count * 4, index: 5)
                u32(te, c.expertFFN, 6); u32(te, c.hidden, 7)
                te.dispatchThreadgroups(grid(c.hidden / 8, groups.count), threadsPerThreadgroup: tg256)
                te.endEncoding()
                try commit(tcb, wait: false)
                inflight.append((tcb, slots))
            }

            (cb, e) = try commandBuffer()
            e.setComputePipelineState(pReduce)
            e.setBuffer(H, offset: 0, index: 0); e.setBuffer(ROUTED, offset: 0, index: 1)
            e.setBuffer(SHARED, offset: 0, index: 2)
            u32(e, c.hidden, 3); u32(e, c.topK, 4)
            e.dispatchThreads(grid(c.hidden, T), threadsPerThreadgroup: tg256)
        }
        e.endEncoding()
        try commit(cb, wait: true)
    }
}

/// The handful of shape constants the prefill encoder needs, flattened from `FlintConfig`.
struct FlintConfigView {
    let hidden, qHeads, kvHeads, headDim, qDim, kvDim, denseFFN, sharedFFN, expertFFN, experts, topK: Int
    let window, ringCap: Int
    let theta, eps: Float
}
