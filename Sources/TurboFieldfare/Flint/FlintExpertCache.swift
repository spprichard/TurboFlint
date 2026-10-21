import Darwin
import Foundation
import Metal
import TurboFieldfareFormat

/// Byte layout of one routed expert inside a cache slot: gate, up, down projections, each as
/// packed int4 weights followed by BF16 scales and biases. Matches `FlintExpertLayout` in flint.metal.
package struct FlintExpertSlotLayout {
    package var offsets: [UInt32] = []  // gate_w, gate_s, gate_b, up_w, up_s, up_b, down_w, down_s, down_b
    package var sizes: [Int] = []
    package var slotBytes = 0
}

package struct FlintExpertCacheStats {
    package var hits = 0
    package var misses = 0
    package var bytesRead = 0
    package var readSeconds = 0.0
}

/// Per-layer routed-expert cache. Slots live in one page-aligned, no-copy Metal buffer; a miss fills
/// its slot with nine `pread`s straight from the expert tensors in Ollama's blobs.
/// Eviction is least-frequently-used per expert, least-recently-used on ties.
package final class FlintExpertCache: @unchecked Sendable {
    private struct Segment {
        let fd: Int32
        let base: UInt64      // file offset of expert 0
        let stride: UInt64    // bytes per expert
        let size: Int
        let dst: Int          // offset within the slot
    }

    package let layer: Int
    package let slotCount: Int
    package let layout: FlintExpertSlotLayout
    package let pool: MTLBuffer
    private let poolBase: UnsafeMutableRawPointer
    private let segments: [Segment]
    private var slotExpert: [Int]
    private var slotLastUse: [UInt64]
    private var expertSlot: [Int: Int] = [:]
    private var expertUses: [UInt32]
    private var clock: UInt64 = 0
    package private(set) var stats = FlintExpertCacheStats()

    package init(layer: Int, checkpoint: FlintOllamaCheckpoint, device: MTLDevice, slotCount: Int,
                 fds: inout [URL: Int32]) throws {
        self.layer = layer
        self.slotCount = slotCount
        let prefix = "model.layers.\(layer).mlp.experts."
        var segs: [Segment] = []
        var layout = FlintExpertSlotLayout()
        var cursor = 0
        var experts = 0
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            let q = try checkpoint.quantized(prefix + proj + ".weight")
            experts = q.expertCount ?? 0
            for loc in [q.weight, q.scale, q.bias] {
                let fd: Int32
                if let existing = fds[loc.file] {
                    fd = existing
                } else {
                    fd = open(loc.file.path, O_RDONLY)
                    guard fd >= 0 else { throw FlintRuntimeError.posix("open expert blob", errno) }
                    fds[loc.file] = fd
                }
                let stride = loc.byteCount / UInt64(experts)
                segs.append(Segment(fd: fd, base: loc.offset, stride: stride, size: Int(stride), dst: cursor))
                layout.offsets.append(UInt32(cursor))
                layout.sizes.append(Int(stride))
                cursor += (Int(stride) + 255) / 256 * 256
            }
        }
        let page = Int(getpagesize())
        layout.slotBytes = (cursor + page - 1) / page * page
        self.layout = layout
        self.segments = segs

        let total = layout.slotBytes * slotCount
        var raw: UnsafeMutableRawPointer?
        guard posix_memalign(&raw, max(page, 1 << 21), total) == 0, let raw else {
            throw FlintRuntimeError.posix("posix_memalign", errno)
        }
        nonisolated(unsafe) let captured = raw
        guard let buf = device.makeBuffer(bytesNoCopy: raw, length: total, options: .storageModeShared,
                                          deallocator: { _, _ in free(captured) })
        else {
            free(raw)
            throw FlintRuntimeError.metal("cannot wrap expert slot pool")
        }
        self.pool = buf
        self.poolBase = raw
        self.slotExpert = Array(repeating: -1, count: slotCount)
        self.slotLastUse = Array(repeating: 0, count: slotCount)
        self.expertUses = Array(repeating: 0, count: experts)
    }

    /// Returns the slot byte offset of each requested expert, loading misses from disk first.
    package func acquire(_ experts: [Int], readConcurrency: Int) throws -> [UInt32] {
        clock += 1
        var slots = [Int](repeating: -1, count: experts.count)
        var pinned = Set<Int>()
        for (i, e) in experts.enumerated() {
            expertUses[e] &+= 1
            if let s = expertSlot[e] {
                slots[i] = s
                pinned.insert(s)
                slotLastUse[s] = clock
                stats.hits += 1
            }
        }
        var loads: [(slot: Int, expert: Int)] = []
        for (i, e) in experts.enumerated() where slots[i] < 0 {
            let victim = chooseVictim(excluding: pinned)
            if slotExpert[victim] >= 0 { expertSlot[slotExpert[victim]] = nil }
            slotExpert[victim] = e
            expertSlot[e] = victim
            slotLastUse[victim] = clock
            pinned.insert(victim)
            slots[i] = victim
            loads.append((victim, e))
            stats.misses += 1
        }
        if !loads.isEmpty { try load(loads, concurrency: readConcurrency) }
        return slots.map { UInt32($0 * layout.slotBytes) }
    }

    private func chooseVictim(excluding pinned: Set<Int>) -> Int {
        var best = -1
        for s in 0..<slotCount where !pinned.contains(s) {
            if slotExpert[s] < 0 { return s }
            if best < 0 { best = s; continue }
            let u = expertUses[slotExpert[s]], bu = expertUses[slotExpert[best]]
            if u < bu || (u == bu && slotLastUse[s] < slotLastUse[best]) { best = s }
        }
        precondition(best >= 0, "expert cache needs more slots than experts per token")
        return best
    }

    private func load(_ loads: [(slot: Int, expert: Int)], concurrency: Int) throws {
        let start = DispatchTime.now().uptimeNanoseconds
        let jobs = loads.flatMap { l in segments.map { (l.slot, l.expert, $0) } }
        let failures = UnsafeMutablePointer<Int32>.allocate(capacity: jobs.count)
        failures.initialize(repeating: 0, count: jobs.count)
        defer { failures.deallocate() }
        nonisolated(unsafe) let base = poolBase
        nonisolated(unsafe) let fail = failures
        let slotBytes = layout.slotBytes
        let lanes = max(1, min(concurrency, jobs.count))
        DispatchQueue.concurrentPerform(iterations: lanes) { lane in
            var j = lane
            while j < jobs.count {
                let (slot, expert, seg) = jobs[j]
                let dst = base.advanced(by: slot * slotBytes + seg.dst)
                var done = 0
                let off = off_t(seg.base + seg.stride * UInt64(expert))
                while done < seg.size {
                    let n = pread(seg.fd, dst.advanced(by: done), seg.size - done, off + off_t(done))
                    if n <= 0 { fail[j] = n == 0 ? EIO : errno; break }
                    done += n
                }
                j += lanes
            }
        }
        if let bad = (0..<jobs.count).first(where: { failures[$0] != 0 }) {
            for l in loads { expertSlot[l.expert] = nil; slotExpert[l.slot] = -1 }
            throw FlintRuntimeError.posix("pread expert (layer \(layer))", failures[bad])
        }
        stats.bytesRead += loads.count * segments.reduce(0) { $0 + $1.size }
        stats.readSeconds += Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
    }
}
