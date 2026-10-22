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
    /// Requested experts already resident.
    package var hits = 0
    /// Requested experts that a prefetch had loaded or started loading.
    package var prefetchHits = 0
    /// Requested experts that had to be read on demand.
    package var misses = 0
    /// Prefetched experts evicted before any token used them.
    package var prefetchWasted = 0
    package var bytesRead = 0
    /// Time the decode thread spent blocked on expert reads.
    package var waitSeconds = 0.0

    package static func + (a: Self, b: Self) -> Self {
        Self(hits: a.hits + b.hits, prefetchHits: a.prefetchHits + b.prefetchHits, misses: a.misses + b.misses,
             prefetchWasted: a.prefetchWasted + b.prefetchWasted, bytesRead: a.bytesRead + b.bytesRead,
             waitSeconds: a.waitSeconds + b.waitSeconds)
    }
}

/// Bounded pool for blocking expert reads. `pread` blocks its thread, so the reads run on an
/// OperationQueue rather than the Swift Concurrency cooperative pool, which they would starve.
package final class FlintIOQueue: @unchecked Sendable {
    let queue: OperationQueue

    package init(concurrency: Int) {
        queue = OperationQueue()
        queue.name = "turboflint.expert-io"
        queue.maxConcurrentOperationCount = max(1, concurrency)
        queue.qualityOfService = .userInitiated
    }
}

/// One in-flight expert load. The loader threads write `error` before leaving `group`; the decode
/// thread reads it only after `group.wait()`, which orders the accesses.
private final class FlintLoad: @unchecked Sendable {
    let group = DispatchGroup()
    var error: Int32 = 0
}

/// Per-layer routed-expert cache. Slots live in one page-aligned, no-copy Metal buffer; a miss fills
/// its slot with nine `pread`s straight from the expert tensors in Ollama's blobs. Loads are
/// asynchronous, so the next layer's predicted experts can stream in while this layer computes.
/// Eviction is least-frequently-used per expert, least-recently-used on ties.
/// All bookkeeping runs on the decode thread; only slot bytes are written by I/O threads.
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
    private let expertBytes: Int
    private let io: FlintIOQueue
    private var slotExpert: [Int]
    private var slotLastUse: [UInt64]
    private var slotLoad: [FlintLoad?]
    private var slotPrefetched: [Bool]
    private var expertSlot: [Int: Int] = [:]
    private var expertUses: [UInt32]
    private var clock: UInt64 = 0
    package private(set) var stats = FlintExpertCacheStats()

    package init(layer: Int, checkpoint: FlintOllamaCheckpoint, device: MTLDevice, slotCount: Int,
                 io: FlintIOQueue, fds: inout [URL: Int32]) throws {
        self.layer = layer
        self.slotCount = slotCount
        self.io = io
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
        self.expertBytes = segs.reduce(0) { $0 + $1.size }

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
        self.slotLoad = Array(repeating: nil, count: slotCount)
        self.slotPrefetched = Array(repeating: false, count: slotCount)
        self.expertUses = Array(repeating: 0, count: experts)
    }

    /// Pending result of `request`: the slot of each expert, plus the loads to wait for.
    package struct Ticket {
        fileprivate let slots: [Int]
        fileprivate let loads: [FlintLoad]
    }

    /// Starts loading any of `experts` that are not resident and returns without waiting.
    package func request(_ experts: [Int]) -> Ticket {
        clock += 1
        var slots = [Int](repeating: -1, count: experts.count)
        var pinned = Set<Int>()
        var waits: [FlintLoad] = []
        for (i, e) in experts.enumerated() {
            expertUses[e] &+= 1
            guard let s = expertSlot[e] else { continue }
            slots[i] = s
            pinned.insert(s)
            slotLastUse[s] = clock
            if slotPrefetched[s] {
                slotPrefetched[s] = false
                stats.prefetchHits += 1
            } else {
                stats.hits += 1
            }
            if let load = slotLoad[s] { waits.append(load) }
        }
        for (i, e) in experts.enumerated() where slots[i] < 0 {
            let s = assign(e, excluding: &pinned, priority: .high)
            slots[i] = s
            waits.append(slotLoad[s]!)
            stats.misses += 1
        }
        return Ticket(slots: slots, loads: waits)
    }

    /// Blocks until every load in `ticket` finished; returns slot byte offsets for the kernels.
    package func wait(_ ticket: Ticket) throws -> [UInt32] {
        let start = DispatchTime.now().uptimeNanoseconds
        for load in ticket.loads { load.group.wait() }
        stats.waitSeconds += Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        for s in ticket.slots {
            guard let load = slotLoad[s] else { continue }
            slotLoad[s] = nil
            if load.error != 0 {
                expertSlot[slotExpert[s]] = nil
                slotExpert[s] = -1
                throw FlintRuntimeError.posix("pread expert (layer \(layer))", load.error)
            }
        }
        return ticket.slots.map { UInt32($0 * layout.slotBytes) }
    }

    /// Starts background loads for experts this layer is likely to need next. Never blocks, and never
    /// evicts a slot that is still loading.
    package func prefetch(_ experts: [Int]) {
        var pinned = Set<Int>()
        for e in experts {
            if let s = expertSlot[e] { pinned.insert(s) }
        }
        for e in experts where expertSlot[e] == nil {
            guard hasEvictableSlot(excluding: pinned) else { return }
            let s = assign(e, excluding: &pinned, priority: .normal)
            slotPrefetched[s] = true
        }
    }

    private func hasEvictableSlot(excluding pinned: Set<Int>) -> Bool {
        (0..<slotCount).contains { !pinned.contains($0) && isSettled($0) }
    }

    /// True when slot `s` has no load in flight. Finished loads that nobody waited on (mispredicted
    /// prefetches) are retired here; a failed one drops its expert mapping so it is read again.
    private func isSettled(_ s: Int) -> Bool {
        guard let load = slotLoad[s] else { return true }
        guard load.group.wait(timeout: .now()) == .success else { return false }
        slotLoad[s] = nil
        if load.error != 0, slotExpert[s] >= 0 {
            expertSlot[slotExpert[s]] = nil
            slotExpert[s] = -1
            slotPrefetched[s] = false
        }
        return true
    }

    /// Picks a victim slot for `expert`, records the mapping, and queues its nine reads.
    private func assign(_ expert: Int, excluding pinned: inout Set<Int>,
                        priority: Operation.QueuePriority) -> Int {
        let victim = chooseVictim(excluding: pinned)
        if slotExpert[victim] >= 0 {
            expertSlot[slotExpert[victim]] = nil
            if slotPrefetched[victim] { stats.prefetchWasted += 1 }
        }
        slotExpert[victim] = expert
        expertSlot[expert] = victim
        slotLastUse[victim] = clock
        slotPrefetched[victim] = false
        pinned.insert(victim)

        let load = FlintLoad()
        slotLoad[victim] = load
        nonisolated(unsafe) let dstBase = poolBase.advanced(by: victim * layout.slotBytes)
        for seg in segments {
            load.group.enter()
            let op = BlockOperation {
                let dst = dstBase.advanced(by: seg.dst)
                let off = off_t(seg.base + seg.stride * UInt64(expert))
                var done = 0
                while done < seg.size {
                    let n = pread(seg.fd, dst.advanced(by: done), seg.size - done, off + off_t(done))
                    if n <= 0 { load.error = n == 0 ? EIO : errno; break }
                    done += n
                }
                load.group.leave()
            }
            op.queuePriority = priority
            io.queue.addOperation(op)
        }
        stats.bytesRead += expertBytes
        return victim
    }

    private func chooseVictim(excluding pinned: Set<Int>) -> Int {
        var best = -1
        for s in 0..<slotCount where !pinned.contains(s) && isSettled(s) {
            if slotExpert[s] < 0 { return s }
            if best < 0 { best = s; continue }
            let u = expertUses[slotExpert[s]], bu = expertUses[slotExpert[best]]
            if u < bu || (u == bu && slotLastUse[s] < slotLastUse[best]) { best = s }
        }
        precondition(best >= 0, "expert cache needs more free slots than experts per token")
        return best
    }
}
