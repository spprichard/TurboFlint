import Darwin
import Foundation
import Metal
import TurboFieldfareFormat

/// A tensor's position inside a Metal buffer.
package struct FlintBufferRef {
    package let buffer: MTLBuffer
    package let offset: Int
}

/// An affine-quantized matrix: packed weights plus BF16 scale and bias, all in one mapped blob.
package struct FlintQMatrix {
    package let w: FlintBufferRef
    package let s: FlintBufferRef
    package let b: FlintBufferRef
    package let outFeatures: Int
    package let inFeatures: Int
}

package enum FlintRuntimeError: Error, CustomStringConvertible {
    case posix(String, Int32)
    case metal(String)
    case invalid(String)

    package var description: String {
        switch self {
        case let .posix(call, err): "\(call) failed: \(String(cString: strerror(err)))"
        case let .metal(what): "Metal: \(what)"
        case let .invalid(what): what
        }
    }
}

/// Maps the resident (non-expert) tensor blobs of the Ollama checkpoint into no-copy Metal buffers.
/// Each blob file is mapped once, read-only, from offset 0 so the mapping is page aligned.
package final class FlintWeightStore {
    package let checkpoint: FlintOllamaCheckpoint
    private let device: MTLDevice
    private var mapped: [URL: MTLBuffer] = [:]
    package private(set) var mappedBytes = 0

    package init(checkpoint: FlintOllamaCheckpoint, device: MTLDevice) {
        self.checkpoint = checkpoint
        self.device = device
    }

    private func buffer(for file: URL) throws -> MTLBuffer {
        if let b = mapped[file] { return b }
        let fd = open(file.path, O_RDONLY)
        guard fd >= 0 else { throw FlintRuntimeError.posix("open \(file.lastPathComponent)", errno) }
        defer { close(fd) }
        var st = stat()
        guard fstat(fd, &st) == 0 else { throw FlintRuntimeError.posix("fstat", errno) }
        let page = Int(getpagesize())
        let length = (Int(st.st_size) + page - 1) / page * page
        guard let base = mmap(nil, length, PROT_READ, MAP_PRIVATE, fd, 0), base != MAP_FAILED else {
            throw FlintRuntimeError.posix("mmap \(file.lastPathComponent)", errno)
        }
        _ = posix_madvise(base, length, POSIX_MADV_WILLNEED)
        nonisolated(unsafe) let captured = base
        guard let buf = device.makeBuffer(bytesNoCopy: base, length: length, options: .storageModeShared,
                                          deallocator: { _, _ in munmap(captured, length) })
        else {
            munmap(base, length)
            throw FlintRuntimeError.metal("cannot wrap \(file.lastPathComponent)")
        }
        mapped[file] = buf
        mappedBytes += length
        return buf
    }

    package func ref(_ name: String) throws -> FlintBufferRef {
        let loc = try checkpoint.tensor(name)
        return FlintBufferRef(buffer: try buffer(for: loc.file), offset: Int(loc.offset))
    }

    package func matrix(_ name: String) throws -> FlintQMatrix {
        let q = try checkpoint.quantized(name)
        guard q.weight.offset % 2 == 0 else { throw FlintRuntimeError.invalid("\(name): weight not 2-aligned") }
        return FlintQMatrix(
            w: FlintBufferRef(buffer: try buffer(for: q.weight.file), offset: Int(q.weight.offset)),
            s: FlintBufferRef(buffer: try buffer(for: q.scale.file), offset: Int(q.scale.offset)),
            b: FlintBufferRef(buffer: try buffer(for: q.bias.file), offset: Int(q.bias.offset)),
            outFeatures: q.outFeatures, inFeatures: q.inFeatures)
    }

    /// Touches every mapped page so the first token does not pay for faulting in the resident core.
    package func prefault() {
        for buf in mapped.values {
            let p = buf.contents().assumingMemoryBound(to: UInt8.self)
            let page = Int(getpagesize())
            var sink: UInt8 = 0
            var i = 0
            while i < buf.length { sink &+= p[i]; i += page }
            withExtendedLifetime(sink) {}
        }
    }
}
