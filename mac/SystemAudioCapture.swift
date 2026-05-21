import AVFoundation
import CoreMedia
import Foundation
import ScreenCaptureKit

final class SystemAudioCapture: NSObject, SCStreamOutput, SCStreamDelegate {
    private let outputHandle = FileHandle.standardOutput
    private var didLogFormat = false

    func stream(
        _ stream: SCStream,
        didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of outputType: SCStreamOutputType
    ) {
        guard outputType == .audio, sampleBuffer.isValid else {
            return
        }
        handleAudio(sampleBuffer)
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        FileHandle.standardError.write(Data("[SystemAudioCapture] stream stopped: \(error)\n".utf8))
        exit(1)
    }

    private func handleAudio(_ sampleBuffer: CMSampleBuffer) {
        let frameCount = CMSampleBufferGetNumSamples(sampleBuffer)
        guard frameCount > 0 else {
            return
        }

        do {
            try sampleBuffer.withAudioBufferList { audioBufferList, _ in
                if !didLogFormat {
                    if let description = sampleBuffer.formatDescription?.audioStreamBasicDescription {
                        FileHandle.standardError.write(
                            Data(
                                "[SystemAudioCapture] format sampleRate=\(description.mSampleRate) channels=\(description.mChannelsPerFrame) flags=\(description.mFormatFlags)\n".utf8
                            )
                        )
                    }
                    didLogFormat = true
                }

                var mixed = [Float](repeating: 0, count: frameCount)
                var channelTotal = 0

                for audioBuffer in audioBufferList {
                    guard let rawData = audioBuffer.mData else {
                        continue
                    }

                    let channelCount = max(1, Int(audioBuffer.mNumberChannels))
                    let sampleCount = Int(audioBuffer.mDataByteSize) / MemoryLayout<Float>.size
                    let samples = rawData.assumingMemoryBound(to: Float.self)

                    if channelCount > 1 {
                        let frames = min(frameCount, sampleCount / channelCount)
                        for frame in 0..<frames {
                            var sum: Float = 0
                            for channel in 0..<channelCount {
                                sum += samples[frame * channelCount + channel]
                            }
                            mixed[frame] += sum / Float(channelCount)
                        }
                    } else {
                        let frames = min(frameCount, sampleCount)
                        for frame in 0..<frames {
                            mixed[frame] += samples[frame]
                        }
                    }

                    channelTotal += 1
                }

                guard channelTotal > 0 else {
                    return
                }

                if channelTotal > 1 {
                    for index in mixed.indices {
                        mixed[index] /= Float(channelTotal)
                    }
                }

                mixed.withUnsafeBufferPointer { pointer in
                    guard let baseAddress = pointer.baseAddress else {
                        return
                    }
                    outputHandle.write(
                        Data(
                            bytes: baseAddress,
                            count: pointer.count * MemoryLayout<Float>.size
                        )
                    )
                }
            }
        } catch {
            FileHandle.standardError.write(Data("[SystemAudioCapture] sample failed: \(error)\n".utf8))
        }
    }
}

@main
struct App {
    static func main() async {
        do {
            guard #available(macOS 13.0, *) else {
                throw NSError(
                    domain: "SystemAudioCapture",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "ScreenCaptureKit audio capture requires macOS 13 or later."]
                )
            }

            let capture = SystemAudioCapture()
            let content = try await SCShareableContent.excludingDesktopWindows(
                false,
                onScreenWindowsOnly: true
            )
            guard let display = content.displays.first else {
                throw NSError(
                    domain: "SystemAudioCapture",
                    code: 2,
                    userInfo: [NSLocalizedDescriptionKey: "No display is available for ScreenCaptureKit."]
                )
            }

            let filter = SCContentFilter(display: display, excludingWindows: [])
            let configuration = SCStreamConfiguration()
            configuration.capturesAudio = true
            configuration.excludesCurrentProcessAudio = true
            configuration.sampleRate = 16_000
            configuration.channelCount = 1
            configuration.width = 2
            configuration.height = 2
            configuration.minimumFrameInterval = CMTime(value: 1, timescale: 1)

            let stream = SCStream(filter: filter, configuration: configuration, delegate: capture)
            try stream.addStreamOutput(
                capture,
                type: .audio,
                sampleHandlerQueue: DispatchQueue(label: "system-audio-capture.audio")
            )
            try await stream.startCapture()
            FileHandle.standardError.write(Data("[SystemAudioCapture] started\n".utf8))
            while true {
                try await Task.sleep(nanoseconds: 1_000_000_000)
            }
        } catch {
            FileHandle.standardError.write(Data("[SystemAudioCapture] failed: \(error)\n".utf8))
            exit(1)
        }
    }
}
