import AppKit
import Foundation
import Vision

guard CommandLine.arguments.count >= 2 else {
    fputs("Usage: text_boxes <image_path>\n", stderr)
    exit(2)
}

let path = CommandLine.arguments[1]
guard let nsImage = NSImage(contentsOfFile: path),
      let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    fputs("Failed to load image: \(path)\n", stderr)
    exit(1)
}

let req = VNRecognizeTextRequest()
req.recognitionLevel = .accurate
req.usesLanguageCorrection = false

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
try handler.perform([req])

var boxes: [[String: Any]] = []
for obs in req.results ?? [] {
    let b = obs.boundingBox
    let topText = obs.topCandidates(1).first?.string ?? ""
    boxes.append([
        "x": b.origin.x,
        "y": 1.0 - b.origin.y - b.size.height,
        "w": b.size.width,
        "h": b.size.height,
        "text": topText,
    ])
}

let data = try JSONSerialization.data(withJSONObject: ["boxes": boxes], options: [])
FileHandle.standardOutput.write(data)
