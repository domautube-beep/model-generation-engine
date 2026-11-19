// 맥 내장 Vision 전신 관절 + 손 영역 좌표 추출 (타인 전신·팔·손·발 100% 완전 소거용). 사용: body_joints 사진경로
import Foundation
import Vision
import ImageIO

func fail(_ code: Int32, _ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(code)
}

guard let path = CommandLine.arguments.dropFirst().first else { fail(64, "사용법: body_joints 사진") }
let url = URL(fileURLWithPath: path)
guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
      let width = props[kCGImagePropertyPixelWidth] as? Double,
      let height = props[kCGImagePropertyPixelHeight] as? Double else { fail(65, "사진 읽기 실패") }

let bodyReq = VNDetectHumanBodyPoseRequest()
let handReq = VNDetectHumanHandPoseRequest()
handReq.maximumHandCount = 4
do {
    try VNImageRequestHandler(url: url, options: [:]).perform([bodyReq, handReq])
} catch {
    fail(70, "관절 검출 실패: \(error)")
}

var bodies: [[[String: Any]]] = []
for obs in (bodyReq.results ?? []) {
    if let pts = try? obs.recognizedPoints(.all) {
        var joints: [[String: Any]] = []
        for (k, pt) in pts where pt.confidence >= 0.05 {
            let px: Double = Double(pt.location.x) * width
            let py: Double = (1.0 - Double(pt.location.y)) * height
            joints.append(["name": k.rawValue.rawValue, "x": px, "y": py, "conf": Double(pt.confidence)])
        }
        if !joints.isEmpty { bodies.append(joints) }
    }
}

var hands: [[[Double]]] = []
for obs in (handReq.results ?? []) {
    if let pts = try? obs.recognizedPoints(.all) {
        var hpts: [[Double]] = []
        for (_, pt) in pts where pt.confidence >= 0.15 {
            let px: Double = Double(pt.location.x) * width
            let py: Double = (1.0 - Double(pt.location.y)) * height
            hpts.append([px, py])
        }
        if !hpts.isEmpty { hands.append(hpts) }
    }
}

let out: [String: Any] = ["bodies": bodies, "hands": hands]
print(String(data: try JSONSerialization.data(withJSONObject: out, options: []), encoding: .utf8)!)
