// 맥 내장 Vision 으로 사진 속 얼굴 칸을 찾는다(설치 없음). 사용: faces 사진1 [사진2 …]
// 출력: 한 줄 JSON {"경로": [[x0,y0,x1,y1,신뢰도], …], …} — 픽셀 좌표, 왼쪽 위 기준
// 종료: 0 = 정상(얼굴이 없으면 빈 목록) · 64 = 인자 없음 · 65 = 사진을 못 읽음 · 70 = Vision 요청 실패 — 고장은 표준 오류에 한국어 한 줄(앱이 「얼굴 없음」과 가른다)
// 스위프트 쓰는 법: 한 식에 map/클로저와 여러 산술을 같이 쓰지 않는다 · 중간값마다 let x: Double = … 로 타입을 적는다 · for 로 푼다
// (2026-11-18 회사 맥 Swift 6.1.2 가 한 줄 map 식을 「reasonable time 안에 타입 검사 못 함」으로 거절했다)
import Foundation
import Vision
import ImageIO

func fail(_ code: Int32, _ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(code)
}

var out: [String: [[Double]]] = [:]
let paths = Array(CommandLine.arguments.dropFirst())
if paths.isEmpty { fail(64, "사용법: faces 사진1 [사진2 …]") }
for path in paths {
    let url = URL(fileURLWithPath: path)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
          let width = props[kCGImagePropertyPixelWidth] as? Double,
          let height = props[kCGImagePropertyPixelHeight] as? Double else { fail(65, "사진을 읽지 못했습니다: " + path) }
    let request = VNDetectFaceRectanglesRequest()
    let handler = VNImageRequestHandler(url: url, options: [:])
    do { try handler.perform([request]) } catch { fail(70, "얼굴 찾기 요청이 실패했습니다: \(error)") }
    var boxes: [[Double]] = []
    for face in request.results ?? [] {
        let b = face.boundingBox
        let x0: Double = b.minX * width
        let y0: Double = (1.0 - b.maxY) * height
        let x1: Double = b.maxX * width
        let y1: Double = (1.0 - b.minY) * height
        boxes.append([x0, y0, x1, y1, Double(face.confidence)])
    }
    out[path] = boxes
}
let data = try JSONSerialization.data(withJSONObject: out, options: [])
print(String(data: data, encoding: .utf8)!)
