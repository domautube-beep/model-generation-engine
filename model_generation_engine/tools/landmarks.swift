// 맥 내장 Vision 으로 얼굴 윤곽점을 찾는다(설치 없음). 사용: landmarks 사진
// 출력: 한 줄 JSON {"contour": [[x,y]…], "leftPupil": …, "rightPupil": …, "leftEye": …, "rightEye": …, "nose": …} — 픽셀 좌표, 왼쪽 위 기준 · 가장 큰 얼굴 하나
// 형님 10/25 밤 「왜 얼굴이 갑자기 넙대대해지는거지」 — 결과 얼굴 비율을 원본과 재기 위해 쓴다(그림을 고치지 않는다).
// 종료: 0 = 정상(얼굴이 없으면 {}) · 64 = 인자 없음 · 65 = 사진을 못 읽음 · 70 = Vision 요청 실패 — 고장은 표준 오류에 한국어 한 줄(앱이 「얼굴 없음」과 가른다)
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
guard let path = CommandLine.arguments.dropFirst().first else { fail(64, "사용법: landmarks 사진") }
let url = URL(fileURLWithPath: path)
guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
      let width = props[kCGImagePropertyPixelWidth] as? Double,
      let height = props[kCGImagePropertyPixelHeight] as? Double else { fail(65, "사진을 읽지 못했습니다: " + path) }
let request = VNDetectFaceLandmarksRequest()
do { try VNImageRequestHandler(url: url, options: [:]).perform([request]) } catch { fail(70, "얼굴 점 찾기 요청이 실패했습니다: \(error)") }
if let face = (request.results ?? []).max(by: { $0.boundingBox.width < $1.boundingBox.width }), let marks = face.landmarks {
    let box = face.boundingBox
    func points(_ region: VNFaceLandmarkRegion2D?) -> [[Double]] {
        guard let region = region else { return [] }
        var result: [[Double]] = []
        for p in region.normalizedPoints {
            let nx: Double = box.minX + Double(p.x) * box.width
            let ny: Double = box.minY + Double(p.y) * box.height
            result.append([nx * width, (1.0 - ny) * height])
        }
        return result
    }
    out = ["contour": points(marks.faceContour), "leftPupil": points(marks.leftPupil), "rightPupil": points(marks.rightPupil),
           "leftEye": points(marks.leftEye), "rightEye": points(marks.rightEye), "nose": points(marks.nose),
           "noseCrest": points(marks.noseCrest), "leftEyebrow": points(marks.leftEyebrow), "rightEyebrow": points(marks.rightEyebrow),
           "outerLips": points(marks.outerLips), "innerLips": points(marks.innerLips)]
}
print(String(data: try JSONSerialization.data(withJSONObject: out, options: []), encoding: .utf8)!)
