// 사람 영역 가면(흰 = 사람) — 맥 내장 Vision 인물 분리. 사용: person_mask 입력.png 출력.png
// 종료: 0 = 가면 저장(사람이 없으면 전부 검은 가면일 수 있다) · 1 = 입력 오류 · 2 = 분리 실패 · 3 = 사람 없음 · 4 = 저장 실패 — 오류 글은 표준 오류로
import Foundation
import Vision
import CoreImage
import ImageIO
import UniformTypeIdentifiers

func fail(_ code: Int32, _ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(code)
}

let args = CommandLine.arguments
guard args.count == 3, let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: args[1]) as CFURL, nil),
      let image = CGImageSourceCreateImageAtIndex(src, 0, nil) else { fail(1, "입력 오류") }
let request = VNGeneratePersonSegmentationRequest()
request.qualityLevel = .accurate
request.outputPixelFormat = kCVPixelFormatType_OneComponent8
let handler = VNImageRequestHandler(cgImage: image, options: [:])
do { try handler.perform([request]) } catch { fail(2, "분리 실패 \(error)") }
guard let buffer = request.results?.first?.pixelBuffer else { fail(3, "사람 없음") }
var mask = CIImage(cvPixelBuffer: buffer)
let sx = CGFloat(image.width) / mask.extent.width, sy = CGFloat(image.height) / mask.extent.height
mask = mask.transformed(by: CGAffineTransform(scaleX: sx, y: sy))
let ctx = CIContext()
guard let out = ctx.createCGImage(mask, from: CGRect(x: 0, y: 0, width: image.width, height: image.height), format: .L8, colorSpace: CGColorSpaceCreateDeviceGray()),
      let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: args[2]) as CFURL, UTType.png.identifier as CFString, 1, nil) else { fail(4, "저장 실패") }
CGImageDestinationAddImage(dest, out, nil)
CGImageDestinationFinalize(dest)
print("ok \(image.width)x\(image.height)")
