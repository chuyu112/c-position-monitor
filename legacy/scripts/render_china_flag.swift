import AppKit

// Render the user's requested 🇨🇳 with the system's color emoji artwork.
let side = 512
let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: side,
    pixelsHigh: side, bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
    isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
let context = NSGraphicsContext(bitmapImageRep: bitmap)!
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = context
NSColor.clear.setFill()
NSRect(x: 0, y: 0, width: side, height: side).fill()
let font = NSFont(name: "Apple Color Emoji", size: 350)!
let text = NSAttributedString(string: "🇨🇳", attributes: [.font: font])
let size = text.size()
text.draw(at: NSPoint(x: (CGFloat(side) - size.width) / 2,
                     y: (CGFloat(side) - size.height) / 2))
NSGraphicsContext.restoreGraphicsState()
let output = URL(fileURLWithPath: CommandLine.arguments[1])
try FileManager.default.createDirectory(at: output.deletingLastPathComponent(),
                                      withIntermediateDirectories: true)
try bitmap.representation(using: .png, properties: [:])!.write(to: output)
print(output.path)
