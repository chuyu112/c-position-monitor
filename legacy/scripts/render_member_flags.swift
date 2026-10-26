import AppKit

// Use the same system color emoji artwork as the reviewed portrait badges.
let output = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
for (code, emoji) in [("CN", "🇨🇳"), ("US", "🇺🇸"), ("JP", "🇯🇵"), ("KR", "🇰🇷"), ("TH", "🇹🇭")] {
    let side = 512
    let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: side,
        pixelsHigh: side, bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
        isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmap)!
    NSColor.clear.setFill()
    NSRect(x: 0, y: 0, width: side, height: side).fill()
    let text = NSAttributedString(string: emoji, attributes: [.font: NSFont(name: "Apple Color Emoji", size: 350)!])
    let size = text.size()
    text.draw(at: NSPoint(x: (CGFloat(side) - size.width) / 2, y: (CGFloat(side) - size.height) / 2))
    NSGraphicsContext.restoreGraphicsState()
    try bitmap.representation(using: .png, properties: [:])!.write(to: output.appendingPathComponent("\(code).png"))
}
print("Rendered five flag assets.")
