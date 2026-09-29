#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/project-env.sh"
cd "$PROJECT_ROOT"
APP="$PROJECT_ROOT/Prompt Lens.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" "$PROJECT_ROOT/.cache/swift"
/usr/bin/xcrun swiftc -parse-as-library -O -target arm64-apple-macos14.0 \
  -module-cache-path "$PROJECT_ROOT/.cache/swift" \
  -framework SwiftUI -framework AppKit \
  "$PROJECT_ROOT/app/macos/PromptLens.swift" -o "$APP/Contents/MacOS/PromptLens"
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleExecutable</key><string>PromptLens</string>
<key>CFBundleIdentifier</key><string>local.promptlens.desktop</string>
<key>CFBundleName</key><string>Prompt Lens</string>
<key>CFBundleDisplayName</key><string>Prompt Lens</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>0.1.0</string>
<key>CFBundleVersion</key><string>1</string>
<key>LSMinimumSystemVersion</key><string>14.0</string>
<key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
printf 'Built %s\n' "$APP"
