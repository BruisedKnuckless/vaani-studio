#!/usr/bin/env bash
# Builds Vaani.app (ad-hoc signed, for this Mac) into build/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${1:-Release}"

cd "$ROOT/app"
xcodegen generate --quiet
xcodebuild -project Vaani.xcodeproj -scheme Vaani -configuration "$CONFIG" \
  -derivedDataPath "$ROOT/build/DerivedData" -destination 'platform=macOS' \
  CODE_SIGN_IDENTITY=- build | xcbeautify 2>/dev/null || \
xcodebuild -project Vaani.xcodeproj -scheme Vaani -configuration "$CONFIG" \
  -derivedDataPath "$ROOT/build/DerivedData" -destination 'platform=macOS' \
  CODE_SIGN_IDENTITY=- -quiet build

APP="$ROOT/build/DerivedData/Build/Products/$CONFIG/Vaani.app"
rm -rf "$ROOT/build/Vaani.app"
cp -R "$APP" "$ROOT/build/Vaani.app"
echo "Built $ROOT/build/Vaani.app"
