#!/usr/bin/env bash
# Builds a shareable dist/VaaniStudio-<version>.dmg:
#   - Release build of Vaani.app (ad-hoc signed; not notarised)
#   - the engine's source and lockfiles, plus the `uv` installer, inside the app bundle,
#     so the app can install its own Python engine on first launch (no Homebrew, no Xcode)
#   - a drag-to-Applications window and a "How to open" note
#
# Requirements on the build machine: Xcode, xcodegen and uv (brew install xcodegen uv).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERSION="$(sed -n 's/^ *MARKETING_VERSION: "\(.*\)"/\1/p' app/project.yml | head -1)"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo dev)"
DIST="$ROOT/dist"
DMG="$DIST/VaaniStudio-$VERSION.dmg"
STAGE="$ROOT/build/dmg"

echo "==> Building Vaani $VERSION ($COMMIT)"
scripts/build_app.sh Release
APP="$ROOT/build/Vaani.app"
RES="$APP/Contents/Resources"

echo "==> Bundling the engine installer"
rm -rf "$RES/engine" "$RES/bin"
mkdir -p "$RES/engine" "$RES/bin"
rsync -a --exclude .venv --exclude __pycache__ --exclude .pytest_cache --exclude tests \
  "$ROOT/engine/" "$RES/engine/"
# Version marker: a hash of the bundled engine files, so the app re-runs setup only when the
# engine or its lockfiles actually change (app-only updates start straight away).
ENGINE_HASH="$(cd "$RES/engine" && find . -type f ! -name VERSION -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256 | cut -c1-12)"
echo "$VERSION+$ENGINE_HASH" > "$RES/engine/VERSION"
UV="$(python3 -c 'import os,shutil; print(os.path.realpath(shutil.which("uv")))')"
cp "$UV" "$RES/bin/uv"
chmod +x "$RES/bin/uv"
cp "$ROOT/LICENSE" "$ROOT/THIRD_PARTY_NOTICES.md" "$RES/" 2>/dev/null || true

echo "==> Signing (ad-hoc)"
codesign --force --sign - "$RES/bin/uv"
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP"

echo "==> Creating the disk image"
rm -rf "$STAGE" && mkdir -p "$STAGE" "$DIST"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
cp "$ROOT/scripts/dmg/How to open Vaani.txt" "$STAGE/"
rm -f "$DMG"
hdiutil create -volname "Vaani Studio $VERSION" -srcfolder "$STAGE" -ov -format UDZO -fs HFS+ "$DMG" >/dev/null
shasum -a 256 "$DMG" | tee "$DMG.sha256"
du -h "$DMG"
echo "Done: $DMG"
