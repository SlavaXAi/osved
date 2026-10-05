# Local map assets

The dashboard never sends the private feed, profile, or selected region to a map API.
Assets are served locally and loaded only when the political-economy folder opens.

- Cesium 1.138.0: copied from the existing verified globe development dependency,
  `cesium/Build/Cesium`. Apache-2.0; see `CESIUM-LICENSE.md` and upstream notices.
- topojson-client 3.1.0: local distribution, ISC license in `TOPOJSON-LICENSE`.
- world-atlas 2.0.2 `countries-110m.json`: Natural Earth public-domain geography,
  https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-110m.json.
  Reference outlines are not a live front line or a claim about disputed sovereignty.

Viewer attribution and MIT notice for God's Eye View are in `../globe.js`.

Public distribution note: the bundled Cesium SDK demonstration Ion token is removed from Cesium.js, index.js and index.cjs. The local dashboard uses no Ion token. Upstream license notices are retained.
