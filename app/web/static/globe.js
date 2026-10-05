/*
 * Viewer configuration adapted from God's Eye View by Bilawal Sidhu.
 * https://github.com/bilawalsidhu/gods-eye-view
 * MIT License — Copyright (c) 2026 Bilawal Sidhu
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to
 * do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * Reference geography: Natural Earth via world-atlas 2.0.2, not military control.
 */
const mount = document.getElementById('globe-widget');
const motion = matchMedia('(prefers-reduced-motion: reduce)');
const places = [
  {id:'moscow', name:'Москва', lon:37.62, lat:55.75},
  {id:'beijing', name:'Пекин', lon:116.4, lat:39.9},
  {id:'gba', name:'Greater Bay Area', lon:113.6, lat:22.6},
  {id:'rostov', name:'Ростов-на-Дону', lon:39.7, lat:47.2},
  {id:'tehran', name:'Тегеран', lon:51.4, lat:35.7},
];
let viewer, C, loading, rotate, credit, observer, intersection;
let visible=false, intersecting=true, rotating=!motion.matches, initializing;
let lastFrame=0, initialView=true;
const markers = new Map();

function script(src) {
  return new Promise((resolve,reject) => {
    const el=document.createElement('script'); el.src=src; el.onload=resolve;
    el.onerror=()=>reject(new Error('Local map asset unavailable')); document.head.append(el);
  });
}
function style(href) {
  const el=document.createElement('link'); el.rel='stylesheet'; el.href=href; document.head.append(el);
}
function synchronize() {
  if (!viewer || viewer.isDestroyed()) return;
  const active=visible && intersecting && !document.hidden;
  viewer.useDefaultRenderLoop=active;
  viewer.scene.requestRenderMode=!rotating;
  if(active) { viewer.resize(); viewer.scene.requestRender(); }
  mount.dataset.rendering=String(active);
  mount.dataset.rotating=String(rotating);
  rotate.textContent=rotating?'Ⅱ':'▷';
  rotate.setAttribute('aria-label',rotating?'Остановить вращение планеты':'Вращать планету');
  lastFrame=0;
}
function home() {
  initialView=true;
  const width=Math.max(mount.clientWidth,100), height=Math.max(mount.clientHeight-22,100);
  const fov=C.Math.toRadians(60);
  const vertical=width>height?2*Math.atan(Math.tan(fov/2)*height/width):fov;
  const radius=Math.min(width*.43,height*.44);
  const angle=Math.atan(2*radius/height*Math.tan(vertical/2));
  const earthRadius=C.Ellipsoid.WGS84.maximumRadius;
  const distance=earthRadius/Math.sin(angle)-earthRadius;
  viewer.camera.setView({destination:C.Cartesian3.fromDegrees(73,29,distance),
    orientation:{heading:0,pitch:-C.Math.PI_OVER_TWO,roll:0}});
  viewer.scene.requestRender();
}
function select(region) {
  for(const [id,point] of markers) {
    point.pixelSize=id===region?8:4;
    point.color=C.Color.fromCssColorString(id===region?'#f2e7b4':'#b3e6d8');
  }
  if(viewer) viewer.scene.requestRender();
}
function trace(geometry, collection, width, color, height=13000) {
  for(const coords of geometry.coordinates) {
    if(coords.length<2) continue;
    collection.add({positions:coords.map(([lon,lat])=>C.Cartesian3.fromDegrees(lon,lat,height)),
      width,material:C.Material.fromType('Color',{color:C.Color.fromCssColorString(color)})});
  }
}
function texture(countries,borders) {
  const canvas=document.createElement('canvas'); canvas.width=2048; canvas.height=1024;
  const ctx=canvas.getContext('2d');
  ctx.fillStyle='#06151d'; ctx.fillRect(0,0,2048,1024);
  const line=ring=>{
    for(let i=0;i<ring.length;i++) {
      const [lon,lat]=ring[i], x=(lon+180)*2048/360, y=(90-lat)*1024/180;
      if(!i)ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
  };
  ctx.fillStyle='#12343a';
  for(const country of countries.features) {
    const g=country.geometry;
    for(const polygon of g.type==='Polygon'?[g.coordinates]:g.coordinates) {
      ctx.beginPath(); for(const ring of polygon){line(ring);ctx.closePath();} ctx.fill('evenodd');
    }
  }
  ctx.strokeStyle='#529f9440'; ctx.lineWidth=.55; ctx.beginPath();
  for(const ring of borders.coordinates)line(ring); ctx.stroke();
  return canvas.toDataURL('image/png');
}
async function initialize() {
  style('/static/vendor/cesium/Widgets/widgets.css'); style('/static/globe.css');
  mount.innerHTML='<div class="pars-globe-canvas"></div><span class="pars-globe-loading" role="status">Собираю географию…</span><div class="pars-globe-controls"><button type="button" class="pars-globe-rotate" aria-label="Остановить вращение планеты">Ⅱ</button><button type="button" class="pars-globe-home" aria-label="Общий вид планеты">↺</button></div><div class="pars-globe-credit"></div>';
  loading=mount.querySelector('.pars-globe-loading'); rotate=mount.querySelector('.pars-globe-rotate');
  credit=mount.querySelector('.pars-globe-credit');
  window.CESIUM_BASE_URL='/static/vendor/cesium/';
  await Promise.all([script('/static/vendor/cesium/Cesium.js'),script('/static/vendor/topojson-client.min.js')]);
  C=window.Cesium; C.Ion.defaultAccessToken='';
  viewer=new C.Viewer(mount.querySelector('.pars-globe-canvas'),{
    timeline:false,animation:false,baseLayerPicker:false,geocoder:false,homeButton:false,
    sceneModePicker:false,navigationHelpButton:false,fullscreenButton:false,vrButton:false,
    selectionIndicator:false,infoBox:false,baseLayer:false,scene3DOnly:true,
    creditContainer:credit,terrainProvider:new C.EllipsoidTerrainProvider(),
    targetFrameRate:24,maximumRenderTimeChange:Infinity,msaaSamples:2,
    contextOptions:{webgl:{alpha:true,preserveDrawingBuffer:false}},
  });
  // Cesium's default uses CSS pixel ratio; cap rather than rendering full Retina resolution.
  viewer.resolutionScale=Math.min(devicePixelRatio||1,1.5);
  const scene=viewer.scene;
  scene.backgroundColor=C.Color.TRANSPARENT;
  scene.postProcessStages.fxaa.enabled=true;
  scene.skyBox.show=false;scene.sun.show=false;scene.moon.show=false;
  scene.skyAtmosphere.show=false;scene.fog.enabled=false;
  scene.globe.enableLighting=false;scene.globe.showGroundAtmosphere=false;
  scene.globe.depthTestAgainstTerrain=true;scene.globe.tileCacheSize=16;
  scene.globe.maximumScreenSpaceError=3;
  scene.globe.baseColor=C.Color.fromCssColorString('#06151d');
  scene.globe.material=new C.Material({fabric:{type:'OSVEDWidgetRim',source:`
    czm_material czm_getMaterial(czm_materialInput materialInput) {
      czm_material m=czm_getDefaultMaterial(materialInput);
      vec3 p=(czm_inverseView*vec4(-materialInput.positionToEyeEC,1.0)).xyz;
      vec3 n=czm_viewRotation*normalize(p*czm_ellipsoidInverseRadii*czm_ellipsoidInverseRadii);
      float facing=clamp(dot(n,normalize(materialInput.positionToEyeEC)),0.0,1.0);
      m.diffuse=vec3(0.08,0.30,0.28);m.alpha=pow(1.0-facing,4.5)*0.65;return m;
    }`},translucent:false});
  const controls=scene.screenSpaceCameraController;
  controls.enableTilt=false;controls.inertiaSpin=0;controls.inertiaZoom=0;
  controls.minimumZoomDistance=6000000;controls.maximumZoomDistance=32000000;
  home();
  const response=await fetch('/static/vendor/countries-110m.json');
  if(!response.ok)throw new Error('Geography unavailable');
  const world=await response.json(), topo=window.topojson;
  const countries=topo.feature(world,world.objects.countries);
  const borders=topo.mesh(world,world.objects.countries,(a,b)=>a!==b);
  const coasts=topo.mesh(world,world.objects.countries,(a,b)=>a===b);
  viewer.imageryLayers.addImageryProvider(await C.SingleTileImageryProvider.fromUrl(texture(countries,borders),{
    credit:new C.Credit('Natural Earth · справочная география',true),
  }));
  const lines=scene.primitives.add(new C.PolylineCollection());
  trace(coasts,lines,.9,'#6ccdc0dc');trace(borders,lines,.55,'#59ada384');
  const grid=scene.primitives.add(new C.PolylineCollection());
  const gridGeometry={coordinates:[]};
  for(let lat=-80;lat<=80;lat+=10)gridGeometry.coordinates.push(Array.from({length:181},(_,i)=>[-180+i*2,lat]));
  for(let lon=-180;lon<180;lon+=10)gridGeometry.coordinates.push(Array.from({length:90},(_,i)=>[lon,-89+i*2]));
  trace(gridGeometry,grid,.5,'#6bb9ab19',7000);
  const points=scene.primitives.add(new C.PointPrimitiveCollection());
  const labels=scene.primitives.add(new C.LabelCollection());
  for(const place of places) {
    const position=C.Cartesian3.fromDegrees(place.lon,place.lat,26000);
    markers.set(place.id,points.add({id:place.id,position,pixelSize:4,
      color:C.Color.fromCssColorString('#b3e6d8'),outlineWidth:3,
      outlineColor:C.Color.fromCssColorString('#62bbaa50')}));
    // Show labels only at close zoom to keep a small widget legible.
    labels.add({id:place.id,position,text:place.name,font:'10px sans-serif',
      fillColor:C.Color.fromCssColorString('#b3d4cb'),outlineColor:C.Color.BLACK,outlineWidth:2,
      style:C.LabelStyle.FILL_AND_OUTLINE,pixelOffset:new C.Cartesian2(7,-7),
      distanceDisplayCondition:new C.DistanceDisplayCondition(0,8000000),
      horizontalOrigin:C.HorizontalOrigin.LEFT});
  }
  const picker=new C.ScreenSpaceEventHandler(scene.canvas);
  picker.setInputAction(event=>{
    const picked=scene.pick(event.position,12,12);
    if(places.some(p=>p.id===picked?.id)) {
      select(picked.id);window.dispatchEvent(new CustomEvent('pars-region-select',{detail:{region:picked.id}}));
    }
  },C.ScreenSpaceEventType.LEFT_CLICK);
  const pause=()=>{initialView=false;rotating=false;synchronize();};
  scene.canvas.addEventListener('pointerdown',pause);
  scene.canvas.addEventListener('wheel',pause,{passive:true});
  rotate.addEventListener('click',()=>{rotating=!rotating;synchronize();});
  mount.querySelector('.pars-globe-home').addEventListener('click',()=>{home();synchronize();});
  scene.preRender.addEventListener(()=>{
    const now=performance.now(), delta=lastFrame?Math.min((now-lastFrame)/1000,.12):0;
    if(rotating && visible && !document.hidden)viewer.camera.rotate(C.Cartesian3.UNIT_Z,-C.Math.toRadians(.55)*delta);
    lastFrame=now;
  });
  scene.renderError.addEventListener(()=>{loading.hidden=false;loading.textContent='Карта недоступна. Фильтры ниже продолжают работать.';});
  observer=new ResizeObserver(()=>{if(!viewer.isDestroyed()){viewer.resize();if(initialView)home();}});
  observer.observe(mount);
  intersection=new IntersectionObserver(entries=>{intersecting=entries[0].isIntersecting;synchronize();});
  intersection.observe(mount);
  loading.hidden=true;mount.dataset.loaded='true';
  mount.dataset.countries=String(countries.features.length);
  synchronize();
  window.addEventListener('pagehide',()=>{
    observer.disconnect();intersection.disconnect();picker.destroy();viewer.destroy();
  },{once:true});
}
window.addEventListener('pars-globe-visibility',event=>{
  visible=Boolean(event.detail?.visible);
  if(visible && !initializing)initializing=initialize().catch(()=>{
    if(viewer&&!viewer.isDestroyed())viewer.destroy();
    if(loading){loading.hidden=false;loading.textContent='Карта недоступна. Используй фильтры регионов ниже.';}
  });
  synchronize();
});
window.addEventListener('pars-region-set',event=>{if(viewer)select(event.detail?.region);});
document.addEventListener('visibilitychange',synchronize);
motion.addEventListener('change',()=>{if(motion.matches){rotating=false;synchronize();}});
