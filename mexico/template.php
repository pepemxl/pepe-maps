<?php
// PWA City Map - Shared rendering template
// Expects $city array with: slug, name, subtitle, center [lat,lng], zoom, icon, attractions[]

if (!isset($city) || !is_array($city)) {
    http_response_code(500);
    exit('City config not provided');
}

header('Content-Type: text/html; charset=UTF-8');
header('Cache-Control: no-cache, must-revalidate');

$cityName     = $city['name']     ?? 'México';
$citySubtitle = $city['subtitle'] ?? '';
$citySlug     = $city['slug']     ?? 'mexico';
$cityCenter   = $city['center']   ?? [23.6345, -102.5528];
$cityZoom     = $city['zoom']     ?? 14;
$cityIcon     = $city['icon']     ?? '🏛️';
$attractions  = $city['attractions'] ?? [];

$attractionsJson = json_encode($attractions,  JSON_UNESCAPED_UNICODE);
$cityCenterJson  = json_encode($cityCenter);
$cityNameJs      = json_encode($cityName,     JSON_UNESCAPED_UNICODE);
$citySlugJs      = json_encode($citySlug,     JSON_UNESCAPED_UNICODE);
$cityIconJs      = json_encode($cityIcon,     JSON_UNESCAPED_UNICODE);
?>
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <meta name="theme-color" content="#1a0a2e">
    <meta name="description" content="Explora <?= htmlspecialchars($cityName) ?> - Guía turística interactiva con rutas a pie">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="<?= htmlspecialchars($cityName) ?>">

    <title><?= htmlspecialchars($cityName) ?> Explorer</title>

    <!-- PWA Manifest -->
    <link rel="manifest" href="#" id="manifest-link">

    <!-- Leaflet CSS -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet-routing-machine/3.2.12/leaflet-routing-machine.min.css">

    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Crimson+Pro:wght@300;400;600&display=swap" rel="stylesheet">

    <style>
        :root {
            --gold: #c9a84c;
            --gold-light: #e8c97a;
            --gold-dim: #8a6e2e;
            --deep: #0d0820;
            --midnight: #1a1035;
            --panel: rgba(13, 8, 32, 0.92);
            --surface: rgba(26, 16, 53, 0.85);
            --text: #f0e8d0;
            --text-dim: #a09078;
            --accent: #e85d3a;
            --teal: #2ab5a0;
            --font-serif: 'Playfair Display', Georgia, serif;
            --font-body: 'Crimson Pro', 'Times New Roman', serif;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: var(--font-body);
            background: var(--deep);
            color: var(--text);
            height: 100dvh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        /* ── HEADER ── */
        #header {
            position: relative;
            z-index: 1000;
            background: var(--panel);
            border-bottom: 1px solid var(--gold-dim);
            padding: 12px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            backdrop-filter: blur(20px);
            flex-shrink: 0;
            gap: 12px;
        }

        #header h1 {
            font-family: var(--font-serif);
            font-size: 1.3rem;
            color: var(--gold-light);
            letter-spacing: 0.05em;
            line-height: 1;
        }

        #header h1 span {
            display: block;
            font-size: 0.65rem;
            font-family: var(--font-body);
            font-weight: 300;
            color: var(--text-dim);
            letter-spacing: 0.2em;
            text-transform: uppercase;
            margin-top: 2px;
        }

        .header-actions { display: flex; gap: 8px; align-items: center; }

        .hbtn {
            background: none;
            border: 1px solid var(--gold-dim);
            color: var(--gold-light);
            padding: 8px 14px;
            border-radius: 4px;
            font-family: var(--font-body);
            font-size: 0.85rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s;
            text-decoration: none;
        }

        .hbtn:hover { background: rgba(201,168,76,0.1); border-color: var(--gold); }
        .hbtn.active { background: rgba(201,168,76,0.2); border-color: var(--gold); color: var(--gold); }

        /* ── MAP ── */
        #map { flex: 1; position: relative; z-index: 1; }
        .leaflet-tile-pane { filter: saturate(0.3) brightness(0.5) hue-rotate(200deg); }

        /* ── BOTTOM PANEL ── */
        #bottom-panel {
            position: relative;
            z-index: 1000;
            background: var(--panel);
            border-top: 1px solid var(--gold-dim);
            backdrop-filter: blur(20px);
            flex-shrink: 0;
            max-height: 45vh;
            transition: max-height 0.4s cubic-bezier(0.4,0,0.2,1);
        }

        #bottom-panel.collapsed { max-height: 64px; }

        #panel-toggle {
            width: 100%;
            padding: 14px 20px;
            background: none;
            border: none;
            color: var(--gold-light);
            font-family: var(--font-serif);
            font-size: 0.9rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: space-between;
            text-align: left;
        }

        #panel-toggle .chevron {
            transition: transform 0.3s;
            font-size: 0.7rem;
            color: var(--gold-dim);
        }

        #bottom-panel.collapsed .chevron { transform: rotate(180deg); }

        #attractions-list {
            display: flex;
            gap: 12px;
            padding: 0 16px 16px;
            overflow-x: auto;
            scrollbar-width: thin;
            scrollbar-color: var(--gold-dim) transparent;
        }

        #attractions-list::-webkit-scrollbar { height: 3px; }
        #attractions-list::-webkit-scrollbar-track { background: transparent; }
        #attractions-list::-webkit-scrollbar-thumb { background: var(--gold-dim); border-radius: 2px; }

        .attraction-card {
            flex-shrink: 0;
            width: 160px;
            background: var(--surface);
            border: 1px solid rgba(201,168,76,0.2);
            border-radius: 8px;
            padding: 12px;
            cursor: pointer;
            transition: all 0.25s;
            position: relative;
            overflow: hidden;
        }

        .attraction-card::before {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(135deg, rgba(201,168,76,0.05) 0%, transparent 100%);
        }

        .attraction-card:hover, .attraction-card.active {
            border-color: var(--gold);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(201,168,76,0.15);
        }

        .attraction-card.active { border-color: var(--teal); }

        .card-icon { font-size: 1.8rem; margin-bottom: 6px; }

        .card-name {
            font-family: var(--font-serif);
            font-size: 0.8rem;
            color: var(--text);
            line-height: 1.3;
            margin-bottom: 4px;
        }

        .card-category {
            font-size: 0.68rem;
            color: var(--gold-dim);
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }

        .card-rating {
            position: absolute;
            top: 10px;
            right: 10px;
            font-size: 0.7rem;
            color: var(--gold);
        }

        .card-route-btn {
            margin-top: 8px;
            width: 100%;
            padding: 5px 8px;
            background: rgba(42,181,160,0.15);
            border: 1px solid rgba(42,181,160,0.4);
            border-radius: 4px;
            color: var(--teal);
            font-family: var(--font-body);
            font-size: 0.72rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .card-route-btn:hover { background: rgba(42,181,160,0.3); }

        /* ── POPUP ── */
        .leaflet-popup-content-wrapper {
            background: var(--panel) !important;
            border: 1px solid var(--gold-dim) !important;
            border-radius: 8px !important;
            box-shadow: 0 20px 60px rgba(0,0,0,0.7) !important;
            backdrop-filter: blur(20px) !important;
            color: var(--text) !important;
            max-width: 300px !important;
            padding: 0 !important;
        }
        .leaflet-popup-tip { background: var(--midnight) !important; }
        .leaflet-popup-close-button { color: var(--gold-dim) !important; font-size: 18px !important; top: 10px !important; right: 12px !important; }

        .popup-content { padding: 18px; font-family: var(--font-body); }
        .popup-header {
            display: flex; align-items: flex-start; gap: 10px;
            margin-bottom: 12px; padding-bottom: 12px;
            border-bottom: 1px solid rgba(201,168,76,0.2);
        }
        .popup-icon { font-size: 2rem; flex-shrink: 0; }
        .popup-title { font-family: var(--font-serif); font-size: 1rem; color: var(--gold-light); line-height: 1.3; }
        .popup-category { font-size: 0.7rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.15em; margin-top: 2px; }
        .popup-desc { font-size: 0.9rem; color: var(--text); line-height: 1.6; margin-bottom: 10px; }
        .popup-section {
            background: rgba(255,255,255,0.03);
            border-left: 2px solid var(--gold-dim);
            padding: 8px 10px; margin-bottom: 8px; border-radius: 0 4px 4px 0;
        }
        .popup-section-title { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.15em; color: var(--gold-dim); margin-bottom: 4px; }
        .popup-section p { font-size: 0.82rem; color: var(--text-dim); line-height: 1.5; }
        .popup-meta {
            display: flex; justify-content: space-between; align-items: center;
            margin-top: 12px; padding-top: 10px;
            border-top: 1px solid rgba(201,168,76,0.1);
        }
        .popup-hours { font-size: 0.75rem; color: var(--text-dim); }
        .popup-rating { color: var(--gold); font-size: 0.85rem; }
        .popup-route-btn {
            width: 100%; margin-top: 12px; padding: 8px;
            background: linear-gradient(135deg, rgba(42,181,160,0.2), rgba(42,181,160,0.1));
            border: 1px solid rgba(42,181,160,0.5);
            border-radius: 5px;
            color: var(--teal);
            font-family: var(--font-body);
            font-size: 0.85rem;
            cursor: pointer; transition: all 0.2s;
            letter-spacing: 0.05em;
        }
        .popup-route-btn:hover { background: rgba(42,181,160,0.3); box-shadow: 0 4px 12px rgba(42,181,160,0.2); }

        /* ── MARKERS ── */
        .custom-marker {
            background: var(--midnight);
            border: 2px solid var(--gold);
            border-radius: 50%;
            width: 40px; height: 40px;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.2rem;
            box-shadow: 0 4px 16px rgba(201,168,76,0.3);
            cursor: pointer; transition: all 0.2s;
        }
        .custom-marker:hover { transform: scale(1.15); box-shadow: 0 6px 24px rgba(201,168,76,0.5); }
        .custom-marker.destination { border-color: var(--teal); box-shadow: 0 4px 16px rgba(42,181,160,0.4); animation: pulse-teal 2s infinite; }

        @keyframes pulse-teal {
            0%, 100% { box-shadow: 0 4px 16px rgba(42,181,160,0.4); }
            50% { box-shadow: 0 4px 32px rgba(42,181,160,0.8); }
        }

        .user-marker {
            background: var(--accent);
            border: 3px solid white;
            border-radius: 50%;
            width: 18px; height: 18px;
            box-shadow: 0 0 0 4px rgba(232,93,58,0.3);
            animation: pulse-user 2s infinite;
        }

        @keyframes pulse-user {
            0%, 100% { box-shadow: 0 0 0 4px rgba(232,93,58,0.3); }
            50% { box-shadow: 0 0 0 10px rgba(232,93,58,0.1); }
        }

        /* ── ROUTE INFO ── */
        #route-info {
            position: absolute; top: 70px; right: 12px; z-index: 900;
            background: var(--panel);
            border: 1px solid var(--teal);
            border-radius: 8px;
            padding: 14px 16px;
            backdrop-filter: blur(20px);
            display: none; min-width: 190px;
        }
        #route-info.visible { display: block; animation: slideIn 0.3s ease; }

        @keyframes slideIn {
            from { opacity: 0; transform: translateX(20px); }
            to { opacity: 1; transform: translateX(0); }
        }

        #route-info h3 { font-family: var(--font-serif); font-size: 0.85rem; color: var(--teal); margin-bottom: 8px; }
        #route-info .route-stat { display: flex; align-items: center; gap: 8px; font-size: 0.82rem; color: var(--text-dim); margin-bottom: 4px; }
        #route-info .route-stat strong { color: var(--text); }

        #clear-route {
            margin-top: 10px; width: 100%; padding: 6px;
            background: rgba(232,93,58,0.1);
            border: 1px solid rgba(232,93,58,0.3);
            border-radius: 4px;
            color: var(--accent);
            font-size: 0.75rem; cursor: pointer;
            font-family: var(--font-body); transition: all 0.2s;
        }
        #clear-route:hover { background: rgba(232,93,58,0.2); }

        /* ── TOAST ── */
        #toast {
            position: fixed;
            bottom: calc(45vh + 20px);
            left: 50%;
            transform: translateX(-50%) translateY(20px);
            background: var(--midnight);
            border: 1px solid var(--gold-dim);
            color: var(--text);
            padding: 10px 20px;
            border-radius: 6px;
            font-family: var(--font-body);
            font-size: 0.85rem;
            z-index: 2000;
            opacity: 0;
            transition: all 0.3s;
            white-space: nowrap;
            pointer-events: none;
        }
        #toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }

        /* ── LOADING ── */
        #loading {
            position: fixed; inset: 0;
            background: var(--deep);
            z-index: 9999;
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            gap: 20px;
        }
        #loading h2 { font-family: var(--font-serif); font-size: 2rem; color: var(--gold-light); font-style: italic; }
        #loading p { font-family: var(--font-body); color: var(--text-dim); font-size: 0.9rem; letter-spacing: 0.2em; text-transform: uppercase; }

        .spinner {
            width: 40px; height: 40px;
            border: 2px solid rgba(201,168,76,0.1);
            border-top-color: var(--gold);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        .leaflet-routing-container { display: none !important; }
    </style>
</head>
<body>

<div id="loading">
    <div class="spinner"></div>
    <h2><?= htmlspecialchars($cityName) ?></h2>
    <p>Cargando el mapa</p>
</div>

<header id="header">
    <h1>
        <?= htmlspecialchars($cityName) ?> Explorer
        <?php if ($citySubtitle): ?><span><?= htmlspecialchars($citySubtitle) ?></span><?php endif; ?>
    </h1>
    <div class="header-actions">
        <a href="../" class="hbtn" title="Todas las ciudades">🇲🇽 Ciudades</a>
        <button id="gps-btn" class="hbtn" onclick="centerOnUser()">📍 Mi ubicación</button>
    </div>
</header>

<div id="map"></div>

<div id="route-info">
    <h3>🚶 Ruta a pie</h3>
    <div class="route-stat">📍 Distancia: <strong id="route-dist">—</strong></div>
    <div class="route-stat">⏱️ Tiempo: <strong id="route-time">—</strong></div>
    <div class="route-stat">📌 Destino: <strong id="route-dest">—</strong></div>
    <button id="clear-route" onclick="clearRoute()">✕ Limpiar ruta</button>
</div>

<div id="toast"></div>

<div id="bottom-panel">
    <button id="panel-toggle" onclick="togglePanel()">
        <span>🗺️ Atracciones Turísticas (<?= count($attractions) ?>)</span>
        <span class="chevron">▲</span>
    </button>
    <div id="attractions-list"></div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet-routing-machine/3.2.12/leaflet-routing-machine.min.js"></script>

<script>
const ATTRACTIONS = <?= $attractionsJson ?>;
const CITY_CENTER = <?= $cityCenterJson ?>;
const CITY_ZOOM   = <?= (int)$cityZoom ?>;
const CITY_NAME   = <?= $cityNameJs ?>;
const CITY_SLUG   = <?= $citySlugJs ?>;
const CITY_ICON   = <?= $cityIconJs ?>;

let map, userMarker, userLatLng = null, routingControl = null;
let markers = {}, activeCard = null, panelCollapsed = false;

function initMap() {
    map = L.map('map', {
        center: CITY_CENTER,
        zoom: CITY_ZOOM,
        zoomControl: false,
        attributionControl: false
    });

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', { maxZoom: 19 }).addTo(map);
    L.control.zoom({ position: 'topright' }).addTo(map);

    ATTRACTIONS.forEach(a => {
        const icon = L.divIcon({
            className: '',
            html: `<div class="custom-marker" id="marker-${a.id}">${a.icon}</div>`,
            iconSize: [40, 40],
            iconAnchor: [20, 20],
            popupAnchor: [0, -24]
        });

        const marker = L.marker([a.lat, a.lng], { icon })
            .addTo(map)
            .bindPopup(buildPopup(a), { maxWidth: 300, minWidth: 260 });

        marker.on('click', () => highlightCard(a.id));
        markers[a.id] = marker;
    });

    buildCards();
    getUserLocation();

    setTimeout(() => {
        document.getElementById('loading').style.opacity = '0';
        setTimeout(() => document.getElementById('loading').remove(), 400);
    }, 800);
}

function buildPopup(a) {
    const stars = '★'.repeat(Math.floor(a.rating)) + (a.rating % 1 >= 0.5 ? '½' : '');
    return `
        <div class="popup-content">
            <div class="popup-header">
                <div class="popup-icon">${a.icon}</div>
                <div>
                    <div class="popup-title">${a.name}</div>
                    <div class="popup-category">${a.category}</div>
                </div>
            </div>
            <p class="popup-desc">${a.description}</p>
            ${a.history ? `<div class="popup-section"><div class="popup-section-title">📜 Historia</div><p>${a.history}</p></div>` : ''}
            ${a.tips ? `<div class="popup-section"><div class="popup-section-title">💡 Consejo del viajero</div><p>${a.tips}</p></div>` : ''}
            <div class="popup-meta">
                <span class="popup-hours">🕐 ${a.hours || 'Consultar horario'}</span>
                <span class="popup-rating">${stars} ${a.rating}</span>
            </div>
            <button class="popup-route-btn" onclick="routeTo(${a.id})">
                🚶 Trazar ruta caminando aquí
            </button>
        </div>
    `;
}

function buildCards() {
    const list = document.getElementById('attractions-list');
    ATTRACTIONS.forEach(a => {
        const card = document.createElement('div');
        card.className = 'attraction-card';
        card.id = `card-${a.id}`;
        card.innerHTML = `
            <div class="card-rating">★ ${a.rating}</div>
            <div class="card-icon">${a.icon}</div>
            <div class="card-name">${a.name}</div>
            <div class="card-category">${a.category}</div>
            <button class="card-route-btn" onclick="event.stopPropagation(); routeTo(${a.id})">
                🚶 Ir caminando
            </button>
        `;
        card.addEventListener('click', () => {
            map.setView([a.lat, a.lng], Math.max(CITY_ZOOM, 17), { animate: true });
            markers[a.id].openPopup();
            highlightCard(a.id);
        });
        list.appendChild(card);
    });
}

function highlightCard(id) {
    if (activeCard) document.getElementById(`card-${activeCard}`)?.classList.remove('active');
    document.getElementById(`card-${id}`)?.classList.add('active');
    document.getElementById(`card-${id}`)?.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
    activeCard = id;
}

function getUserLocation(showToastMsg = false) {
    if (!navigator.geolocation) {
        showToastMsg && toast('Geolocalización no disponible');
        return;
    }
    navigator.geolocation.getCurrentPosition(pos => {
        userLatLng = L.latLng(pos.coords.latitude, pos.coords.longitude);
        placeUserMarker();
        if (showToastMsg) {
            map.setView(userLatLng, 17, { animate: true });
            toast('📍 Ubicación actualizada');
        }
        document.getElementById('gps-btn').classList.add('active');
    }, err => {
        userLatLng = L.latLng(CITY_CENTER[0], CITY_CENTER[1]);
        placeUserMarker();
        showToastMsg && toast(`Usando ubicación aproximada de ${CITY_NAME}`);
    }, { enableHighAccuracy: true, timeout: 10000 });
}

function placeUserMarker() {
    if (userMarker) userMarker.remove();
    const icon = L.divIcon({
        className: '',
        html: `<div class="user-marker"></div>`,
        iconSize: [18, 18],
        iconAnchor: [9, 9]
    });
    userMarker = L.marker(userLatLng, { icon, zIndexOffset: 1000 })
        .addTo(map)
        .bindTooltip('📍 Tú estás aquí', { permanent: false, direction: 'top' });
}

function centerOnUser() {
    if (userLatLng) {
        map.setView(userLatLng, 17, { animate: true });
        toast('📍 Centrando en tu ubicación');
    } else {
        getUserLocation(true);
    }
}

function routeTo(id) {
    const attraction = ATTRACTIONS.find(a => a.id === id);
    if (!attraction) return;

    if (!userLatLng) {
        toast('⚠️ Primero activa tu ubicación GPS');
        getUserLocation(true);
        return;
    }

    if (routingControl) { map.removeControl(routingControl); routingControl = null; }

    Object.values(markers).forEach(m => {
        const el = m.getElement()?.querySelector('.custom-marker');
        if (el) el.classList.remove('destination');
    });

    const destMarkerEl = document.getElementById(`marker-${id}`);
    if (destMarkerEl) destMarkerEl.classList.add('destination');

    routingControl = L.Routing.control({
        waypoints: [
            L.latLng(userLatLng.lat, userLatLng.lng),
            L.latLng(attraction.lat, attraction.lng)
        ],
        router: L.Routing.osrmv1({
            serviceUrl: 'https://router.project-osrm.org/route/v1',
            profile: 'foot'
        }),
        lineOptions: {
            styles: [
                { color: '#2ab5a0', weight: 5, opacity: 0.8 },
                { color: '#1a7a6e', weight: 7, opacity: 0.3 }
            ],
            addWaypoints: false
        },
        createMarker: () => null,
        fitSelectedRoutes: true,
        showAlternatives: false
    }).addTo(map);

    routingControl.on('routesfound', function(e) {
        const route = e.routes[0];
        const dist = (route.summary.totalDistance / 1000).toFixed(2);
        const mins = Math.ceil(route.summary.totalTime / 60);
        document.getElementById('route-dist').textContent = dist + ' km';
        document.getElementById('route-time').textContent = mins + ' min caminando';
        document.getElementById('route-dest').textContent = attraction.name;
        document.getElementById('route-info').classList.add('visible');
        highlightCard(id);
        toast(`🚶 Ruta a ${attraction.name} calculada`);
    });

    routingControl.on('routingerror', function() {
        toast('❌ No se pudo calcular la ruta. Verifica tu conexión.');
    });

    map.closePopup();
}

function clearRoute() {
    if (routingControl) { map.removeControl(routingControl); routingControl = null; }
    document.getElementById('route-info').classList.remove('visible');
    Object.values(markers).forEach(m => {
        const el = m.getElement()?.querySelector('.custom-marker');
        if (el) el.classList.remove('destination');
    });
    toast('Ruta eliminada');
}

function togglePanel() {
    panelCollapsed = !panelCollapsed;
    document.getElementById('bottom-panel').classList.toggle('collapsed', panelCollapsed);
}

function toast(msg) {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 3000);
}

(function() {
    const iconSvg = `data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect width="100" height="100" fill="%231a1035"/><text y="70" x="50" text-anchor="middle" font-size="60">${encodeURIComponent(CITY_ICON)}</text></svg>`;
    const manifest = {
        name: `${CITY_NAME} Explorer`,
        short_name: CITY_NAME,
        description: `Guía turística interactiva de ${CITY_NAME} con rutas a pie`,
        start_url: '.',
        display: 'standalone',
        background_color: '#0d0820',
        theme_color: '#1a0a2e',
        icons: [
            { src: iconSvg, sizes: '192x192', type: 'image/svg+xml' },
            { src: iconSvg, sizes: '512x512', type: 'image/svg+xml' }
        ]
    };
    const blob = new Blob([JSON.stringify(manifest)], { type: 'application/json' });
    document.getElementById('manifest-link').href = URL.createObjectURL(blob);
})();

if ('serviceWorker' in navigator) {
    const swCode = `
        const CACHE = '${CITY_SLUG}-explorer-v1';
        const ASSETS = [location.href];
        self.addEventListener('install', e => e.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS))));
        self.addEventListener('fetch', e => e.respondWith(caches.match(e.request).then(r => r || fetch(e.request))));
    `;
    const blob = new Blob([swCode], { type: 'application/javascript' });
    navigator.serviceWorker.register(URL.createObjectURL(blob)).catch(() => {});
}

window.addEventListener('DOMContentLoaded', initMap);
</script>
</body>
</html>
