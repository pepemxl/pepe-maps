<?php
// Country-wide explorer: Mexico map with one marker per state. Clicking a
// state opens a side panel listing its cities (clickable to each city page).

$cities = require __DIR__ . '/../cities.php';

header('Content-Type: text/html; charset=UTF-8');

// Group cities by state, computing each state's centroid as the average of
// its cities' map centers — good enough for state-level pins.
$states = [];
foreach ($cities as $slug => $c) {
    $stateSlug = $c['state_slug'] ?? 'sin-estado';
    $stateName = $c['state']      ?? 'Sin estado';
    if (!isset($states[$stateSlug])) {
        $states[$stateSlug] = [
            'slug'   => $stateSlug,
            'name'   => $stateName,
            'cities' => [],
            'sumLat' => 0.0,
            'sumLng' => 0.0,
        ];
    }
    [$lat, $lng] = $c['center'];
    $states[$stateSlug]['cities'][] = [
        'slug'      => $slug,
        'name'      => $c['name'],
        'subtitle'  => $c['subtitle'] ?? '',
        'icon'      => $c['icon'] ?? '🏛️',
        'attrCount' => count($c['attractions'] ?? []),
        'lat'       => $lat,
        'lng'       => $lng,
    ];
    $states[$stateSlug]['sumLat'] += $lat;
    $states[$stateSlug]['sumLng'] += $lng;
}
foreach ($states as &$s) {
    $n = count($s['cities']);
    $s['lat'] = $s['sumLat'] / $n;
    $s['lng'] = $s['sumLng'] / $n;
    unset($s['sumLat'], $s['sumLng']);
}
unset($s);

uasort($states, fn($a, $b) => strcoll($a['name'], $b['name']));

$totalCities      = count($cities);
$totalAttractions = 0;
foreach ($cities as $c) { $totalAttractions += count($c['attractions'] ?? []); }
?>
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="theme-color" content="#1a0a2e">
<title>Pepe Maps · Explorador por estados</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Crimson+Pro:wght@300;400;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<style>
    :root {
        --gold: #c9a84c;
        --gold-light: #e8c97a;
        --gold-dim: #8a6e2e;
        --deep: #0d0820;
        --midnight: #1a1035;
        --panel: rgba(13, 8, 32, 0.94);
        --surface: rgba(26, 16, 53, 0.85);
        --text: #f0e8d0;
        --text-dim: #a09078;
        --teal: #2ab5a0;
        --font-serif: 'Playfair Display', Georgia, serif;
        --font-body: 'Crimson Pro', 'Times New Roman', serif;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    html, body { height: 100%; }
    body {
        font-family: var(--font-body);
        background: var(--deep);
        color: var(--text);
        overflow: hidden;
    }
    .topbar {
        position: fixed; top: 0; left: 0; right: 0; z-index: 1100;
        display: flex; align-items: center; gap: 14px;
        padding: 14px 20px;
        background: var(--panel);
        border-bottom: 1px solid var(--gold-dim);
    }
    .topbar h1 {
        font-family: var(--font-serif);
        font-size: 1.2rem; font-style: italic;
        color: var(--gold-light); letter-spacing: 0.05em;
        flex: 1; min-width: 0;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .topbar h1 .accent { color: var(--teal); font-style: normal; font-weight: 700; }
    .topbar a.back {
        color: var(--teal); text-decoration: none;
        font-size: 0.78rem; letter-spacing: 0.15em; text-transform: uppercase;
        border: 1px solid rgba(42,181,160,0.4);
        padding: 6px 12px; border-radius: 3px; transition: all 0.2s;
    }
    .topbar a.back:hover { background: rgba(42,181,160,0.15); border-color: var(--teal); }
    .topbar .stats {
        font-size: 0.78rem; letter-spacing: 0.1em;
        color: var(--gold); text-transform: uppercase;
    }

    #map {
        position: absolute; inset: 56px 0 0 0;
        background: var(--deep);
    }
    .leaflet-tile-pane { filter: saturate(0.3) brightness(0.5) hue-rotate(200deg); }

    /* State pin */
    .state-pin {
        background: rgba(13,8,32,0.92);
        border: 1px solid var(--gold);
        color: var(--gold-light);
        font-family: var(--font-serif);
        font-size: 0.9rem; font-style: italic;
        padding: 6px 12px; border-radius: 18px;
        white-space: nowrap;
        box-shadow: 0 4px 14px rgba(0,0,0,0.5);
        transition: all 0.2s;
        cursor: pointer;
    }
    .state-pin:hover {
        background: var(--gold);
        color: var(--deep);
        transform: scale(1.06);
    }
    .state-pin .badge {
        display: inline-block;
        margin-left: 6px;
        font-family: var(--font-body);
        font-style: normal;
        font-size: 0.72rem;
        color: var(--teal);
        background: rgba(42,181,160,0.15);
        border: 1px solid rgba(42,181,160,0.4);
        padding: 1px 6px; border-radius: 10px;
    }
    .state-pin:hover .badge {
        color: var(--deep);
        background: rgba(13,8,32,0.2);
        border-color: rgba(13,8,32,0.4);
    }

    /* Sliding catalog panel */
    .catalog {
        position: fixed; top: 56px; right: 0; bottom: 0;
        width: min(420px, 92vw);
        background: var(--panel);
        border-left: 1px solid var(--gold-dim);
        transform: translateX(100%);
        transition: transform 0.3s ease;
        z-index: 1200;
        display: flex; flex-direction: column;
    }
    .catalog.open { transform: translateX(0); }
    .catalog-head {
        padding: 18px 20px 14px;
        border-bottom: 1px solid rgba(201,168,76,0.25);
        display: flex; align-items: center; gap: 12px;
    }
    .catalog-head h2 {
        flex: 1;
        font-family: var(--font-serif);
        color: var(--gold-light);
        font-size: 1.3rem;
        font-style: italic;
        letter-spacing: 0.04em;
    }
    .catalog-head .close-btn {
        background: transparent;
        border: 1px solid var(--gold-dim);
        color: var(--gold);
        font-size: 1.1rem;
        width: 32px; height: 32px; border-radius: 50%;
        cursor: pointer;
        transition: all 0.2s;
    }
    .catalog-head .close-btn:hover { background: var(--gold); color: var(--deep); }
    .catalog-body {
        flex: 1; overflow-y: auto;
        padding: 18px 20px 30px;
        display: flex; flex-direction: column; gap: 12px;
    }
    .catalog-empty {
        color: var(--text-dim);
        font-size: 0.9rem;
        text-align: center;
        padding: 40px 20px;
    }
    .city-card {
        background: var(--surface);
        border: 1px solid rgba(201,168,76,0.2);
        border-radius: 10px;
        padding: 14px 16px;
        text-decoration: none;
        color: inherit;
        display: flex; gap: 12px;
        transition: all 0.2s;
    }
    .city-card:hover {
        border-color: var(--gold);
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(201,168,76,0.15);
    }
    .city-icon {
        font-size: 1.6rem; line-height: 1;
        margin-top: 2px; flex-shrink: 0;
    }
    .city-meta { flex: 1; min-width: 0; }
    .city-name {
        font-family: var(--font-serif);
        font-size: 1.05rem;
        color: var(--gold-light);
        line-height: 1.2; margin-bottom: 4px;
    }
    .city-subtitle {
        font-size: 0.78rem;
        color: var(--text-dim);
        line-height: 1.4; margin-bottom: 8px;
    }
    .city-attr-count {
        display: inline-block;
        font-size: 0.66rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--teal);
        border: 1px solid rgba(42,181,160,0.4);
        padding: 2px 8px; border-radius: 3px;
    }

    @media (max-width: 480px) {
        .topbar { gap: 8px; padding: 10px 14px; }
        .topbar h1 { font-size: 1rem; }
        .topbar .stats { display: none; }
        .topbar a.back { padding: 4px 8px; font-size: 0.7rem; }
        #map { inset: 50px 0 0 0; }
        .catalog { top: 50px; }
    }
</style>
</head>
<body>
    <div class="topbar">
        <a class="back" href="../">← Inicio</a>
        <h1>Pepe Maps · <span class="accent">Explorador</span></h1>
        <span class="stats"><?= count($states) ?> estados · <?= $totalCities ?> ciudades · <?= $totalAttractions ?> atracciones</span>
    </div>

    <div id="map"></div>

    <aside class="catalog" id="catalog" aria-hidden="true">
        <div class="catalog-head">
            <h2 id="catalog-title">Selecciona un estado</h2>
            <button class="close-btn" id="catalog-close" aria-label="Cerrar catálogo">×</button>
        </div>
        <div class="catalog-body" id="catalog-body">
            <p class="catalog-empty">Haz clic en un estado del mapa para ver sus ciudades.</p>
        </div>
    </aside>

<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script>
    const STATES = <?= json_encode(array_values($states), JSON_HEX_TAG | JSON_HEX_AMP | JSON_UNESCAPED_UNICODE) ?>;

    const map = L.map('map', {
        zoomControl: true,
        attributionControl: false,
        worldCopyJump: false,
    }).setView([23.6, -102.5], 5);

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        maxZoom: 12,
        minZoom: 4,
    }).addTo(map);

    const catalog       = document.getElementById('catalog');
    const catalogTitle  = document.getElementById('catalog-title');
    const catalogBody   = document.getElementById('catalog-body');
    const catalogClose  = document.getElementById('catalog-close');

    catalogClose.addEventListener('click', () => closeCatalog());

    function closeCatalog() {
        catalog.classList.remove('open');
        catalog.setAttribute('aria-hidden', 'true');
    }

    function openStateCatalog(state) {
        catalogTitle.textContent = state.name;
        catalogBody.innerHTML = '';
        const cities = [...state.cities].sort((a, b) => a.name.localeCompare(b.name, 'es'));
        for (const c of cities) {
            const a = document.createElement('a');
            a.className = 'city-card';
            a.href = `../${state.slug}/${c.slug}/`;
            a.innerHTML = `
                <div class="city-icon">${c.icon}</div>
                <div class="city-meta">
                    <div class="city-name"></div>
                    <div class="city-subtitle"></div>
                    <span class="city-attr-count"></span>
                </div>`;
            a.querySelector('.city-name').textContent = c.name;
            a.querySelector('.city-subtitle').textContent = c.subtitle;
            a.querySelector('.city-attr-count').textContent =
                `${c.attrCount} ${c.attrCount === 1 ? 'atracción' : 'atracciones'}`;
            catalogBody.appendChild(a);
        }
        catalog.classList.add('open');
        catalog.setAttribute('aria-hidden', 'false');
    }

    for (const state of STATES) {
        const cityCount = state.cities.length;
        const html = `<div class="state-pin">${state.name}<span class="badge">${cityCount}</span></div>`;
        const icon = L.divIcon({
            className: 'state-pin-wrapper',
            html,
            iconSize: null,
            iconAnchor: [0, 0],
        });
        const marker = L.marker([state.lat, state.lng], { icon, riseOnHover: true }).addTo(map);
        marker.on('click', () => {
            openStateCatalog(state);
            map.flyTo([state.lat, state.lng], Math.max(map.getZoom(), 6), { duration: 0.6 });
        });
    }
</script>
</body>
</html>
