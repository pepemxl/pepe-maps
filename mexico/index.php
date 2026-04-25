<?php
// Landing page: lists all cities defined in cities.php, grouped by state.
$cities = require __DIR__ . '/cities.php';

header('Content-Type: text/html; charset=UTF-8');

function url_segment($s) { return rawurlencode($s); }

// Group cities by state, preserving first-seen order of states and cities.
$byState = [];
foreach ($cities as $slug => $c) {
    $stateSlug = $c['state_slug'] ?? 'sin-estado';
    $stateName = $c['state']      ?? 'Sin estado';
    if (!isset($byState[$stateSlug])) {
        $byState[$stateSlug] = ['name' => $stateName, 'cities' => []];
    }
    $byState[$stateSlug]['cities'][$slug] = $c;
}
// Sort state groups alphabetically by display name (Spanish locale-friendly enough).
uasort($byState, fn($a, $b) => strcoll($a['name'], $b['name']));

$totalAttractions = 0;
foreach ($cities as $c) { $totalAttractions += count($c['attractions'] ?? []); }
?>
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="theme-color" content="#1a0a2e">
<title>Pepe Maps · 50 ciudades de México</title>
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
        --teal: #2ab5a0;
        --font-serif: 'Playfair Display', Georgia, serif;
        --font-body: 'Crimson Pro', 'Times New Roman', serif;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        font-family: var(--font-body);
        background: var(--deep);
        color: var(--text);
        min-height: 100vh;
        background-image:
            radial-gradient(circle at 10% 10%, rgba(201,168,76,0.08) 0%, transparent 50%),
            radial-gradient(circle at 90% 80%, rgba(42,181,160,0.05) 0%, transparent 50%);
    }
    header {
        text-align: center;
        padding: 60px 20px 30px;
        border-bottom: 1px solid var(--gold-dim);
    }
    header h1 {
        font-family: var(--font-serif);
        font-size: 2.6rem;
        color: var(--gold-light);
        letter-spacing: 0.05em;
        font-style: italic;
    }
    header h1 .accent { color: var(--teal); font-style: normal; font-weight: 700; }
    header p {
        margin-top: 12px;
        color: var(--text-dim);
        font-size: 1rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
    }
    header .count {
        margin-top: 18px;
        display: inline-block;
        padding: 6px 14px;
        border: 1px solid var(--gold-dim);
        border-radius: 4px;
        color: var(--gold);
        font-size: 0.85rem;
        letter-spacing: 0.1em;
    }
    main {
        max-width: 1200px;
        margin: 0 auto;
        padding: 40px 20px 80px;
    }
    .state-section { margin-bottom: 48px; }
    .state-heading {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 16px;
        margin: 0 0 18px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(201,168,76,0.25);
    }
    .state-heading h2 {
        font-family: var(--font-serif);
        font-size: 1.4rem;
        color: var(--gold-light);
        letter-spacing: 0.04em;
        font-style: italic;
    }
    .state-heading a {
        color: var(--teal);
        text-decoration: none;
        font-size: 0.78rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        border: 1px solid rgba(42,181,160,0.4);
        padding: 4px 10px;
        border-radius: 3px;
        transition: all 0.2s;
    }
    .state-heading a:hover { background: rgba(42,181,160,0.15); border-color: var(--teal); }
    .state-heading .state-count {
        font-size: 0.78rem;
        color: var(--text-dim);
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
        gap: 18px;
    }
    .city-card {
        background: var(--surface);
        border: 1px solid rgba(201,168,76,0.2);
        border-radius: 10px;
        padding: 22px 20px;
        text-decoration: none;
        color: inherit;
        transition: all 0.25s ease;
        position: relative;
        overflow: hidden;
        display: flex;
        gap: 14px;
        align-items: flex-start;
    }
    .city-card::before {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, rgba(201,168,76,0.05), transparent 60%);
        pointer-events: none;
    }
    .city-card:hover {
        border-color: var(--gold);
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(201,168,76,0.15);
    }
    .city-icon {
        font-size: 2rem;
        flex-shrink: 0;
        line-height: 1;
        margin-top: 4px;
    }
    .city-meta { flex: 1; min-width: 0; }
    .city-name {
        font-family: var(--font-serif);
        font-size: 1.15rem;
        color: var(--gold-light);
        line-height: 1.2;
        margin-bottom: 6px;
    }
    .city-subtitle {
        font-size: 0.82rem;
        color: var(--text-dim);
        line-height: 1.4;
        margin-bottom: 10px;
    }
    .city-attr-count {
        display: inline-block;
        font-size: 0.7rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--teal);
        border: 1px solid rgba(42,181,160,0.4);
        padding: 3px 8px;
        border-radius: 3px;
    }
    footer {
        text-align: center;
        padding: 30px 20px 60px;
        color: var(--text-dim);
        font-size: 0.78rem;
        letter-spacing: 0.1em;
    }
    @media (max-width: 480px) {
        header h1 { font-size: 1.9rem; }
        main { padding: 24px 14px 60px; }
        .state-heading { flex-wrap: wrap; }
    }
</style>
</head>
<body>
    <header>
        <h1>Pepe Maps · <span class="accent">México</span></h1>
        <p>Guías turísticas interactivas con rutas a pie</p>
        <span class="count"><?= count($byState) ?> estados · <?= count($cities) ?> ciudades · <?= $totalAttractions ?> atracciones</span>
    </header>

    <main>
        <?php foreach ($byState as $stateSlug => $group):
            $stateName = htmlspecialchars($group['name']);
            $stateCount = count($group['cities']);
        ?>
        <section class="state-section" id="<?= htmlspecialchars($stateSlug) ?>">
            <div class="state-heading">
                <h2><?= $stateName ?></h2>
                <span class="state-count"><?= $stateCount ?> ciudad<?= $stateCount === 1 ? '' : 'es' ?></span>
                <a href="<?= url_segment($stateSlug) ?>/">Ver estado →</a>
            </div>
            <div class="grid">
                <?php foreach ($group['cities'] as $slug => $c):
                    $name = htmlspecialchars($c['name']);
                    $sub  = htmlspecialchars($c['subtitle'] ?? '');
                    $icon = $c['icon'] ?? '🏛️';
                    $attrCount = count($c['attractions'] ?? []);
                ?>
                <a class="city-card" href="<?= url_segment($stateSlug) ?>/<?= url_segment($slug) ?>/">
                    <div class="city-icon"><?= $icon ?></div>
                    <div class="city-meta">
                        <div class="city-name"><?= $name ?></div>
                        <?php if ($sub): ?><div class="city-subtitle"><?= $sub ?></div><?php endif; ?>
                        <span class="city-attr-count"><?= $attrCount ?> atracciones</span>
                    </div>
                </a>
                <?php endforeach; ?>
            </div>
        </section>
        <?php endforeach; ?>
    </main>

    <footer>
        Pepe Maps · Open data &middot; Mapas con OpenStreetMap &middot; Rutas con OSRM
    </footer>
</body>
</html>
